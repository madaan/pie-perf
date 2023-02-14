"""Generates train/test/val splits for the codenet.
The splits are made on two columns: problem_id, and user_id
"""
import datetime
import pathlib
from typing import List, Dict, Optional
import numpy as np
import sys
import json
import os
import pandas as pd
from pandarallel import pandarallel
import multiprocessing

pandarallel.initialize(progress_bar=False, nb_workers=multiprocessing.cpu_count())

from src.utils.format_utils import clean_code
from src.utils.diff_utils import get_minimal_diff
from src.utils.format_utils import convert_2to3
from src.utils.name_utils import standardize_lang_name


def run_split_gen(data, basedir: str, lang: str = None, test_size=0.05):

    # in data, rename code_v0 to input and code_v1 to target

    train, test = make_train_test_split_on_problem_id(data, test_size=test_size)
    while len(test) < 1000:
        test_size += 0.01
        print(f"Increasing test size to {test_size:.2f} to get 1000 test examples for {lang}")
        if test_size > 0.75:
            raise ValueError(f"Could not get 1000 test examples, even with test_size={test_size}")
        train, test = make_train_test_split_on_problem_id(data, test_size=test_size)

    sanity_check(train, test)

    # print the size of splits
    print(f"Lang: {lang}, Train: {len(train)}, Test: {len(test)}, Val: {len(test)}")

    train, val = make_train_test_split_on_problem_id(train)
    sanity_check(train, val)
    test_1k = test.sample(n=1000, random_state=0)

    
    outpath = f"{basedir}/codenet-" if lang is None else f"{basedir}/codenet-{lang}-"
    
    train.to_json(f"{outpath}train.jsonl", orient="records", lines=True)
    test.to_json(f"{outpath}test.jsonl", orient="records", lines=True)
    val.to_json(f"{outpath}val.jsonl", orient="records", lines=True)
    test_1k.to_json(f"{outpath}test-1k.jsonl", orient="records", lines=True)



def make_train_test_split_on_problem_id(df, test_size=0.05):
    """Make train/test split on problem_id."""
    np.random.seed(0)
    problem_ids = df["problem_id"].unique()
    test_problem_ids = np.random.choice(
        problem_ids, size=int(len(problem_ids) * test_size), replace=False
    )
    train = df[~df["problem_id"].isin(test_problem_ids)]
    test = df[df["problem_id"].isin(test_problem_ids)]
    return train, test


def sanity_check(train_df, test_df):
    train_problem_ids = train_df["problem_id"].unique()
    test_problem_ids = test_df["problem_id"].unique()
    train_user_ids = train_df["user_id"].unique()
    test_user_ids = test_df["user_id"].unique()
    assert len(set(train_problem_ids).intersection(set(test_problem_ids))) == 0


def read_and_filter_pairs(
    path: str,
    filters_kwargs,
    submission_id_to_runtime_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:  # type: ignore
    """Reads the pairs file and filters out the pairs:
    1. Are duplicates
    2. Are not accepted
    Args:
        path (str): the path to the pairs file. Each row is a pair of code snippets, with metadata.
        submission_id_to_runtime_map (dict): a map from submission_id to runtime. Introduced to filter out pairs with a runtime difference of less than 1.25x. The runtime information is measured with our own runtime measurement tool.

    Returns:
        pd.DataFrame: filtered dataframe
    """

    data = pd.read_csv(path, sep="\t")
    data['language'] = data['language'].apply(standardize_lang_name)
    # only for Python, run the 2to3 tool

    filters = Filters(
        df=data, submission_id_to_runtime_map=submission_id_to_runtime_map, **filters_kwargs
    )
    filters.apply_filters()
    filtered_data = filters.df
    filtered_data.rename(columns={"code_v0": "input", "code_v1": "target"}, inplace=True)
    
    
    

    return filtered_data



class Filters(object):
    def __init__(
        self,
        df,
        min_our_runtime_lift: float,
        char_percentile_to_filter: float,
        max_loc: int,
        min_time_impro_perc: int,
        langs: List[str],
        max_rel_loc_diff: float,
        submission_id_to_runtime_map: Optional[Dict[str, float]],
    ):
        self.df = df
        self.char_percentile_to_filter = char_percentile_to_filter
        self.max_loc = max_loc
        self.min_time_impro_perc = min_time_impro_perc
        self.langs = langs
        self.max_rel_loc_diff = max_rel_loc_diff
        self.submission_id_to_runtime_map = submission_id_to_runtime_map
        self.min_our_runtime_lift = min_our_runtime_lift

    def _filter_same(self):
        self.df["code_v0"] = self.df.parallel_apply(
            lambda x: convert_2to3(x["code_v0"])
            if "python" in x["language"].lower()
            else x["code_v0"],
            axis=1,
        )
        self.df["code_v1"] = self.df.parallel_apply(
            lambda x: convert_2to3(x["code_v1"])
            if "python" in x["language"].lower()
            else x["code_v1"],
            axis=1,
        )
        self.df["code_v0_no_empty_lines"] = self.df.parallel_apply(
            lambda x: clean_code(x["code_v0"]), axis=1
        )
        self.df["code_v1_no_empty_lines"] = self.df.parallel_apply(
            lambda x: clean_code(x["code_v1"]), axis=1
        )
        # parallelize this

        self.df["code_same"] = self.df.apply(
            lambda x: x["code_v0_no_empty_lines"] == x["code_v1_no_empty_lines"], axis=1
        )
        self.df = self.df[~self.df["code_same"]]

    def _filter_min_our_runtime_lift(self):
        self.df["measured_runtime_v0"] = self.df["submission_id_v0"].apply(
            lambda sid: self.submission_id_to_runtime_map[sid]
            if sid in self.submission_id_to_runtime_map
            else -100
        )
        self.df["measured_runtime_v1"] = self.df["submission_id_v1"].apply(
            lambda sid: self.submission_id_to_runtime_map[sid]
            if sid in self.submission_id_to_runtime_map
            else -100
        )
        print(
            f"Found {self.df[self.df['measured_runtime_v0'] == -100].shape[0]} submissions without runtime out of {self.df.shape[0]}"
        )
        print(self.df[self.df["measured_runtime_v0"] == -100]["submission_id_v0"])
        self.df["runtime_lift"] = self.df.apply(
            lambda r: r["measured_runtime_v0"] / (r["measured_runtime_v1"] + 1e-11)
            if r["measured_runtime_v0"] > 0 and r["measured_runtime_v1"] > 0
            else 0,
            axis=1,
        )
        print(self.df["runtime_lift"].describe())
        self.df = self.df[self.df["runtime_lift"] > self.min_our_runtime_lift]

    def _filter_num_chars(self):
        # filters out cases where users have added zip files or other large files
        self.df["code_v0_num_chars"] = self.df["code_v0"].apply(lambda x: len(x))
        self.df["code_v1_num_chars"] = self.df["code_v1"].apply(lambda x: len(x))
        self.df = self.df[
            (
                self.df["code_v0_num_chars"]
                < np.percentile(self.df["code_v0_num_chars"], self.char_percentile_to_filter)
            )
        ]
        self.df = self.df[
            (
                self.df["code_v1_num_chars"]
                < np.percentile(self.df["code_v1_num_chars"], self.char_percentile_to_filter)
            )
        ]

    def _filter_unverified(self):
        def read_set_from_file(path):
            with open(path, "r") as f:
                return set(f.read().splitlines())

        # do not include solutions that could not be verified
        unverified_path = (
            "project_codenet/Project_CodeNet/derived/input_output/unverified_accepted_solutions.txt"
        )
        unverified = read_set_from_file(unverified_path)
        self.df = self.df[~self.df["problem_id"].isin(unverified)]

    def _filter_accepted(self):
        # v0 could have been TLE due to a bug. We want to focus on cases where both the versions passed test cases
        self.df = self.df[
            (self.df["status_v0"] == "Accepted") & (self.df["status_v1"] == "Accepted")
        ]

    def _filter_loc(self):
        # we don't want long code
        self.df = self.df[self.df.code_v0_loc < self.max_loc]

    def _filter_improvement(self):
        # we only want to keep cases where the improvement was > time_improvement_threshold%
        self.df = self.df[self.df.improvement_frac > self.min_time_impro_perc]

    def _filter_language(
        self,
    ):
        self.df = self.df[self.df["language"].isin(self.langs)]
        self.df["language"] = self.df["language"].apply(lambda x: x.replace("+", "p").lower())
        print(self.df["language"].value_counts())

    def _filter_relative_loc_diff(self):
        # we only want to keep cases where the relative difference in loc is < relative_size_threshold%. This is to discourage complete re-writes.

        self.df["relative_loc_diff_percent"] = self.df.apply(
            lambda x: abs(x["code_v0_loc"] - x["code_v1_loc"])
            / max(x["code_v0_loc"], x["code_v1_loc"]),
            axis=1,
        )
        self.df["relative_loc_diff_percent"] = self.df["relative_loc_diff_percent"].apply(
            lambda x: x * 100
        )
        self.df = self.df[self.df.relative_loc_diff_percent < self.max_rel_loc_diff]

    def _filter_only_import_comment_diff(self):

        # we only want to remove cases where the only diff is in the import statements or comments
        def _is_only_import_comment_diff(diff):
            for line in diff:
                line = line[1:].strip()
                if not (line.startswith("import") or line.startswith("#")):
                    return False
            return True

        self.df["diff"] = self.df.apply(
            lambda row: get_minimal_diff(
                row["code_v0_no_empty_lines"], row["code_v1_no_empty_lines"], return_lines=True
            ),
            axis=1,
        )
        self.df["diff_only_import_comment"] = self.df["diff"].apply(_is_only_import_comment_diff)
        self.df = self.df[~self.df["diff_only_import_comment"]]

    @staticmethod
    def get_loc_diff(loc_v0: int, loc_v1: int) -> float:
        """difference in loc in terms of %"""
        return abs(loc_v0 - loc_v1) / loc_v0

    def apply_filters(self):
        filter_to_name = {
            self._filter_language: "language filtering",
            self._filter_num_chars: "excessive code length",
            self._filter_unverified: "unverified solutions",
            self._filter_accepted: "accepted solutions only",
            self._filter_same: "identical code",
            self._filter_loc: f"LOC < {self.max_loc}",  # loc filtering
            self._filter_improvement: f"improvement > {self.min_time_impro_perc}%",
            self._filter_relative_loc_diff: f"relative loc diff < {self.max_rel_loc_diff}%",
            self._filter_only_import_comment_diff: "only import/comment diff",
        }

        if self.submission_id_to_runtime_map is not None:
            filter_to_name[self._filter_min_our_runtime_lift] = "runtime lift filtering"

        for (filter_func, filter_name) in filter_to_name.items():
            before_size = len(self.df)
            filter_func()
            perc_change = round(((before_size - len(self.df)) / before_size) * 100, 2)
            print(f"{filter_name}: {perc_change}% ({before_size} -> {len(self.df)})")


def read_submission_id_to_runtime_map(args) -> Dict[str, float]:
    with open(args.submission_id_to_runtime_map, "r") as f:
        print(f"Using runtime map: {args.submission_id_to_runtime_map}")
        submission_id_to_runtime_map = json.load(f)
        print(f"Loaded {len(submission_id_to_runtime_map)} entries.")
            # remove entries that map to nan
        submission_id_to_runtime_map_no_nans = {}
        for k, v in submission_id_to_runtime_map.items():
                # patched_submission_id_to_runtime_map[k] = v["public"]
            if np.isnan(v["all"]) and not np.isnan(v["public"]):
                submission_id_to_runtime_map_no_nans[k] = v["public"]
            elif not np.isnan(v["all"]):
                submission_id_to_runtime_map_no_nans[k] = v["all"]

        print(f"Removed nan entries. Now {len(submission_id_to_runtime_map)} entries.")
    return submission_id_to_runtime_map_no_nans


    
    

if __name__ == "__main__":

    import sys
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file_path", type=str, required=True)
    parser.add_argument("--submission_id_to_runtime_map", type=str, required=False, default=None)
    parser.add_argument(
        "--output_dir", type=str, default="data/codenet/splits/problem_id"
    )

    # add an option for each Filter argument
    parser.add_argument("--f_max_loc", type=int, required=False, default=150)
    parser.add_argument("--f_min_time_impro_perc", type=float, required=False, default=10.0)
    parser.add_argument("--f_max_rel_loc_diff", type=float, required=False, default=70.0)
    parser.add_argument("--f_langs", type=str, required=False, default="Python")
    parser.add_argument("--f_min_our_runtime_lift", type=float, required=False, default=1.0)
    parser.add_argument("--f_char_percentile_to_filter", type=float, required=False, default=99.5)

    args = parser.parse_args()

    args.f_langs = args.f_langs.split(",")
    args.f_langs = [standardize_lang_name(lang) for lang in args.f_langs]
    
    submission_id_to_runtime_map_no_nans = read_submission_id_to_runtime_map(args) if args.submission_id_to_runtime_map else None


    args.output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + datetime.datetime.now().strftime("%p").lower())
    print(f"Saving to {args.output_dir}")
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    data = read_and_filter_pairs(
        args.data_file_path,
        submission_id_to_runtime_map=submission_id_to_runtime_map_no_nans,
        filters_kwargs={k.replace('f_', ''): v for k, v in vars(args).items() if k.startswith("f_")},
    )
    
    # data.to_json(os.path.join(args.output_dir, "data.json"), orient="records", lines=True)
    
    for lang in args.f_langs:  # note we change C++ to cpp, and lower case

        run_split_gen(data[data["language"] == lang], lang=lang, basedir=args.output_dir)

    
    run_split_gen(data, basedir=args.output_dir)

    # save the arguments, the submission_id_to_runtime_map_no_nans, and the command for reproducibility
    if args.submission_id_to_runtime_map is not None:
        with open(os.path.join(args.output_dir, "submission_id_to_runtime_map.json"), "w") as f:
            json.dump(submission_id_to_runtime_map_no_nans, f)

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)

    with open(os.path.join(args.output_dir, "command.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv))
    
    with open(os.path.join(args.output_dir, "split_gen.py"), "w") as f:
        f.write(open(__file__).read())