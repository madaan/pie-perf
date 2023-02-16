"""
Runs evaluation on the model generated outputs. The rough flow is as follows:

1. Reads the inputs and model generated programs, writes them to a temporary directory.
2. Runs each program several times and computes the average time taken and accuracy.
3. Writes the results to a json file.

Sample usage:
export REF_FILE="data/codenet/splits/problem_id/2023-01-13_12-56pm/seq2seq_splits/test.jsonl" && export OP="/usr1/amadaan/learning2perf/data/outputs/beam_outputs_codegen_2b_jan_13_split.jsonl" && export CPU=30 && nohup python -u src/codenet_eval/run_eval.py  --model_generated_outputs_path ${OP} --reference_file_path ${REF_FILE} --output_report_file_path ${OP}.25_tries.report --slow_code_col input --model_generated_potentially_faster_code_col beam_generated_target_from_input --reference_code_col target --num_problems_to_evaluate -1 --cpu_number $CPU --max_time_per_run 10 --num_trials 25 --ignore_first_k 2

"""
import pathlib
import tempfile
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple
import os
import logging
import glob
import numpy as np
from collections import defaultdict

from src.codenet_eval.evalconfig import EvaluationConfig
from src.codenet_eval.sandbox import run_code_on_inputs

import pdb

logging.basicConfig(level=logging.CRITICAL)

lang2file_ending = {
    "python": "py", 
    "cpp": "cpp"
}
    


    

def evaluate_generated_outputs(cfg: EvaluationConfig) -> None:
    """Evaluates model generated programs for accuracy and runtime. See the documenation of the EvaluationConfig class for more details."""

    # Step 0
    merged = read_inputs_and_prepare(cfg)
    logging.info(f"Number of programs to evaluate: {len(merged)}")
    logging.info(f"Number of trials per program: {cfg.num_trials}")
    logging.info(f"Number of trials to ignore: {cfg.ignore_first_k}")
    logging.info(f"Maximum time per run: {cfg.max_time_per_run}")
    logging.info(f"Input column: {cfg.slow_code_col}")
    logging.info(f"Reference column: {cfg.reference_code_col}")
    logging.info(f"Model generated column: {cfg.model_generated_potentially_faster_code_col}")
    logging.info(f"inputs/outputs basepath: {cfg.inputs_outputs_basepath}")

    # Step 1: Write the inputs and model generated programs to a temporary directory
    problem_id_to_ground_truths, output_code_location = write_programs_read_ground_truth(
        cfg, merged
    )

    # Step 2: run the programs
    
    # (dataframe col, suffix for the file)
    
    lang_file_ending = lang2file_ending[cfg.language]
    tag_to_path = [
        ("input", f"_slow.{lang_file_ending}"),
        ("reference", f"_reference.{lang_file_ending}"),
    ]
    
    # check if there are multiple generations per input
    is_multigen = isinstance(merged[cfg.model_generated_potentially_faster_code_col].iloc[0], list)
    if is_multigen:
        
        num_generations = len(merged[cfg.model_generated_potentially_faster_code_col].iloc[0])
        tag_to_path.extend([(f"{cfg.model_generated_potentially_faster_code_col}_{i}", f"_maybe_faster_{i}.{lang_file_ending}") for i in range(num_generations)])

    else:
        tag_to_path.append((cfg.model_generated_potentially_faster_code_col, f"_maybe_faster_0.{lang_file_ending}"))
    
    results = run_programs(cfg, merged, problem_id_to_ground_truths, output_code_location, tag_to_path)
    
    if is_multigen:
        results = get_best_generation_per_submission(results, gen_col=cfg.model_generated_potentially_faster_code_col)
    
    # Step 3: summarize the results, write report
    print_summary(cfg, merged, results, gen_col=cfg.model_generated_potentially_faster_code_col)

    if isinstance(cfg.temp_dir, tempfile.TemporaryDirectory):
        cfg.temp_dir.cleanup()
        


def read_inputs_and_prepare(cfg) -> pd.DataFrame:
    """Reads the model generated output, the reference, joins them, and returns a dataframe with the merged data."""
    logging.info(f"Reading reference file from {cfg.reference_file_path}")
    logging.info(f"Reading model generated outputs from {cfg.model_generated_outputs_path}")

    logging.info(
        f"Running each program {cfg.num_trials} times, skipping the first {cfg.ignore_first_k} runs, and getting the input-output pairs from {cfg.inputs_outputs_basepath}"
    )
    
    gen_df = pd.read_json(
        cfg.model_generated_outputs_path, lines=True, orient="records"
    )
    
    # if file ends in .report, then it is a re-run. We can filter out the rows that have already been run and just return the new rows
    if cfg.model_generated_outputs_path.endswith(".report"):
        return _prepare_for_rerun(gen_df, cfg)
    
    logging.info(f"Read {len(gen_df)} rows from {cfg.model_generated_outputs_path}")
    if cfg.is_prompt_based:
        gen_df["slower_program"] = gen_df.apply(
            lambda x: get_input_from_prompt(x), axis=1
        )
    else:
        gen_df["slower_program"] = gen_df[cfg.slow_code_col].apply(lambda x: x.strip())
        


    if cfg.reference_file_path is not None:
        ref_df = pd.read_json(cfg.reference_file_path, lines=True, orient="records")
        ref_df["slower_program"] = ref_df["input"].apply(
            lambda x: x.strip().replace("\n\n\n\n\n", "")
            # TODO: remove this hack, which is needed because sometimes prompt-lib does not retain the 
            # entire input
        )
        
        logging.info(f"Unique inputs in reference: {len(ref_df['slower_program'].unique())}")
        gen_df["slower_program"] = gen_df[
            "slower_program"
        ].apply(lambda x: x.strip().replace("\n\n\n\n\n", ""))
        assert len(ref_df["submission_id_v0"].unique()) == len(
            ref_df
        ), "submission_id_v0 should be unique"

        merged = pd.merge(
            gen_df,
            ref_df,
            left_on="slower_program",
            right_on="slower_program",
            suffixes=("", "_ref"),
            how="inner",
        )


        merged = merged.drop_duplicates(subset=["slower_program"])
        
        assert abs(len(merged) - len(gen_df)) < 10, f"Merging should not lose too many rows! Check if the inputs are the same. Merge lost {len(gen_df) - len(merged)} rows. len(gen_df)={len(gen_df)}, len(merged)={len(merged)}"
    else:
        assert (
            cfg.reference_code_col in gen_df.columns
        ), f"Column {cfg.reference_code_col} not found in {cfg.model_generated_outputs_path}"
        merged = gen_df
        
        
        merged = merged[merged[cfg.slow_code_col] != merged[cfg.reference_code_col]]

    assert (
        len(merged) > 0
    ), f"{cfg.slow_code_col} and {cfg.reference_code_col} are the same for all programs"
    
    if cfg.num_problems_to_evaluate != -1:
        merged = merged[: cfg.num_problems_to_evaluate]
    
    
    # if the generated code is a list, then we have multiple generations per input. 
    # we add one column per generation
    if isinstance(merged[cfg.model_generated_potentially_faster_code_col].iloc[0], list):
        num_generations = len(merged[cfg.model_generated_potentially_faster_code_col].iloc[0])
        for i in range(num_generations):
            merged[f"{cfg.model_generated_potentially_faster_code_col}_{i}"] = merged[cfg.model_generated_potentially_faster_code_col].apply(lambda x: x[i])
    return merged


def _prepare_for_rerun(df: pd.DataFrame, cfg: EvaluationConfig) -> pd.DataFrame:
    acc_columns = {"generated_answers_acc", "generated_answer_acc"}
    acc_column = list(acc_columns.intersection(set(df.columns)))[0]
    logging.info("ALERT! THIS IS A RERUN")
    logging.info("Preparing for rerun...")
    logging.info(f"Found accuracy column: {acc_column}, {len(df)} rows")
    df = df[df[acc_column] > 0.99]
    # remove all columns that have 'mean', 'std', 'acc' in them just to be safe
    df = df[[c for c in df.columns if not any(x in c for x in ["mean", "std", "acc"])]]
    logging.info(f"Filtered to {len(df)} rows")
    if cfg.num_problems_to_evaluate != -1:
        df = df[: cfg.num_problems_to_evaluate]
    return df
    

def write_programs_read_ground_truth(
    cfg: EvaluationConfig, merged: pd.DataFrame
) -> Tuple[Dict[str, List[str]], str]:
    # Writes all the programs to a temp directory, load ground truth
    # we don't want to do I/O repeatedly as it adds to the variance
    problem_id_to_ground_truths = defaultdict(list)
    if cfg.temp_dir is None:
        cfg.temp_dir = tempfile.TemporaryDirectory()
        output_code_location = cfg.temp_dir.name
    else:
        output_code_location = cfg.temp_dir
        pathlib.Path(output_code_location).mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="writing programs"):
        problem_id = row["problem_id"]

        # read the ground truth

        if problem_id not in problem_id_to_ground_truths:
            num_test_cases = len(
                glob.glob(f"{cfg.inputs_outputs_basepath}/{problem_id}/output*.txt")
            )
            assert (
                num_test_cases > 0
            ), f"{cfg.inputs_outputs_basepath}/{problem_id} has no ground truth files!"
            for i in range(num_test_cases):
                with open(f"{cfg.inputs_outputs_basepath}/{problem_id}/output.{i}.txt") as f:
                    problem_id_to_ground_truths[problem_id].append(f.read().strip() + "\n")

        # write both generated and reference programs to the temp directory

        lang_file_ending = lang2file_ending[cfg.language]
        submission_id_v0 = row["submission_id_v0"]
        with open(
            os.path.join(output_code_location, f"{submission_id_v0}_{problem_id}_slow.{lang_file_ending}"), "w"
        ) as f:
            ## This change in order to keep the comments out
            f.write(row["slower_program"])
            # f.write(row[cfg.slow_code_col].strip())

        # to deal with the case where there are multiple generated programs
        generated_programs = row[cfg.model_generated_potentially_faster_code_col]
        if isinstance(generated_programs, str):
            generated_programs = [generated_programs]
        
        for i, generated_program in enumerate(generated_programs):
            with open(
                os.path.join(output_code_location, f"{submission_id_v0}_{problem_id}_maybe_faster_{i}.{lang_file_ending}"),
                "w"
            ) as f:
                f.write(generated_program.strip())

        with open(
            os.path.join(output_code_location, f"{submission_id_v0}_{problem_id}_reference.{lang_file_ending}"), "w"
        ) as f:
            f.write(row[cfg.reference_code_col].strip())

    logging.info(f"finished writing programs to {output_code_location}")
    return problem_id_to_ground_truths, output_code_location


def run_programs(
    cfg: EvaluationConfig,
    merged: pd.DataFrame,
    problem_id_to_ground_truths: Dict,
    output_code_location: str,
    tag_to_path
):

    """Actually runs the programs.

    Args:
        cfg (EvaluationConfig): The evaluation config
        merged (pd.DataFrame): Dataframe with merged
        problem_id_to_ground_truths (Dict): ground truths for each problem id. These are compared with the output of the program
        output_code_location (str): the directory where the programs are written.

    Returns:
        _type_: _description_
    """

    results = dict()
    # NOTE: every row has a unique submission_id_v0, so we can use that as the submission_id
    # This is because for three submissions A, B, C, we create two pairs (A, B) and (B, C).
    # If we change this to also include (A, C), then we need to change the logic here. The following
    # assert checks that this is the case.
    assert len(merged["submission_id_v0"].unique()) == len(
        merged
    ), f"Every row should have a unique submission_id_v0. This is not the case: number of unique submission_id_v0: {len(merged['submission_id_v0'].unique())}, number of rows: {len(merged)}"

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="running programs"):
        problem_id = row["problem_id"]
        submission_id_v0 = row["submission_id_v0"]
        unit_test_data_basepath = f"{cfg.inputs_outputs_basepath}/{problem_id}"
        try:
            problem_execution_stats = dict()
            # run the generated program (maybe faster), input program (slower), and reference program (definitely faster)
            for (tag, suffix) in tag_to_path:
                code_path = os.path.join(
                    output_code_location, f"{submission_id_v0}_{problem_id}{suffix}"
                )

                logging.info(
                    f"running {tag} program for problem {problem_id}, submission {submission_id_v0}"
                )
                
                avg_time, std_time, avg_acc = run_code_on_inputs(  # type: ignore
                    language=cfg.language,
                    code_path=code_path,
                    ground_truths=problem_id_to_ground_truths[problem_id],
                    unit_test_data_basepath=unit_test_data_basepath,
                    num_runs_per_test_case=cfg.num_trials,
                    ignore_first_k=cfg.ignore_first_k,
                    max_seconds_per_run=cfg.max_time_per_run,
                    cpu_number=cfg.cpu_number,
                    cflags=cfg.cflags,
                    return_if_acc_below=cfg.return_if_acc_below,
                )

                problem_execution_stats.update(
                    {
                        f"{tag}_time_mean": avg_time,
                        f"{tag}_time_std": std_time,
                        f"{tag}_acc": avg_acc,
                    }
                )
            results[submission_id_v0] = problem_execution_stats

        except Exception as e:
            logging.error(e)
            tmp = dict()
            for tag, suffix in tag_to_path:
                tmp[f"{tag}_time_mean"] = np.nan
                tmp[f"{tag}_time_std"] = np.nan
                tmp[f"{tag}_acc"] = 0.0
            results[submission_id_v0] = tmp
            continue

    logging.info(f"Ran for {len(results)} problems")
    return results


def get_best_generation_per_submission(results: Dict, gen_col: str):
    """Given the results, get the best generation for each submission which is also correct.
    The best is defined as the generation with the lowest time.

    Args:
        results (Dict): results of running the programs
        gen_col (str): the column name for the generation

    Returns:
        Dict: the best generation for each submission
    """
    best_per_sub = dict()
    for submission_id_v0, result_dict in results.items():
        gen_op_times = [(k, v) for k, v in result_dict.items() if gen_col in k and "time_mean" in k]
        gen_op_times = sorted(gen_op_times, key=lambda x: x[1])
        
        # itearte and find the first generation that is correct
        for gen_op_time in gen_op_times:
            if result_dict[f"{gen_op_time[0].replace('_time_mean', '')}_acc"] == 1.0:
                gen_op_times = [gen_op_time]
                break
        # find out which generation is the best
        try: 
            best_gen_key = gen_op_times[0][0].replace("_time_mean", "")
            best_per_sub[submission_id_v0] = result_dict
            best_per_sub[submission_id_v0][f"{gen_col}_time_mean"] = gen_op_times[0][1]
            best_per_sub[submission_id_v0][f"{gen_col}_time_std"] = result_dict[f"{best_gen_key}_time_std"]
            best_per_sub[submission_id_v0][f"{gen_col}_acc"] = result_dict[f"{best_gen_key}_acc"]
        except IndexError:
            pdb.set_trace()

    return best_per_sub

def print_summary(cfg, merged, results, gen_col: str):
    report_rows = []
    for _, row in tqdm(merged.iterrows(), total=len(merged)):
        submission_id_v0 = row["submission_id_v0"]

        if submission_id_v0 not in results:
            continue

        report_row = row.to_dict()

        report_row.update(results[submission_id_v0])
        report_rows.append(report_row)

    assert len(results) == len(report_rows)
    logging.info(f"Writing report to {cfg.output_report_file_path} with {len(report_rows)} rows")
    run_metrics = pd.DataFrame(report_rows)
    
    # drop na
    # run_metrics = run_metrics.dropna(how="any")
    run_metrics.to_json(cfg.output_report_file_path, orient="records", lines=True)

    run_metrics = run_metrics[
        (run_metrics[f"{gen_col}_acc"] > 0.99) & (run_metrics["input_acc"] > 0.99)
    ]
    if run_metrics.empty:
        return

    logging.info("---Execution time---")
    logging.info(
        f"[Reported in CodeNet] input program (ms): {mean_std(run_metrics, 'cpu_time_v0')}"
    )
    logging.info(
        f"[Reported in CodeNet] reference (output) program (ms): {mean_std(run_metrics, 'cpu_time_v1')}"
    )

    logging.info("-" * 80)
    logging.info(f"[Our measurement] input program (ms): {mean_std(run_metrics, 'input_time')}")
    logging.info(
        f"[Our measurement] reference (output) program (ms): {mean_std(run_metrics, 'reference_time')}"
    )
    logging.info(
        f"[Our measurement] {gen_col} program (ms): {mean_std(run_metrics, f'{gen_col}_time')}"
    )

    run_metrics_improved = run_metrics[
        run_metrics[f"{gen_col}_time_mean"] < run_metrics["reference_time_mean"]
    ]
    if len(run_metrics_improved) > 0:
        logging.info("----Metrics when improved--")
        logging.info(
            f"Found {len(run_metrics_improved)} problems where the {gen_col} program is faster than the input program"
        )
        logging.info(
            f"[Our measurement] input program (ms): {mean_std(run_metrics_improved, 'input_time')}"
        )
        logging.info(
            f"[Our measurement] reference (output) program (ms): {mean_std(run_metrics_improved, 'reference_time')}"
        )
        logging.info(
            f"[Our measurement] {gen_col} program (ms): {mean_std(run_metrics_improved, f'{gen_col}_time')}"
        )
    logging.info(
        f"Number of cases where reference took longer by our measurement: {len(get_anomalies(run_metrics))}"
    )


def mean_std(df, col) -> str:
    mean_col = f"{col}_mean"
    std_col = f"{col}_std"
    if mean_col not in df.columns or std_col not in df.columns:
        return f"{df[col].mean():.4f} ± {df[col].std():.4f}"

    return f"{df[mean_col].mean():.4f} ± {df[std_col].mean():.4f}"


def get_anomalies(run_metrics):
    run_metrics["codenet_reported_rel_improvement"] = (
        run_metrics["cpu_time_v0"] - run_metrics["cpu_time_v1"]
    ) / run_metrics["cpu_time_v0"]
    run_metrics["codenet_reported_rel_improvement"] = run_metrics[
        "codenet_reported_rel_improvement"
    ].apply(lambda x: round(x * 100, 2))
    run_metrics["measured_rel_improvement"] = (
        run_metrics["input_time_mean"] - run_metrics["reference_time_mean"]
    ) / run_metrics["input_time_mean"]
    run_metrics["measured_rel_improvement"] = run_metrics["measured_rel_improvement"].apply(
        lambda x: round(x * 100, 2)
    )
    run_metrics["is_anomaly"] = run_metrics.apply(
        lambda x: x["codenet_reported_rel_improvement"] > 10 and x["measured_rel_improvement"] < 0,
        axis=1,
    )
    run_metrics_anomalies = run_metrics[run_metrics["is_anomaly"]]
    return run_metrics_anomalies


def get_input_from_prompt(
    row: pd.Series,
    question_sep: str = "# slower version:",
    answer_sep: str = "# optimized version of the same code:",
) -> str:
    
    if "entire_prompt" in row:
        prompt_str = row["entire_prompt"]
    else:
        prompt_str = row["prompt"] + row["question"]
    prompt_str = prompt_str.replace("\n\n\n\n\n", "")
    return prompt_str.split(question_sep)[-1].split(answer_sep)[0].strip()


if __name__ == "__main__":

    args = EvaluationConfig.get_args()
    args.add_argument("--eval_config", type=str, required=False)
    args = args.parse_args()

    if args.eval_config is not None:
        evaluation_config = EvaluationConfig.from_yaml(args.eval_config)
    else:
        evaluation_config = EvaluationConfig.from_args(args)

    evaluate_generated_outputs(evaluation_config)
