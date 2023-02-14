"""Runs time evaluation for all submissions in the training file.
See the issue #1 on Github for more details. Essentially, we want to re-run the time evaluation.

Reads the global metadata file that has the code, runs each submission on public and generated test cases, and records the time.
user_id problem_id      language        submission_id_v0        submission_id_v1        cpu_time_v0     cpu_time_v1     memory_v0       memory_v1       status_v0       status_v1      improvement_frac        code_v0 code_v1 code_v0_loc     code_v1_loc

"""
from tqdm import tqdm
import joblib
import pandas as pd
import logging

from src.utils.parallel_utils import tqdm_joblib

logging.basicConfig(level=logging.INFO)

from src.codenet_eval.sandbox import run_python_code_on_inputs
from src.codenet_eval.eval_submissions.prep_for_eval import write_code_for_eval

public_cases_basedir = "data/codenet/public_test_cases/"
generated_test_cases_basedir = "data/codenet/generated_test_cases/"
temp_dir = "/tmp/codenet_eval/"

# get the count of number of CPU cores
import multiprocessing

core_numbers = list(range(5, multiprocessing.cpu_count() + 1))
num_cores_avail = len(core_numbers)




def eval_code(pairs_dataset_path: str):
    df = pd.read_csv(pairs_dataset_path, sep="\t")
    submission_id_to_metadata = write_code_for_eval(df, basedir=temp_dir)

    problem_id_to_num_test_cases = load_num_test_cases(submission_id_to_metadata)

    i = 0
    
    args = []
    for submission_id, metadata in tqdm(
        submission_id_to_metadata.items(),
        total=len(submission_id_to_metadata),
        desc="Preparing submissions for evaluation",
    ):
        args.append((submission_id,
            f"{temp_dir}/{metadata['problem_id']}/{submission_id}/{submission_id}.py",
            problem_id_to_num_test_cases,
            metadata['problem_id'],
            core_numbers[i % num_cores_avail],
            i + 1
        ))
        
        i += 1


    global total_submissions
    total_submissions = len(args)
    # parallelize the evaluation, collect the results
    with tqdm_joblib(tqdm(desc="Evaluating submissions", total=total_submissions)) as progress_bar:
        results = joblib.Parallel(n_jobs=num_cores_avail)(joblib.delayed(run_submission)(*arg) for arg in args)
    
    # convert the results to a dictionary
    results_dict = {}
    for result in results:
        results_dict.update(result)
    
    return results_dict
    

def load_num_test_cases(submission_id_to_metadata):
    from glob import glob

    def _get_num_test_cases(test_cases_basedir, problem_id):
        return len(glob(f"{test_cases_basedir}/{problem_id}/input*.txt"))

    problem_id_to_num_test_cases = {}
    for _, metadata in submission_id_to_metadata.items():
        problem_id = metadata["problem_id"]
        problem_id_to_num_test_cases[problem_id] = {}
        problem_id_to_num_test_cases[problem_id]["public"] = _get_num_test_cases(
            public_cases_basedir, problem_id
        )
        problem_id_to_num_test_cases[problem_id]["generated"] = _get_num_test_cases(
            generated_test_cases_basedir, problem_id
        )
    return problem_id_to_num_test_cases

num_submissions_processed = 0
total_submissions = 0

def run_submission(
    submission_id: str,
    submission_code_path: str,
    problem_id_to_num_test_cases: dict,
    problem_id: str,
    cpu_code_binary: str,
    job_number: int = 0,
):

    """Runs the submission on both public and generated test cases.
    Returns a dict with the following"""

    # defaults here to simplify joblib
    ignore_first_k = 0
    num_trials_per_test = 5
    max_seconds_per_run = 5

    try:
        public_per_trial_times = run_python_code_on_inputs(
            code_path=submission_code_path,
            ignore_first_k=ignore_first_k,
            max_seconds_per_run=max_seconds_per_run,
            unit_test_data_basepath=f"{public_cases_basedir}/{problem_id}",
            num_test_cases=problem_id_to_num_test_cases[problem_id]["public"],
            num_runs_per_test_case=num_trials_per_test,
            return_per_trial_times=True,
            cpu_number=cpu_code_binary,
        )
        
        generated_per_trial_times = run_python_code_on_inputs(
            code_path=submission_code_path,
            ignore_first_k=ignore_first_k,
            max_seconds_per_run=max_seconds_per_run,
            unit_test_data_basepath=f"{generated_test_cases_basedir}/{problem_id}",
            num_test_cases=problem_id_to_num_test_cases[problem_id]["generated"],
            num_runs_per_test_case=num_trials_per_test,
            return_per_trial_times=True,
            cpu_number=cpu_code_binary,

        )
        
        
        with open(f"data/runtime_eval/{submission_id}.pkl", "wb") as f:
            pickle.dump({submission_id: {"public": public_per_trial_times, "generated": generated_per_trial_times}}, f)
        return {submission_id: {"public": public_per_trial_times, "generated": generated_per_trial_times}} 
    except Exception as e:
        print(f"Error in submission {submission_id}: {e}")
        return {submission_id: {"public": None, "generated": None}}


if __name__ == "__main__":
    
    import pickle
    import warnings
    import sys
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output = eval_code(sys.argv[1])
    
        with open("data/all_submissions_time_eval.pkl", "wb") as f:
            pickle.dump(output, f)