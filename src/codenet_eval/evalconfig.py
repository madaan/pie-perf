from typing import Any
import yaml
import argparse
from dataclasses import dataclass


@dataclass
class EvaluationConfig:

    # where is the input/output
    model_generated_outputs_path: str  #  the file to be evaluated
    inputs_outputs_basepath: str  # the basepath of the input and output files for evaluating each problem. unit tests are located in ${inputs_outputs_basepath}/${problem_id}/{input.txt, output.txt}
    reference_file_path: str  # the reference file. Sometimes the reference file is not needed, when the outputs are located in the evaluation file
    output_report_file_path: str  #  the output report file path.
    
    # language
    language: str # programming language (e.g. python/cpp)

    # parameters for running evaluation
    num_problems_to_evaluate: int  # number of problems to evaluate. -1 means all problems.
    num_trials: int  # number of times to execute each program for a test case
    ignore_first_k: int  # ignore the first k runs
    max_time_per_run: int  # maximum time to allow each run to take
    temp_dir: Any  # temporary directory to use for writing the programs that are evaluated


    # the columns for input, output, reference
    model_generated_potentially_faster_code_col: str  # column in the output file that contains the model generated code
    slow_code_col: str  # column in the output file that contains the slow code
    reference_code_col: str  # column in the output file that contains the reference code
    is_prompt_based: bool  # whether the outputs are generated from a few-shot model

    cpu_number: int  # we use taskset to run the code on a specific cpu
    
    # optional parameters    
    return_if_acc_below: float = None #  type: ignore | if the accuracy is below this value, then return immediately
    cpp_results_path: str = None #  type: ignore | the path to the results of compiling and running the ref cpp code
    cflags: str = None #  type: ignore | the cflags to use for compiling the code
    

    @classmethod
    def from_args(cls, args):
        return cls(
            model_generated_outputs_path=args.model_generated_outputs_path,
            inputs_outputs_basepath=args.inputs_outputs_basepath,
            reference_file_path=args.reference_file_path,
            num_problems_to_evaluate=args.num_problems_to_evaluate,
            language=args.language, 
            num_trials=args.num_trials,
            ignore_first_k=args.ignore_first_k,
            max_time_per_run=args.max_time_per_run,
            temp_dir=args.temp_dir,
            model_generated_potentially_faster_code_col=args.model_generated_potentially_faster_code_col,
            slow_code_col=args.slow_code_col,
            reference_code_col=args.reference_code_col,
            is_prompt_based=args.is_prompt_based,
            output_report_file_path=args.output_report_file_path,
            cpu_number=args.cpu_number,
            cflags=args.cflags,
            cpp_results_path=args.cpp_results_path,
            return_if_acc_below=args.return_if_acc_below,
        )

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "EvaluationConfig":
        with open(yaml_file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationConfig":
        return cls(**data)
    
    def to_yaml(self) -> str:
        return yaml.dump(self.__dict__)
    
    @staticmethod
    def get_args() -> argparse.ArgumentParser:
        """Returns an ArgumentParser for the evaluation config."""
        args = argparse.ArgumentParser()

        args.add_argument("--reference_file_path", type=str)

        args.add_argument(
            "--model_generated_outputs_path",
            type=str,
            help="The output file path. It should be a jsonl file with outputs in `model_generated_potentially_faster_code_col`. See sample in `data/codenet/unit_test/output.jsonl`",
        )
        
        args.add_argument("--language", type=str, help="programmming language (e.g. 'python' or 'cpp')")

        args.add_argument(
            "--model_generated_output_col",
            type=str,
            default="generated_answer",
            help="Column name for the generated output",
        )

        args.add_argument(
            "--model_generated_potentially_faster_code_col",
            type=str,
            default="generated_answer",
            help="Column name for the generated output",
        )

        args.add_argument("--output_report_file_path", type=str, help="Where to write the report")
        
        args.add_argument(
            "--cpp_results_path", 
            type=str, 
            default=None, # only used for cpp
            help="Where to write the report")
        
        args.add_argument(
            "--reference_code_col",
            type=str,
            default="target",
            help="path to the reference file. It should be a jsonl with a column `slow_code_col` and `reference_code_col`. See sample in `data/codenet/unit_test/reference.jsonl`. If this is None, then the model_generated_output_paths should include a column called `reference_code_col`.",
        )

        args.add_argument(
            "--slow_code_col",
            type=str,
            default="input",
            help="Column name for the slow code",
        )

        args.add_argument(
            "--num_trials",
            type=int,
            default=3,
            help="Number of times to run each program for a test case. Defaults to 3.",
        )
        args.add_argument(
            "--ignore_first_k",
            type=int,
            default=0,
            help="The first `ignore_first_k` runs are ignored to alleviate outliers due to caching. Defaults to 2.",
        )
        args.add_argument(
            "--inputs_outputs_basepath",
            type=str,
        )

        args.add_argument(
            "--max_time_per_run",
            type=int,
            default=5,
            help="The maximum time allowed for each run. Defaults to 5 seconds.",
        )
        args.add_argument("--num_problems_to_evaluate", type=int, default=-1)
        args.add_argument(
            "--temp_dir",
            type=str,
            default=None,
            help="The temporary directory to write the code to. Defaults to None. If None, a temporary directory is created. If not, the directory is used to write the code to.",
        )
        
        args.add_argument("--cflags", type=str, default="--std=c++17 -O1")

        args.add_argument("--is_prompt_based", action="store_true")

        args.add_argument(
            "--cpu_number",
            type=int,
            default=1,
            help="We use taskset to assign a single CPU to each process. This is the CPU number to use. Defaults to 1.",
        )
        
        args.add_argument("--return_if_acc_below", type=float, default=0.0, help="If the accuracy is below this value, then the evaluation stops. Defaults to 0.0.")
        
        return args


# extract all the defaults in a dict

defaults = {
    'reference_file_path': None, 'model_generated_outputs_path': None, 'model_generated_output_col': 'generated_answer', 'model_generated_potentially_faster_code_col': 'generated_answer', 'output_report_file_path': None, 'reference_code_col': 'target', 'slow_code_col': 'input', 'num_trials': 3, 'ignore_first_k': 0, 'max_time_per_run': 5, 'num_problems_to_evaluate': -1, 'temp_dir': None, 'is_prompt_based': False, 'cpu_number': 1}
