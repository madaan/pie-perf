# pie-perf

## Dataset

* Python splits

* C++ splits

* Public test cases

Each file is a jsonl with the following fields:

```

```

We use `src/make_splits.py` to create these splits. The exact configuration for creating each split is specified in the folder.


## Evaluating Your Method


* Suppose you have a new method for code optimization, say `awesome_optimization`. We provide a sandbox for evaluating the generated code. The sandbox runs the input and the generated code over a set of test cases and reports the performance of both. We provide 



1. Save the generations in a jsonl file with the following fields:

```js
{
    "slow_code_col": "the column name for the input code",
    "model_generated_potentially_faster_code_col": "slow_code_col after applying awesome_optimization. This is the code that will be evaluated. You can also provide a list of different candidates here, and the evaluation will be done for each candidate",
}
```

2. Next, we need to provide the path to the file with some metadata. We call it the `reference_file` but providing references are optional. The main purpose of this file is to provide information like the language of the code, the problem id, etc. The file should have `slow_code_col` (same as the generations file) and `problem_id`. We join the generations file and the references file on the `slow_code_col` to get the problem id.


3. Finally, we need to provide the path to the file with the actual test cases. We call it the `inputs_outputs_basepath`. This is a directory with the following structure:

```
inputs_outputs_basepath/{problem_id}/{inputs, outputs}.txt
```

where `{inputs, outputs}.txt` are the input and output files for the problem with id `problem_id`. The input and output are plain text files. Each program is fed `inputs.txt` and the output is compared with `outputs.txt`.


4. So far, we have discussed the generation file, the reference file, and the inputs/outputs directory. In addition to these, we need to provide some information about the run. Specifically, the number of times each program should be run, the number of programs to evaluate, the timeout, and so on.

We wrap all of this information is provided in a yaml file. Here is an example:

```yaml
model_generated_outputs_path: "data/sample/codex_greedy_outputs.jsonl"
inputs_outputs_basepath: "data/codenet/public_test_cases/"
reference_file_path: "data/sample/py_reference.jsonl"
output_report_file_path: "data/sample//codex_greedy_outputs.jsonl.report"
num_problems_to_evaluate: -1
num_trials: 25
ignore_first_k: 1
max_time_per_run: 10
temp_dir: null
model_generated_potentially_faster_code_col: "generated_answers"
slow_code_col: "input"
reference_code_col: "target"
is_prompt_based: true
cpu_number: 0
```

Please see `src/codenet_eval/evalconfig.py` for the full list of parameters and their descriptions.

5. Finally, we can run the evaluation. We provide a script for this: `src/codenet_eval/run_eval.py`. The script takes the yaml file as input. Here is an example:

```bash
python src/codenet_eval/run_eval.py --eval_config data/sample/sample_eval_config.yaml
```
