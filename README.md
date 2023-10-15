# Learning Performance-Improving Code Edits



- Repository for Learning Performance-Improving Code Edits ([paper](https://arxiv.org/pdf/2302.07867.pdf), [website](https://pie4perf.com/)).

<img width="879" alt="image" src="https://raw.githubusercontent.com/madaan/pie-perf/main/docs/static/images/mainfig-v4.jpg">


## Updates 📢
[May 2023] A large number of problem statements in codenet were in Japanese. We have translated them to English using ChatGPT/GPT-4. The files are located [here](data/problem_statements_translated.zip)



## Dataset

- PIE is based on [IBM CodeNet](https://github.com/IBM/Project_CodeNet). Huge thanks to the authors of CodeNet for making their curated dataset available!

- All trajectories (`tsv`) are located [here](https://drive.google.com/file/d/19IL3VETwVI9rdibB979Xm4gEWYwn0CkV/view?usp=sharing). Columns description:

- `user_id`: user id
- `problem_id`: problem id. Details about the problems can be found in `data/problem_list.csv`
- `language`: programming language
- `submission_id_v0`: submission id of the first version of the code
- `submission_id_v1`: submission id of the improved version of the code
- `cpu_time_v0`: cpu time of the first version of the code
- `cpu_time_v1`: cpu time of the second version of the code. `cpu_time_v0` > `cpu_time_v1` by at least 1% for all the pairs in the dataset. For pairs where the first version was TLE, `cpu_time_v0` is set to some high value (e.g. 1000000).
- `memory_v{0,1}`: memory used by the code in the two versions. We can also use `memory_v0` > `memory_v1` to filter out pairs.
- `status_v{0,1}`: status of the code in the two versions. `status_v0` can be `Accepted` or `Time Limit Exceeded`, but `status_v1` is always `Accepted`.
- `improvement_frac`: percentage of improvement of the second version of the code with respect to the first version. `improvement_frac` is always > 0.

* [Python splits](https://drive.google.com/file/d/1ec8eOWgnBrzy2HlNDlTX6iURwQcIxDXH/view?usp=sharing)

* [C++ splits](https://drive.google.com/file/d/1NqMT7kqCwk99hj4BjpUcsxLIzPFv_DtT/view?usp=sharing)

Each file is a jsonl:

```
{
    "user_id": "u187233527",
    "problem_id": "p03317",
    "language": "python",
    "submission_id_v0": "s743350482",
    "submission_id_v1": "s961810347",
    "cpu_time_v0": 28.0,
    "cpu_time_v1": 17.0,
    "memory_v0": 3060.0,
    "memory_v1": 3060.0,
    "status_v0": "Accepted",
    "status_v1": "Accepted",
    "improvement_frac": 39.29,
    "input": "N, K = list(map(int, input().split()))\n\nN -= K\n\nans = 1\n\nwhile N > 0:\n\n  N -= K - 1\n\n  ans += 1\n\nprint(ans)",
    "target": "import math\n\n\n\nn, k = list(map(int, input().split()))\n\nprint((math.ceil((n - 1) / (k - 1))))",
    "code_v0_loc": 7.0,
    "code_v1_loc": 4.0,
    "code_v0_num_chars": 101,
    "code_v1_num_chars": 84,
    "code_v0_no_empty_lines": "N, K = list(map(int, input().split()))\nN -= K\nans = 1\nwhile N > 0:\n    N -= K - 1\n    ans += 1\nprint(ans)\n",
    "code_v1_no_empty_lines": "import math\n\nn, k = list(map(int, input().split()))\nprint((math.ceil((n - 1) / (k - 1))))\n",
    "code_same": false,
    "relative_loc_diff_percent": 42.8571428571,
    "diff": [
        "-N, K = list(map(int, input().split()))",
        "-N -= K",
        "-ans = 1",
        "-while N > 0:",
        "-    N -= K - 1",
        "-    ans += 1",
        "-print(ans)",
        "+import math",
        "+",
        "+n, k = list(map(int, input().split()))",
        "+print((math.ceil((n - 1) / (k - 1))))"
    ],
    "diff_only_import_comment": false,
    "measured_runtime_v0": 0.045435272,
    "measured_runtime_v1": 0.0459265449,
    "runtime_lift": 0.9893030722
}
```

We use `src/make_splits.py` to create these splits. The exact configuration for creating each split is specified in the folder.

- [Public test cases](https://drive.google.com/file/d/1RcUpZMOR8L2xYYWDZx7I0tHFzFgg7COO/view?usp=share_link)
- [Generated test cases](https://drive.google.com/file/d/1migwX4wpED0gDDxn7gS6q55vWeXIDgId/view?usp=drive_link). These test cases are sourced from [alphacode](https://github.com/google-deepmind/code_contests).

## Evaluating Your Method

- Suppose you have a new method for code optimization, say `awesome_optimization`. We provide a sandbox for evaluating the generated code. The sandbox runs the input and the generated code over a set of test cases and reports the performance of both. We provide

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


----

## Citation

```
@article{madaan2023learning,
    title={Learning Performance-Improving Code Edits},
    author={Madaan, Aman and Shypula, Alexander and Alon, Uri and Hashemi, Milad and Ranganathan, Parthasarathy and Yang, Yiming and Neubig, Graham and Yazdanbakhsh, Amir},
    journal={arXiv preprint arXiv:2302.07867},
    year={2023}
}
```
