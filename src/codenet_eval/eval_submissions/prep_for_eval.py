"""Prepares submissions for evaluation.
This involves flattening the submissions in the following directory structure:

eval_basedir/problem_id/submission_id
"""
import pathlib
from tqdm import tqdm


def write_code_for_eval(df, basedir):
    
    df.rename(columns={"input": "code_v0", "target": "code_v1"}, inplace=True)
    all_submissions = list(df['submission_id_v0'].unique()) + list(df['submission_id_v1'].unique())
    all_submissions = set(all_submissions)
    print(f"Number of unique submissions: {len(all_submissions)}")
    assert len(set(df['submission_id_v0'].unique())) == len(df)

    submission_id_to_metadata = get_submission_id_to_metadata(df)
    print(f"Number of submissions: {len(submission_id_to_metadata)}")
    write_submissions(submission_id_to_metadata, basedir=basedir)
    return submission_id_to_metadata
    


def write_submissions(submission_id_to_metadata, basedir):
    for submission_id, metadata in tqdm(submission_id_to_metadata.items(), total=len(submission_id_to_metadata), desc="Writing submissions"):
        problem_id = metadata["problem_id"]
        user_id = metadata["user_id"]
        language = metadata["language"]
        code = metadata["code"]
        loc = metadata["loc"]
        dir = f"{basedir}/{problem_id}/{submission_id}"
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        with open(f"{dir}/{submission_id}.py", "w") as f:
            f.write(code + "\n")
        with open(f"{dir}/metadata.txt", "w") as f:
            f.write(f"""
user_id: {user_id}
language: {language}
loc: {loc}
problem_id: {problem_id}
submission_id: {submission_id}
""")
    


def get_submission_id_to_metadata(df):
    """user_id problem_id      language        subm`ission_id_v0        submission_id_v1        cpu_time_v0     cpu_time_v1     memory_v0       memory_v1       status_v0       status_v1      improvement_frac        code_v0 code_v1 code_v0_loc     code_v1_loc"""
    submission_id_to_metadata = dict()
    df = df.sort_values(by="improvement_frac", ascending=False).reset_index(drop=True)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if row["submission_id_v0"] not in submission_id_to_metadata:
                submission_id_to_metadata[row["submission_id_v0"]] = {
                    "user_id": row["user_id"],
                    "problem_id": row["problem_id"],
                    "language": row["language"],
                    "code": row["code_v0"],
                    "loc": row["code_v0_loc"],
                }

            if row["submission_id_v1"] not in submission_id_to_metadata:
                submission_id_to_metadata[row["submission_id_v1"]] = {
                    "user_id": row["user_id"],
                    "problem_id": row["problem_id"],
                    "language": row["language"],
                    "code": row["code_v1"],
                    "loc": row["code_v1_loc"],
                }
        except Exception as e:
            print(row)
            raise e
    return submission_id_to_metadata


if __name__ == "__main__":
    write_code_for_eval()
