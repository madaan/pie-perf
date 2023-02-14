import json
import pickle
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np

def nanmean(arr):
    return np.mean(arr)
    nan_mask = np.isnan(arr)
    arr = arr[~nan_mask]
    return np.mean(arr)

def run(runtime_pickles_dir: str):
    results = []
    submission_id_to_time = {}
    for path in tqdm(glob.glob(f"{runtime_pickles_dir}/*.pkl"), desc="Loading runtime pickles"):
        with open(path, "rb") as f:
            d = pickle.load(f)
            submission_id = list(d.keys())[0]
            tmp = {
                "submission_id": submission_id,
                "public": nanmean(np.array(d[submission_id]["public"])),
                "generated": nanmean(np.array(d[submission_id]["generated"])),
            }
            tmp['all'] = (tmp['public'] + tmp['generated']) / 2
            submission_id_to_time[submission_id] = tmp
            results.append(tmp)
            
    df = pd.DataFrame(results)
    print(df.describe())
    print(df.head())
    codenet_df = pd.read_csv("data/improvement_pairs_additional_metadata.csv.filtered.bak", sep="\t")

    # here's how to do this in 2 lines:
    for key in ["public", "generated", "all"]:
        codenet_df[f"measured_time_v0_{key}"] = codenet_df['submission_id_v0'].apply(lambda x: submission_id_to_time[x][key] if x in submission_id_to_time else -1)
        codenet_df[f"measured_time_v1_{key}"] = codenet_df['submission_id_v1'].apply(lambda x: submission_id_to_time[x][key] if x in submission_id_to_time else -1)
    
    codenet_df = codenet_df[(codenet_df["measured_time_v0_all"] != -1) & (codenet_df["measured_time_v1_all"] != -1)]
    codenet_df.to_csv("data/runtime_eval/improvement_pairs_additional_metadata.csv.filtered.measured_time", sep="\t", index=False)

    with open("data/runtime_eval/submission_id_to_avg_time.json", "w") as f:
        json.dump(submission_id_to_time, f)

    
if __name__ == "__main__":
    import sys
    run(runtime_pickles_dir=sys.argv[1])