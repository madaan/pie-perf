import pandas as pd

def run(path: str):
    def _remove_sep(x):
        return x.split('|||')[1].strip() if '|||' in x else x
    df = pd.read_json(path, lines=True, orient="records")
    if 'greedy_generated_target_from_input' in df.columns:
        df['greedy_generated_target_from_input'] = df['greedy_generated_target_from_input'].apply(_remove_sep)
    
    for col in ['beam_generated_target_from_input', 'sample_generated_target_from_input']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x_list: [_remove_sep(x) for x in x_list])

    df.to_json(path, orient="records", lines=True)



if __name__ == "__main__":
    import sys
    from glob import glob
    for path in glob(sys.argv[1]):
        print(path)
        run(path)
    