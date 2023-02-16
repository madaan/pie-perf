import pandas as pd
import difflib

from src.utils.diff_utils import get_minimal_diff

def run(path: str):
    data = pd.read_json(path, lines=True, orient="records")
    for i, row in data.iterrows():
        # data.iloc[i, "diff"] = diff
        # print a nice report: the input, the target, and the diff
        # first, a header
        print("# " + '-' * 80)
        print("# " + '*' * 80)

        # then the input
        print(f"# Problem ID: {row['problem_id']} Submission ID v0: {row['submission_id_v0']}")
        print(f"\n# input:\n")
        input = clean_code(row['input'])
        print(f"{input}")
        # then the target
        print("\n#" + '|' * 80 + "\n")
        print(f"\n# target:\n")
        target = clean_code(row['target'])
        
        diff = get_minimal_diff(input, target)

        print(f"{target}")
        
        # then the diff
        print("\n#" + '|' * 80 + "\n")
        print(f"\n# diff:\n")

        # make a triple quote comment from the diff
        diff_comment = '"""\n' + "".join(diff) + '"""\n'
        print("\n\n" + diff_comment + "\n\n")
        
        
        # then a footer
        print("# " + '*' * 80)
        print("# " + '-' * 80)
        print('\n\n')

def get_diff(a, b):
    return difflib.ndiff(a.splitlines(keepends=True), b.splitlines(keepends=True))


def clean_code(code):
    code_lines = code.split("\n")
    # remove empty lines
    code_lines = [line for line in code_lines if len(line.strip()) > 0]
    return "\n".join(code_lines)
    
                                
if __name__ == '__main__':
    import sys
    run(sys.argv[1])