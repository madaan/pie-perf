from lib2to3 import refactor
import re
import subprocess
import tempfile

from black import format_str, FileMode

def clean_code(code_str: str, lang: str = "py") -> str:
    """Removes whitespace, runs black, and removes trailing whitespace.

    Args:
        code_str (str): _description_

    Returns:
        str: _description_
    """
    code_str = code_str.strip().replace("\n\n", "\n").strip()
    # remove empty lines
    all_lines = code_str.split("\n")
    non_empty_lines = [line for line in all_lines if line.strip() != ""]
    # remove lines with only whitespace
    code = "\n".join(non_empty_lines).strip()
    if lang.lower() in {"py", "python"}:
        try:  # black fails on Python 2 code
            code = format_str(code, mode=FileMode())
            return code
        except Exception:
            return code
    elif lang.lower() in {"cpp", "c++"}:
        try:
            code = subprocess.check_output(["clang-format", "-style=LLVM"], input=code.encode('utf-8')).decode('utf-8')
            code = re.sub(r"\n+", "\n", code)
            return code
        except Exception:
            return code
    else:
        raise NotImplementedError(f"Language {lang} not supported")


def remove_unused_cpp(code):
    return code
    # TODO: this doesn't really work
    with tempfile.NamedTemporaryFile(mode='w+t', suffix='.cpp', delete=False) as temp_file:
        temp_file.write(code)
        temp_file.seek(0)
        print(temp_file.name)
        input()
        process = subprocess.Popen(["cppclean", temp_file.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stderr)
        cleaned_code = stdout.decode().strip()
    return cleaned_code


avail_fixes = refactor.get_fixers_from_package('lib2to3.fixes')
py_converter = refactor.RefactoringTool(avail_fixes)
def convert_2to3(py_script):
    try:
        # convert python2 to python3
        # taken from https://stackoverflow.com/questions/30340151/using-2to3-on-in-memory-scripts
        # if the script does not end with a newline, add one
        added_newline = False
        if py_script[-1] != '\n':
            py_script += '\n'
            added_newline = True
        ast = py_converter.refactor_string(py_script, '<script>')
        converted_code = str(ast)
        if added_newline:
            converted_code = converted_code[:-1]
        return converted_code
    except Exception as e:  # if 2to3 fails, just return the original code
        return py_script

def test():
    input = """
n, k = map(int,input().split())

h = list(map(int,input().split()))

INF = float('inf')

dp = [INF] * n 

dp[0] = 0 

dp[1] = abs(h[1] - h[0])







for i in range(2,n):

    for j in range(1, min(i, k) + 1):

        dp[i] = min(dp[i], dp[i - j] + abs(h[i] - h[i - j]))

        



print(dp[n - 1])"""

    print(clean_code(input))

def test_clean_cpp_code():
    code = """
    #define UNUSED_VARIABLE 0

    int main() {
        int used_variable = 42;
        int UNUSED_VARIABLE;
        return 0;
    }
    """

    cleaned_code = remove_unused_cpp(code)
    print(cleaned_code)
    # Verify that the unused macro has been removed from the code
    assert "UNUSED_VARIABLE" not in cleaned_code



if __name__ == "__main__":
    # test()
    test_clean_cpp_code()