import difflib


def get_minimal_diff(code1, code2, return_lines: bool = False) -> str:
    diff = difflib.unified_diff(
        code1.splitlines(keepends=True), code2.splitlines(keepends=True), n=0
    )
    meta_symbols = set(["---", "+++", "@@"])
    diff_minus_meta = []
    has_meta = False
    for line in diff:
        for meta_symb in meta_symbols:
            if meta_symb in line:
                has_meta = True
                break

        if not has_meta:
            diff_minus_meta.append(line.strip())
        has_meta = False
    
    if return_lines:
        return diff_minus_meta

    return "\n".join(diff_minus_meta)


def is_only_diff_in_criteria(code1, code2, criteria, diff=None):
    if diff is None:
        diff = get_minimal_diff(code1, code2)
    for line in diff.splitlines():
        if line.startswith("+") or line.startswith("-"):
            line = line[1:]
            if not line.startswith(criteria):  # has diff in something other than criteria
                return False
    return True


def is_only_diff_in_imports(code1, code2,  diff=None) -> bool:
    return is_only_diff_in_criteria(code1, code2, "import", diff)

def is_only_diff_in_comments(code1, code2, lang: str = "python", diff=None) -> bool:
    if lang in {"python", "py"}:
        return is_only_diff_in_criteria(code1, code2, "#", diff)
    elif lang in {"java", "cpp", "c"}:
        return is_only_diff_in_criteria(code1, code2, "//", diff) or is_only_diff_in_criteria(code1, code2, "/*", diff)
    else:
        return False

def has_diff_with_tok(code1, code2, tok, diff=None) -> bool:
    if diff is None:
        diff = get_minimal_diff(code1, code2)
    for line in diff.splitlines():
        if line.startswith("+") or line.startswith("-"):
            line = line[1:]
            if tok in line:
                return True
    return False

# tests for is_only_diff_in_imports

code1 = """
import numpy as np
def foo():
    pass
"""

code2 = """
def foo():
    pass
"""

code3 = """
import scipy as sp
def bar():
    pass
"""

def test():
    assert is_only_diff_in_imports(code1, code2)
    assert not is_only_diff_in_imports(code1, code3)
    assert not is_only_diff_in_imports(code2, code3)

if __name__ == "__main__":
    test()