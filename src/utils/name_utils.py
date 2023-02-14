
# variants using which c++ is referred to in the dataset
cpp_name_variants = ["c++", "cpp", "c++14", "c++17", "c++11", "c++20"]
cpp_name_variants.extend([n.upper() for n in cpp_name_variants])
cpp_name_variants = set(cpp_name_variants)


# variants using which python is referred to in the dataset
python_name_variants = ["python", "py", "Py", "Python"]
python_name_variants.extend([n.upper() for n in python_name_variants])
python_name_variants = set(python_name_variants)

java_name_variants = ["java", "Java"]
java_name_variants.extend([n.upper() for n in java_name_variants])
java_name_variants = set(java_name_variants)

lang_name_to_standardized_name = {c: "cpp" for c in cpp_name_variants}
lang_name_to_standardized_name.update({p: "python" for p in python_name_variants})
lang_name_to_standardized_name.update({j: "java" for j in java_name_variants})
lang_name_to_standardized_name.update({"c": "c", "C": "c"})


def standardize_lang_name(lang_name: str) -> str:
    """Standardizes the language name to one of the following: "c", "cpp", "java", "python"."""
    if lang_name in lang_name_to_standardized_name:
        return lang_name_to_standardized_name[lang_name]
    else:
        raise ValueError(f"Unknown language name: {lang_name}")