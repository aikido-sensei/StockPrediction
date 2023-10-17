import os
import re
from re import Pattern


def delete_matching_files(path: str, pattern_str: str) -> None:
    """
    Deletes the files matching the pattern given in the path given.
    :param path: String that indicates the paths in which the files should be deleted.
    :param pattern_str: String that indicates the pattern of the files that should be deleted.
    """
    file_pattern: Pattern = re.compile(pattern_str)
    for file_name in os.listdir(path):
        if file_pattern.match(file_name):
            file_path: str = os.path.join(path, file_name)
            os.remove(file_path)
