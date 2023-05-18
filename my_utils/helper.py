import os 
from pathlib import Path

def generate_treefolder(report_name, result_folder, slash="/"):
    """ Generate a txt report file """
    if not os.path.exists(result_folder):
        # os.mkdir(result_folder)
        path = Path(result_folder)
        path.mkdir(parents=True, exist_ok=True)
    report_file = result_folder + slash + f"{report_name}.txt"
    return report_file

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """ Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc. """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

