import os
from pathlib import Path
import yaml
import shutil
import subprocess


def rel_to_absolute(file_path, relative_path):
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(file_path)), relative_path)
    )
    
def get_config() -> dict:
    config_fp = Path(rel_to_absolute(__file__, '../config.yaml'))
    if not config_fp.exists():
        raise RuntimeError(f"No config file at {config_fp}")
    return yaml.safe_load(config_fp.read_text())
    
def copy_and_run(name: str, config: dict):
    try:
        bin_path = Path(config[name]['binary_path'])
    except KeyError:
        raise RuntimeError(f"Config dict missing {name} or binary_path key: {config}")

    # Copy binary and run
    local_bin_path = Path(rel_to_absolute(__file__, '../bin'))
    if not local_bin_path.exists():
        os.mkdir(local_bin_path)
        
    shutil.copy(bin_path, local_bin_path)
    
    log_dir = local_bin_path.joinpath(f'{name}-log')
    process = subprocess.Popen([local_bin_path.joinpath(bin_path.name), log_dir], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
        
    _, stderr = process.communicate()
    if stderr:
        print(f"Error output:\n{stderr.decode()}")
        
    return log_dir
