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
                            stderr=subprocess.PIPE,
                            text=True)
        
    stdout, stderr = process.communicate()
    for line in stdout.splitlines():
        print(line)
        
    if stderr:
        print(f"Error output:\n{stderr.decode()}")
        
    return log_dir

def extract_plottable_rectangles(df, ep):
    verts_l = []
    df = df[df['episode'] == ep]
    for _, row in df.iterrows():
        xs = [row['x0'], row['x1'], row['x2'], row['x3'], row['x0']]
        ys = [row['y0'], row['y1'], row['y2'], row['y3'], row['y0']]
        verts_l.append((xs, ys))
    return verts_l
