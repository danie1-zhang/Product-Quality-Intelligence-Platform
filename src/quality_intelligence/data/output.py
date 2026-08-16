import os
import shutil
from pathlib import Path


def replace_output_directory(staged_path: Path, output_path: Path) -> None:
    """Replace an output directory, restoring the previous output if the swap fails."""
    backup_path = output_path.with_name(f"{output_path.name}.backup")
    if backup_path.exists():
        if output_path.exists():
            shutil.rmtree(backup_path)
        else:
            os.replace(backup_path, output_path)
    if output_path.exists():
        os.replace(output_path, backup_path)
    try:
        os.replace(staged_path, output_path)
    except Exception:
        if backup_path.exists():
            os.replace(backup_path, output_path)
        raise
    if backup_path.exists():
        shutil.rmtree(backup_path)
