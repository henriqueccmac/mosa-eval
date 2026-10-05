"""Apply the recorded production-source patch for one extension."""
from pathlib import Path
import subprocess
from support import ROOT

def apply(source: Path, scenario: str) -> None:
    patch = ROOT / 'patches' / f'{scenario}.patch'
    if patch.exists():
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=source, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=source, check=True)
