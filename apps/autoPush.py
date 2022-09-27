# git add/commit/push automatically

from pathlib import Path
from subprocess import Popen

def main():

    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    p = Popen("autoPush.bat", cwd=root)
    stdout, stderr = p.communicate()