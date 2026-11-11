"""ACPDAD synthetic research workflow and isolated ArduPilot SITL tools."""
import os,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.venv/matplotlib-cache'))
sys.path.insert(0,str(ROOT/'src'))
if len(sys.argv) > 1 and sys.argv[1] == 'sitl':
    from acpdad.sitl.cli import main as sitl_main
    if __name__ == '__main__':
        raise SystemExit(sitl_main(sys.argv[2:]))
else:
    from acpdad.workflow import main

if __name__=='__main__':
    main()
