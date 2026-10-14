"""python -m testbedlens [out_dir]: generate all scenario logs and write one report per log."""
import sys
from pathlib import Path

from .analyze import report
from .model import SCENARIOS, generate

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
spec_dir = Path(__file__).resolve().parent.parent / "specs"
for name in SCENARIOS:
    log = out / "logs" / f"{name}.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    generate(name, log)
    spec = spec_dir / ("start.json" if name == "start" else "step.json")
    results, diag = report(log, spec, out / "reports" / name)
    fails = [r["id"] for r in results if not r["passed"]]
    print(f"{name:14s} {'PASS' if not fails else 'FAIL ' + ','.join(fails)}")
