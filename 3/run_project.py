"""Run the saved research notebooks, without downloading or changing raw data."""

import argparse
import ast
import base64
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from io import BytesIO, StringIO
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import nbformat


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--section", choices=["all", "pipeline", "models", "strategy", "backtest"], default="all")
args = parser.parse_args()
root = args.root.resolve()
os.environ["TTF_PROJECT_ROOT"] = str(root)
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"  # small regressions; avoid native-thread oversubscription
os.environ["MPLBACKEND"] = "Agg"

steps = {
    "pipeline": [
        "2. Data Processing/prices_processing.ipynb",
        "2. Data Processing/funda_processing.ipynb",
        "2. Data Processing/lng_sendout_processing.ipynb",
        "2. Data Processing/weather_hist_processing.ipynb",
        "2. Data Processing/weather_forecast_processing.ipynb",
        "3. Final_DB/final_table.ipynb",
    ],
    "models": ["4. Modelling/spread_models.ipynb"],
    "strategy": ["4. Strategy/spread_strategy.ipynb"],
    "backtest": ["5. Backtest/spread_backtest.ipynb"],
}

started = datetime.now(timezone.utc).isoformat()
manifest_path = root / "Documentation" / "run_manifest.json"
manifest = {
    "started_utc": started,
    "section": args.section,
    "python": platform.python_version(),
    "packages": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "nbformat", "nbclient", "ipykernel", "tzdata"]},
    "raw_sources": [],
    "executed_notebooks": [],
    "status": "running",
    "limitations": [
        "Bloomberg generic delivery mappings are inferred, not verified contract histories.",
        "AGSI/ALSI histories are revised snapshots, not archived publication vintages.",
        "Forecast issue hours are unavailable; a one-calendar-day safety lag is applied.",
        "Costs, fractional-MW positions and capital are research assumptions, not broker quotes.",
    ],
}

for path in sorted((root / "1. Raw Data").glob("*.csv")):
    manifest["raw_sources"].append({"file": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

manifest_path.parent.mkdir(exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

try:
    for section, notebooks in steps.items():
        if args.section not in ("all", section):
            continue
        for name in notebooks:
            path = root / name
            print(f"Running {name}", flush=True)
            nb = nbformat.read(path, as_version=4)
            namespace = {"__name__": "__main__", "__file__": str(path)}
            previous = Path.cwd()
            os.chdir(path.parent)
            count = 0
            try:
                for cell in nb.cells:
                    if cell.cell_type != "code":
                        continue
                    count += 1
                    cell.execution_count, cell.outputs = count, []
                    stdout, stderr = StringIO(), StringIO()
                    start = time.perf_counter()
                    try:
                        with redirect_stdout(stdout), redirect_stderr(stderr):
                            tree = ast.parse(cell.source)
                            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                            exec(compile(tree, str(path), "exec"), namespace)
                            value = eval(compile(ast.Expression(last.value), str(path), "eval"), namespace) if last else None
                        if stdout.getvalue():
                            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=stdout.getvalue()))
                        if stderr.getvalue():
                            cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=stderr.getvalue()))
                        if value is not None:
                            cell.outputs.append(nbformat.v4.new_output("execute_result", execution_count=count, data={"text/plain": repr(value)}))
                        if "matplotlib.pyplot" in sys.modules:
                            plt = sys.modules["matplotlib.pyplot"]
                            for number in plt.get_fignums():
                                blob = BytesIO()
                                plt.figure(number).savefig(blob, format="png", dpi=120, bbox_inches="tight")
                                cell.outputs.append(nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(blob.getvalue()).decode("ascii")}))
                            plt.close("all")
                        cell.metadata["execution_seconds"] = round(time.perf_counter() - start, 3)
                    except Exception as exc:
                        cell.outputs.append(nbformat.v4.new_output("error", ename=type(exc).__name__, evalue=str(exc), traceback=traceback.format_exc().splitlines()))
                        nbformat.write(nb, path)
                        raise
            finally:
                os.chdir(previous)
            nbformat.write(nb, path)
            manifest["executed_notebooks"].append({"file": name, "code_sha256": hashlib.sha256("\n".join(c.source for c in nb.cells if c.cell_type == "code").encode()).hexdigest(), "code_cells": sum(c.cell_type == "code" for c in nb.cells)})
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["status"] = "passed"
except Exception as exc:
    manifest["status"] = "failed"
    manifest["error"] = f"{type(exc).__name__}: {str(exc)[-1500:]}"
    raise
finally:
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("Saved executed notebooks and Documentation/run_manifest.json", flush=True)
