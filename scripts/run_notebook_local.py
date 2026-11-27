"""Run a notebook's code cells in-process (no Jupyter), overriding parameters after the parameter cell.
Used to smoke-test the full official flow (data report -> train -> ONNX export -> platform self-check -> zip -> val score)
on CPU with a tiny dataset before spending Kaggle GPU time.

  python scripts/run_notebook_local.py notebooks/kaggle/VietSight_TEAM55_kaggle.ipynb \
      --set DATASET_PATH=data/versions/vtest EPOCHS=1 BATCH=2 REQUIRE_GPU=False INSTALL_PACKAGES=False WORK_DIR=runs/smoke
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path


def bang_to_subprocess(src):
    """Jupyter `!cmd {VAR}` lines -> subprocess (shell=True, {VAR} expanded like IPython), keeping indentation."""
    out = []
    for line in src.splitlines():
        body = line.lstrip()
        if body.startswith("!"):
            ind = line[: len(line) - len(body)]
            cmd = body[1:].replace("'''", "")
            out.append(f"{ind}__import__('subprocess').run(f'''{cmd}''', shell=True, check=True)")
        else:
            out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--set", nargs="*", default=[], help="NAME=python_literal (strings may be bare)")
    ap.add_argument("--stop-after", type=int, help="stop after this code cell index")
    args = ap.parse_args()
    over = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            over[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            over[k] = v
    if sys.platform == "win32" and "resource" not in sys.modules:
        # LOCAL WINDOWS SMOKE TEST ONLY: the embedded grader imports the Unix-only `resource` module (rlimits of the
        # Linux grading container). A no-op stub lets the same code run here; Kaggle/Colab (Linux) use the real module.
        import types

        stub = types.ModuleType("resource")
        stub.RLIM_INFINITY = -1
        for n in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_NPROC", "RLIMIT_NOFILE", "RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_CORE", "RLIMIT_STACK"):
            setattr(stub, n, 0)
        stub.getrlimit = lambda *_: (-1, -1)
        stub.setrlimit = lambda *_: None
        stub.getrusage = lambda *_: types.SimpleNamespace(ru_maxrss=0, ru_utime=0.0, ru_stime=0.0)
        stub.RUSAGE_SELF = stub.RUSAGE_CHILDREN = 0
        sys.modules["resource"] = stub
        import os
        import shutil

        os.symlink = lambda src, dst, *a, **k: shutil.copyfile(src, dst)  # Windows needs admin rights for symlinks
    nb = json.loads(Path(args.notebook).read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = bang_to_subprocess(src)
        t0 = time.time()
        print(f"\n===== cell {i}: {src.splitlines()[0][:90] if src else ''}", flush=True)
        exec(compile(src, f"<cell {i}>", "exec"), ns)  # noqa: S102
        if src.startswith("#@title Tham số của đội") or (over and any(k in ns for k in over) and not ns.get("_over_done")):
            ns["_over_done"] = True
            ns.update(over)
            if "PARAMS" in ns:
                ns["PARAMS"].update({k: v for k, v in over.items() if k in ns["PARAMS"]})
            print("overrides:", over)
        print(f"----- cell {i} ok ({time.time() - t0:.1f}s)", flush=True)
        if args.stop_after is not None and i >= args.stop_after:
            break


if __name__ == "__main__":
    sys.exit(main())
