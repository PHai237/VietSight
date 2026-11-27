# VietSight — instructions for AI coding agents

See AGENTS.md (same rules). Summary:
- Data-centric competition: never change the model (YOLO26n-seg, imgsz 640) or the official notebook's library/export/submission cells.
  Only the parameter cell may change, and only via `scripts/make_kaggle_notebooks.py` (it asserts the other cells are identical).
- Class order is fixed (`scripts/vslib/__init__.py` CLASSES == notebook CLASSES). Never pass `nms=` to ONNX export.
- Reuse the notebook's validators through `scripts/vslib/notebook_lib.py`; do not copy them.
- Motorcyclist/Cyclist = one mask = vehicle ∪ rider(s) (official sample). Never map riders or two-wheelers blindly: use `vslib/instances.py`.
- Ambiguous or AI-prelabelled images go to `review/`; only `status: ok` enters a version. Val list `configs/val_frozen.txt` is frozen.
- Do not invent numbers or licenses in docs; write "chưa đo"/"unknown" until measured.
- Never commit data, weights, ONNX, zips, tokens. Never push or create a public repo unless the user asks.
- Python: `.venv` (3.12). Test changes with `scripts/run_notebook_local.py` / the converters on `data/raw/sample_data` before claiming done.
