# AGENTS.md — VietSight

## Hard rules
1. Model: YOLO26n-seg (`yolo26n-seg.pt`), Ultralytics 8.4.214, imgsz 640, 11 classes in this order:
   Car, Bus, Truck, Other vehicle, Motorcycle, Motorcyclist, Bicycle, Cyclist, Pedestrian, Sign, Traffic light.
2. Official notebook `notebooks/VietSight_TEAM55.ipynb`: read-only. Generated Kaggle copy changes only the parameter cell
   (+ one appended results cell). Export contract: `format=onnx, imgsz=640, opset=17, simplify=True, dynamic=False, batch=1`,
   no `nms`, no `half`. Submission = zip with exactly `model.onnx`, ≤ 42 MB.
3. TEAM_ID / seed = 55 everywhere.
4. Validation set frozen in `configs/val_frozen.txt`; never train on it.
5. No data/weights/ONNX/zip/cache/secrets in git. No push / public repo without the user's explicit request.
6. `sample_data.zip` = pipeline tests only.

## Layout
- `scripts/vslib/`: shared lib (geometry, rider pairing, writer, notebook loader, source mapping)
- `scripts/*.py`: CLI tools (see README table)
- `configs/sources/*.yaml`: per-source label mapping + license; `configs/versions/*.yaml`: dataset versions
- `data/raw|processed|versions/`: gitignored data; `data/sources_manifest.yaml`: provenance (committed)
- `docs/`: LABEL_GUIDELINE, DATA_SOURCES, VERSIONS
- `runs/`: RESULTS.md + run_report.json per run (weights gitignored)

## Before saying "done"
Run the relevant converter on real or sample data, `build_version.py`, and the local notebook smoke test when the
training/export path is touched. Report failures as they are.
