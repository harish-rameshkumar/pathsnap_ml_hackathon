# PathSnap – GPS map matching (classifier + HMM/Viterbi)

Pipeline: raw GPS → candidate road arcs (≤12 within 80 m) → 19 features → LightGBM candidate probability
→ HMM/Viterbi over candidates with road-network transitions → predicted arc per GPS point.

## Layout
- `pathsnap/` library: `geo` (loading, local metric projection), `labels` (route alignment, training/eval only), `candidates`, `hmm`, `pipeline`, `metrics`
- `scripts/` experiments: `00_build_dataset` (candidates, features, labels), `01_compare_models`, `02_tune_bayes` (Optuna), `03_hmm_and_final` (HMM tuning, final model), `04/08` EDA, `05/11` figures, `06/06b` route plots + Folium maps, `07` tuned-vs-untuned, `12` candidate graphs, `13` evaluator run, `09` report builder, `10` training curves
- `api/main.py` FastAPI backend (Laptop 1) · `ui/index.html` UI (Laptop 2)
- `artifacts/model.joblib`, `artifacts/hmm_config.json` inference assets (`dataset.pkl` is a regenerable cache, git-ignored)
- `results/` metrics, logs, figures · `report/PathSnap_Report.pdf` technical report (built by `scripts/09_build_report.py`)

## Setup (run everything from the project root)
```bash
pip install -r requirements.txt
export PATHSNAP_DATA="/path/to/Student folder"      # must contain train/ and student_test/
python scripts/00_build_dataset.py "$PATHSNAP_DATA"  # builds artifacts/dataset.pkl (cache, ~2 min)
python scripts/01_compare_models.py                  # 9-model comparison  (~10 min)
python scripts/02_tune_bayes.py 40                   # Optuna tuning       (~18 min)
python scripts/03_hmm_and_final.py                   # HMM tuning, final model, artifacts (~5 min)
python scripts/04_explore.py && python scripts/08_eda_plots.py      # EDA tables/plots
python scripts/05_figures.py && python scripts/11_extra_figures.py   # comparison / Bayes / HMM figures
python scripts/07_tuned_vs_untuned.py && python scripts/10_training_curves.py
python scripts/06_visualise.py && python scripts/06b_folium_maps.py # route plots + interactive Folium maps
python scripts/12_candidate_plots.py                                   # candidate-generation graphs
EVAL_DIR=/path/to/evaluator_test python scripts/13_evaluator_run.py  # final evaluation on evaluator trips (after development)
EVAL_DIR=/path/to/evaluator_test python scripts/14_rmf.py            # Route Mismatch Fraction (needs the trained model)
python scripts/09_build_report.py                                    # builds report/PathSnap_Report.pdf (needs Google Chrome)
```
Scripts read the dataset location from `PATHSNAP_DATA`; if it is not set they use `<project>/data` (copy or symlink the dataset folder, containing `train/` and `student_test/`, to `data/`). `data/` is git-ignored.
The API needs only `artifacts/model.joblib` and `artifacts/hmm_config.json`, both committed, so steps above are not required to run it.

## Final evaluation
Laptop 1 (API; folder holding the 10 evaluator trips, same per-trip layout incl. `.route` for server-side scoring):
```bash
EVAL_DIR=/path/to/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Laptop 2 (UI): `cd ui && python -m http.server 8080`, open `http://localhost:8080`, set API URL to `http://<laptop1-ip>:8000`, click *Load evaluation trips*.
The API log (terminal on Laptop 1) shows each `POST /predict`. The reference route is read only inside `pathsnap.pipeline.evaluate`; the prediction path never touches `.route`, and the UI only receives predictions and metrics.
