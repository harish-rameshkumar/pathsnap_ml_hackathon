"""PathSnap FastAPI backend (Laptop 1).
Run:  EVAL_DIR=/path/to/evaluator_trips uvicorn api.main:app --host 0.0.0.0 --port 8000"""
import os, sys, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathsnap.geo import list_trips, load_trip
from pathsnap.pipeline import Predictor, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.environ.get("EVAL_DIR", os.path.join(os.path.dirname(ROOT), "Student folder", "student_test"))


def _resolve_eval_dir(d):
    """Accept the trips folder itself, or a parent containing it (one level down)."""
    def has_trips(p):
        return os.path.isdir(p) and any(os.path.isfile(os.path.join(p, t, t + ".track")) for t in list_trips(p))
    if has_trips(d): return d
    for sub in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if has_trips(os.path.join(d, sub)): return os.path.join(d, sub)
    return d


EVAL_DIR = _resolve_eval_dir(EVAL_DIR)
log = logging.getLogger("pathsnap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = FastAPI(title="PathSnap")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = Predictor(os.path.join(ROOT, "artifacts"))
log.info("Serving trips from %s", EVAL_DIR)


class Req(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "eval_dir": EVAL_DIR}


@app.get("/trips")
def trips():
    out = []
    for tid in list_trips(EVAL_DIR):
        if os.path.isfile(os.path.join(EVAL_DIR, tid, tid + ".track")):
            out.append({"trip_id": tid})
    return out


@app.post("/predict")
def predict(req: Req):
    tid = req.trip_id
    d = os.path.join(EVAL_DIR, tid)
    if tid not in list_trips(EVAL_DIR) or not os.path.isfile(os.path.join(d, tid + ".track")):
        raise HTTPException(404, f"unknown trip {tid}")
    t0 = time.time()
    log.info("POST /predict trip_id=%s -> running pipeline", tid)
    res = predictor.predict(d)
    poly, arcs = predictor.route_polyline(res)
    try:
        metrics = evaluate(d, res)       # reference route stays server-side
    except Exception as e:               # e.g. .route missing: still return the prediction
        log.warning("could not score trip %s: %s", tid, e); metrics = None
    gps = [[float(a), float(b)] for b, a in res["trip"].lonlat]
    dt = time.time() - t0
    if metrics:
        log.info("trip %s n_gps=%d hmm_acc=%.4f clf_acc=%.4f cont=%.3f time=%.2fs", tid, len(gps),
                 metrics["hmm"]["pointwise_accuracy"], metrics["clf"]["pointwise_accuracy"],
                 metrics["hmm"]["route_continuity"], dt)
    return {"trip_id": tid, "n_gps": len(gps), "gps": gps, "route": poly, "n_route_arcs": len(arcs),
            "metrics": metrics, "timing": {**res["timing"], "total": dt}}
