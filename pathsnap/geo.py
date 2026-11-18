"""Trip loading and local metric projection.

Every trip comes with its own (small) road network anywhere in the world, so we
project each trip into a local equirectangular plane (metres) centred on the
trip. Over a single-city extent (< ~100 km) the distortion is well below GPS
noise; this avoids the cost of a UTM-zone lookup per trip and is fully
vectorised. (Documented choice: see report section 2.)
"""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np

M_PER_DEG = 111_320.0


@dataclass
class Trip:
    trip_id: str
    lonlat: np.ndarray      # (n,2) raw GPS lon,lat
    t: np.ndarray           # (n,) seconds
    nodes_ll: np.ndarray    # (N,2) lon,lat
    arcs: np.ndarray        # (M,2) from,to node ids (row index = arc id)
    route: np.ndarray | None  # reference arc ids (server-side only)
    lon0: float = 0.0
    lat0: float = 0.0

    # filled by project()
    xy: np.ndarray = None
    nodes_xy: np.ndarray = None

    def project(self):
        self.lon0 = float(np.mean(self.lonlat[:, 0]))
        self.lat0 = float(np.mean(self.lonlat[:, 1]))
        self.xy = to_xy(self.lonlat, self.lon0, self.lat0)
        self.nodes_xy = to_xy(self.nodes_ll, self.lon0, self.lat0)
        return self


def to_xy(ll, lon0, lat0):
    ll = np.asarray(ll, float)
    c = np.cos(np.radians(lat0))
    return np.c_[(ll[:, 0] - lon0) * M_PER_DEG * c, (ll[:, 1] - lat0) * M_PER_DEG]


def _load(path, **kw):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    return np.loadtxt(path, ndmin=2, **kw)


def load_trip(trip_dir: str, with_route: bool = True) -> Trip:
    tid = os.path.basename(trip_dir.rstrip("/"))
    base = os.path.join(trip_dir, tid)
    tr = _load(base + ".track")
    nodes = _load(base + ".nodes")
    arcs = _load(base + ".arcs", dtype=np.int64)
    route = None
    if with_route:
        r = _load(base + ".route", dtype=np.int64)
        route = None if r is None else r.ravel()
    # sanitise GPS: drop NaN / out-of-range rows, enforce time order
    ok = np.isfinite(tr).all(1) & (np.abs(tr[:, 0]) <= 180) & (np.abs(tr[:, 1]) <= 90)
    tr = tr[ok]
    tr = tr[np.argsort(tr[:, 2], kind="stable")]
    return Trip(tid, tr[:, :2], tr[:, 2], nodes[:, :2], arcs, route).project()


def list_trips(root: str):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and not d.startswith("."))
