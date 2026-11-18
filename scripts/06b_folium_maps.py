"""Interactive Folium maps: GPS, road network, reference route, predicted route, with legend + layer control."""
import numpy as np, folium
from folium.plugins import MiniMap
from common import *
from pathsnap.geo import load_trip
from pathsnap.pipeline import Predictor
from pathsnap.labels import align_to_route
F = os.path.join(ROOT, "results/figures")
pred = Predictor(os.path.join(ROOT, "artifacts"))
LEGEND = """<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;border:1px solid #999;font:12px sans-serif">
<b>{title}</b><br><span style="color:#2563eb">&#9679;</span> raw GPS fixes<br><span style="color:#16a34a">&#9632;</span> reference route<br>
<span style="color:#e11d48">&#9632;</span> predicted route (classifier + HMM)<br><span style="color:#999">&#9632;</span> road network (arcs near trip)<br>
<span style="color:#f59e0b">&#9679;</span> fixes whose predicted arc != reference</div>"""
for part, tid in [("student_test", "00000007"), ("student_test", "00000056"), ("student_test", "00000077"), ("train", "00000034")]:
    d = os.path.join(DATA_ROOT, part, tid)
    res = pred.predict(d); trip = res["trip"]; rtrip = load_trip(d); ref = align_to_route(rtrip)
    poly, _ = pred.route_polyline(res)
    m = folium.Map(location=[trip.lat0, trip.lon0], zoom_start=14, tiles="OpenStreetMap")
    pad = 150.0; lo, hi = trip.xy.min(0) - pad, trip.xy.max(0) + pad
    A, B = trip.nodes_xy[trip.arcs[:, 0]], trip.nodes_xy[trip.arcs[:, 1]]
    near = np.nonzero((np.minimum(A, B) <= hi).all(1) & (np.maximum(A, B) >= lo).all(1))[0]
    if len(near) > 6000: near = near[np.linspace(0, len(near) - 1, 6000).astype(int)]
    g_net = folium.FeatureGroup("Road network", show=False)
    for a in near:
        folium.PolyLine([[trip.nodes_ll[x][1], trip.nodes_ll[x][0]] for x in trip.arcs[a]], color="#999", weight=1).add_to(g_net)
    g_gps = folium.FeatureGroup("GPS observations"); g_ref = folium.FeatureGroup("Reference route"); g_pred = folium.FeatureGroup("Predicted route"); g_err = folium.FeatureGroup("Mismatched fixes")
    for p in trip.lonlat[::3]: folium.CircleMarker([p[1], p[0]], radius=2, color="#2563eb", weight=1, fill=True).add_to(g_gps)
    chain = [int(trip.arcs[rtrip.route[0], 0])] + [int(trip.arcs[a, 1]) for a in rtrip.route]   # route is a connected chain, keep order
    folium.PolyLine([[trip.nodes_ll[x][1], trip.nodes_ll[x][0]] for x in chain], color="#16a34a", weight=8, opacity=.5, tooltip="reference route").add_to(g_ref)
    folium.PolyLine(poly, color="#e11d48", weight=3, tooltip="predicted route").add_to(g_pred)
    wrong = np.nonzero(res["hmm"] != ref)[0]
    for i in wrong[::2]: folium.CircleMarker([trip.lonlat[i][1], trip.lonlat[i][0]], radius=3, color="#f59e0b", fill=True, tooltip=f"fix {i}").add_to(g_err)
    for gr in (g_net, g_gps, g_ref, g_pred, g_err): gr.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m); MiniMap().add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(title=f"{part} trip {tid}")))
    m.fit_bounds([[trip.lonlat[:, 1].min(), trip.lonlat[:, 0].min()], [trip.lonlat[:, 1].max(), trip.lonlat[:, 0].max()]])
    m.save(os.path.join(F, f"map_pred_vs_ref_{tid}.html")); print("saved", tid)
