# -*- coding: utf-8 -*-
"""Run --self-test, --means-only, or normally after checking the paths."""
from pathlib import Path
import argparse, datetime as dt, importlib.util, json, sys
import numpy as np
import pandas as pd

MEAN_DIR = Path(r"C:\Users\zyd\Desktop\spring_winter_mean_fields")
OLD_RESULTS = Path(r"C:\Users\zyd\Desktop\spring_winter_transport_experiments")
HELPER = Path(r"C:\Users\zyd\PycharmProjects\spring_winter_transport_experiments.py")
BOUNDARIES = Path(r"C:\Users\zyd\PycharmProjects\province_to_province_boundary_grid")
WIND_DIR = Path(r"E:\wind_24h")
PM_DIR = Path(r"E:\pm25_1km_24h_new")
PBL_DIR = Path(r"C:\Users\zyd\PycharmProjects\pbl_height")
OUTPUT_ROOT = Path(__file__).resolve().parent / "wind_diagnostics"
START, END = "2023-03-01", "2024-02-29"
RUN_REPLACEMENTS = True
MEAN_WIND_EPS = 0.05
NORMAL_EPS = 0.05
FACTOR = 86400e-12

def ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=np.isfinite(b) & (b != 0))

def wav(x, w):
    x, w = np.asarray(x), np.asarray(w)
    ok = np.isfinite(x) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan

def split_wind(us, vs, uw, vw):
    ss, sw = np.hypot(us, vs), np.hypot(uw, vw)
    ok = np.isfinite(ss) & np.isfinite(sw) & (ss >= MEAN_WIND_EPS) & (sw >= MEAN_WIND_EPS)
    winds = {
        "SS": (us, vs),
        "WS": (sw * ratio(us, ss), sw * ratio(vs, ss)),
        "SW": (ss * ratio(uw, sw), ss * ratio(vw, sw)),
        "WW": (uw, vw),
    }
    return {k: tuple(np.where(ok, x, np.nan) for x in uv) for k, uv in winds.items()}, ok

def load_helper():
    spec = importlib.util.spec_from_file_location("_wind_helper", HELPER)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    source = HELPER.read_text(encoding="utf-8-sig")
    exec(compile(source.replace("@njit(cache=True)", "@njit(cache=False)"), str(HELPER), "exec"), m.__dict__)
    m.MEAN_DIR, m.BOUNDARY_DIR = MEAN_DIR, BOUNDARIES
    return m

def existing(out):
    frames = []
    for season in ("spring", "winter"):
        f = pd.read_csv(MEAN_DIR / (season + "_pbl_weighted_wind_mean.csv"))
        if f.duplicated(["lon", "lat"]).any():
            raise ValueError("Duplicate wind coordinates")
        f["season"] = season
        f["directional_persistence_ratio"] = ratio(
            np.hypot(f.u_mean_m_s, f.v_mean_m_s), f.mean_hourly_speed_m_s)
        if (f.directional_persistence_ratio.dropna() > 1.00001).any():
            raise ValueError("Inconsistent scalar/vector means")
        frames.append(f)
    pd.concat(frames, ignore_index=True).to_csv(
        out / "existing_grid_wind_metrics.csv", index=False, encoding="utf-8-sig")
    return frames

class Accumulator:
    def __init__(self, n):
        self.n = np.zeros(n, int)
        self.s = np.zeros((6, n))
        self.cos_n = np.zeros(n, int)
        self.pairs = np.zeros(n, int)
        self.flips = np.zeros(n, int)
        self.prev = np.zeros(n)
        self.prev_ok = np.zeros(n, bool)
        self.prev_time = None

    def add(self, t, u, v, nx, ny):
        speed, un = np.hypot(u, v), u * nx + v * ny
        ok = np.isfinite(speed) & np.isfinite(un)
        self.n += ok
        values = np.array([u, v, speed, un, np.abs(un), np.zeros_like(un)])
        self.s += np.where(ok[None, :], values, 0)
        co = ok & (speed > 1e-8)
        self.s[5] += np.where(co, ratio(np.abs(un), speed), 0)
        self.cos_n += co
        sign_ok = ok & (np.abs(un) >= NORMAL_EPS)
        if self.prev_time is not None and t - self.prev_time == dt.timedelta(hours=1):
            pairs = sign_ok & self.prev_ok
            self.pairs += pairs
            self.flips += pairs & (np.sign(un) != self.prev)
        self.prev, self.prev_ok, self.prev_time = np.sign(un), sign_ok, t

    def metrics(self):
        u, v, speed, un, abs_un = [ratio(x, self.n) for x in self.s[:5]]
        return pd.DataFrame({
            "n_hours": self.n,
            "mean_hourly_speed_m_s": speed,
            "vector_mean_speed_m_s": np.hypot(u, v),
            "directional_persistence_ratio": ratio(np.hypot(u, v), speed),
            "mean_abs_normal_speed_m_s": abs_un,
            "abs_mean_normal_speed_m_s": np.abs(un),
            "normal_persistence_ratio": ratio(np.abs(un), abs_un),
            "mean_abs_cos_to_normal": ratio(self.s[5], self.cos_n),
            "normal_sign_reversal_fraction": ratio(self.flips, self.pairs),
            "eligible_consecutive_pairs": self.pairs,
        })

def endpoint_geometry(bs, points):
    idx, nx, ny, weight, rows, groups = [], [], [], [], [], []
    for b in bs:
        begin = len(idx)
        for j, length in enumerate(b.lengths):
            if length <= 0:
                continue
            for e in (0, 1):
                i = b.start + j + e
                idx.append(i); nx.append(b.nx_length[j] / length)
                ny.append(b.ny_length[j] / length); weight.append(length / 2)
                rows.append([b.name, j, e, points[i, 0], points[i, 1]])
        groups.append(slice(begin, len(idx)))
    return np.array(idx), np.array(nx), np.array(ny), np.array(weight), pd.DataFrame(
        rows, columns=["boundary_name", "segment", "endpoint", "lon", "lat"]), groups

def self_test():
    a = Accumulator(1); t = dt.datetime(2023, 3, 1)
    for k, u in enumerate((2., -2.)):
        a.add(t + dt.timedelta(hours=k), np.array([u]), np.array([0.]), np.array([1.]), np.array([0.]))
    z = a.metrics().iloc[0]
    assert z.mean_hourly_speed_m_s == 2 and z.vector_mean_speed_m_s == 0
    assert z.mean_abs_normal_speed_m_s == 2 and z.normal_sign_reversal_fraction == 1
    winds, ok = split_wind(np.array([2.]), np.array([0.]), np.array([0.]), np.array([3.]))
    assert ok.all() and np.allclose(winds["WS"][0], 3) and np.allclose(winds["SW"][1], 2)
    print("Self-test passed.")

def main(args):
    out = OUTPUT_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    print("Output:", out, flush=True)
    sf, wf = existing(out)
    if args.means_only:
        return
    for folder in ([WIND_DIR, PBL_DIR] + ([PM_DIR] if RUN_REPLACEMENTS else [])):
        if not folder.is_dir():
            raise FileNotFoundError("Check raw input directory: " + str(folder))
    m = load_helper()
    if m.Dataset is None:
        raise ImportError("Install netCDF4")
    from scipy.spatial import Delaunay
    bs, points = m.load_boundaries()
    wp = sf[["lon", "lat"]].to_numpy(float)
    iw = m.LinearWeights(wp, points, True)
    us, vs = iw(sf.u_mean_m_s.to_numpy()), iw(sf.v_mean_m_s.to_numpy())
    uw = iw(m.reorder_exact(wf[["lon","lat"]].to_numpy(float), wf.u_mean_m_s.to_numpy(), wp))
    vw = iw(m.reorder_exact(wf[["lon","lat"]].to_numpy(float), wf.v_mean_m_s.to_numpy(), wp))
    wind_cases, direction_ok = split_wind(us, vs, uw, vw)
    hp = pd.read_csv(MEAN_DIR / "spring_pblh_mean_native_grid.csv")[["lon","lat"]].to_numpy(float)
    ht = Delaunay(hp)
    inside, wi = ht.find_simplex(points) >= 0, ht.find_simplex(wp) >= 0
    support_ok = ((iw.weights <= 1e-12) | wi[iw.vertices]).all(axis=1) & iw.usable
    audit = pd.DataFrame([{
        "boundary_name": b.name,
        "pbl_outside_nodes": int((~inside[b.start:b.stop]).sum()),
        "pbl_outside_wind_support_nodes": int((~support_ok[b.start:b.stop]).sum()),
        "weak_mean_direction_nodes": int((~direction_ok[b.start:b.stop]).sum()),
        "strict_core": bool(inside[b.start:b.stop].all() and support_ok[b.start:b.stop].all())
    } for b in bs])
    idx, nx, ny, weight, locations, groups = endpoint_geometry(bs, points)
    stats = {s: Accumulator(len(idx)) for s in ("spring","winter")}
    oldlog = pd.read_csv(OLD_RESULTS / "processing_log.csv")
    allowed = set(pd.to_datetime(oldlog.loc[oldlog.status == "processed", "time_cst"]))
    cases = ["actual_spring_same_mask"] + list(wind_cases)
    sums, counts = np.zeros((5,len(bs))), np.zeros((5,len(bs)), int)
    scopes = {"all_boundaries": [np.zeros(5), np.zeros(5,int)],
              "strict_core": [np.zeros(5), np.zeros(5,int)]}
    logs, pm_points, ip = [], None, None
    expected = {"spring":0, "winter":0}
    exp_rows = []
    day, end, nd = dt.date.fromisoformat(START), dt.date.fromisoformat(END), 0
    try:
        while day <= end:
            season = "spring" if day.month in (3,4,5) else "winter" if day.month in (12,1,2) else None
            if season is None:
                day += dt.timedelta(days=1); continue
            if args.max_days is not None and nd >= args.max_days:
                break
            nd += 1; expected[season] += 24
            print(day, season, flush=True)
            try:
                h = pd.read_csv(PBL_DIR / f"pbl_height_{day:%Y%m%d}.csv")
                pp, hv = h[["lon","lat"]].to_numpy(float), h.pbl_heightchem.to_numpy(float) * m.PBL_SCALE
                hb, hw = m.interpolate_pbl(pp,hv,points), m.interpolate_pbl(pp,hv,wp)
            except Exception as error:
                logs.append([str(day),"pbl_error",repr(error)])
                day += dt.timedelta(days=1); continue
            for hour in range(24):
                t = dt.datetime.combine(day,dt.time(hour))
                if pd.Timestamp(t) not in allowed:
                    logs.append([str(t),"excluded_by_old_log",""]); continue
                try:
                    u,v,dz,wp_now = m.read_wind(WIND_DIR / f"{t-dt.timedelta(hours=8):%Y%m%d%H}.nc", True)
                    m.check_grid(wp,wp_now,"wind")
                    u0,v0 = m.weight(u,v,dz,hw)
                    ub,vb = iw(u0),iw(v0)
                    valid_h = np.isfinite(hb) & (hb > 0)
                    ub,vb = np.where(valid_h,ub,np.nan),np.where(valid_h,vb,np.nan)
                    stats[season].add(t,ub[idx],vb[idx],nx,ny)
                    logs.append([str(t),"wind_ok",""])
                except Exception as error:
                    logs.append([str(t),"wind_error",repr(error)]); continue
                if season != "spring" or not RUN_REPLACEMENTS:
                    continue
                try:
                    cr,cp = m.read_pm(PM_DIR / f"{t:%Y%m%d%H}.nc",True)
                    if pm_points is None:
                        pm_points = cp; ip = m.LinearWeights(cp,points,True)
                        # The helper repeats a single nearest vertex for outside points.
                        # Reuse that audit instead of triangulating the million-node PM grid twice.
                        pi = ip.usable & ~(ip.vertices == ip.vertices[:, :1]).all(axis=1)
                        audit["pm_outside_nodes"] = [int((~pi[b.start:b.stop]).sum()) for b in bs]
                        audit["strict_core"] &= np.array([pi[b.start:b.stop].all() for b in bs], dtype=bool)
                    else:
                        m.check_grid(pm_points,cp,"PM")
                    cb = ip(cr)
                    fields = {"actual_spring_same_mask":(cb,hb,ub,vb)}
                    fields.update({k:(cb,hb,a,b) for k,(a,b) in wind_cases.items()})
                    names, flux, coverage = m.integrate_scenarios(bs,fields)
                    assert names == cases
                    for k,values in enumerate(flux):
                        ok = np.isfinite(values)
                        sums[k] += np.where(ok,np.abs(values)*FACTOR,0)
                        counts[k] += ok
                        for scope,mask in [("all_boundaries",np.ones(len(bs),bool)),("strict_core",audit.strict_core.to_numpy(bool))]:
                            if mask.any() and ok[mask].all():
                                scopes[scope][0][k] += np.abs(values[mask]).sum()*FACTOR
                                scopes[scope][1][k] += 1
                    logs.append([str(t),"experiment_ok",""])
                except Exception as error:
                    logs.append([str(t),"experiment_error",repr(error)])
            pd.DataFrame(logs,columns=["time_cst","status","detail"]).to_csv(out/"processing_log.csv",index=False,encoding="utf-8-sig")
            day += dt.timedelta(days=1)
    finally:
        audit.to_csv(out/"boundary_audit.csv",index=False,encoding="utf-8-sig")
        rows, details = [], []
        for season,a in stats.items():
            z = a.metrics()
            details.append(pd.concat([locations.assign(season=season),z],axis=1))
            for b,sl in zip(bs,groups):
                row = {"season":season,"boundary_name":b.name,
                       "length_m":float(weight[sl].sum()),
                       "min_valid_hours":int(z.iloc[sl].n_hours.min()),
                       "strict_core":bool(audit.loc[audit.boundary_name==b.name,"strict_core"].iloc[0])}
                for key in z.columns.difference(["n_hours","eligible_consecutive_pairs"]):
                    row[key] = wav(z.iloc[sl][key].to_numpy(),weight[sl])
                rows.append(row)
        pd.concat(details,ignore_index=True).to_csv(out/"boundary_endpoint_wind.csv",index=False,encoding="utf-8-sig")
        boundary = pd.DataFrame(rows)
        boundary.to_csv(out/"boundary_seasonal_wind.csv",index=False,encoding="utf-8-sig")
        summary = []
        for season in ("spring","winter"):
            for scope in ("all_boundaries","strict_core"):
                z = boundary[(boundary.season==season) & (boundary.min_valid_hours>0)]
                if scope=="strict_core": z=z[z.strict_core]
                row={"season":season,"scope":scope,"n_boundaries":len(z)}
                for key in z.columns:
                    if key not in ("season","boundary_name","length_m","min_valid_hours","strict_core"):
                        row[key]=wav(z[key].to_numpy(),z.length_m.to_numpy())
                summary.append(row)
        pd.DataFrame(summary).to_csv(out/"boundary_wind_summary.csv",index=False,encoding="utf-8-sig")
        for k,name in enumerate(cases):
            for j,b in enumerate(bs):
                exp_rows.append([b.name,name,int(counts[k,j]),float(ratio(sums[k,j],counts[k,j]))])
            for scope,(su,n) in scopes.items():
                exp_rows.append([scope,name,int(n[k]),float(ratio(su[k],n[k]))])
        ex = pd.DataFrame(exp_rows,columns=["scope","scenario","valid_hours","mean_abs_flux_t_day"])
        ex.to_csv(out/"speed_direction_experiments.csv",index=False,encoding="utf-8-sig")
        differences=[]
        for scope,z in ex.groupby("scope",sort=False):
            z=z.set_index("scenario")
            base=z.loc["actual_spring_same_mask","mean_abs_flux_t_day"]
            ss=z.loc["SS","mean_abs_flux_t_day"]
            for label,k in [("speed_only","WS"),("direction_only","SW"),("full_wind","WW")]:
                value=z.loc[k,"mean_abs_flux_t_day"]
                differences.append([scope,label,ss,value,value-ss,float(ratio(value-ss,ss)*100),
                                    float(ratio(value-ss,base)*100),base])
        pd.DataFrame(differences,columns=["scope","contrast","SS_control_t_day","replacement_t_day",
            "difference_t_day","percent_of_SS","percent_of_actual_spring_same_mask","actual_spring_same_mask_t_day"]
            ).to_csv(out/"speed_direction_contrasts.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(logs,columns=["time_cst","status","detail"]).to_csv(out/"processing_log.csv",index=False,encoding="utf-8-sig")
        meta={"expected_hours_in_attempted_days":expected,"direction_threshold_m_s":MEAN_WIND_EPS,
            "normal_sign_threshold_m_s":NORMAL_EPS,"n_boundaries":len(bs),
            "mode":"partial" if args.max_days else "full_requested_check_logs_and_counts",
            "speed_definition":"magnitude of seasonal mean vector, NOT mean hourly speed",
            "normal_definition":"left normal, NOT automatically city outward",
            "warnings":["legacy scope uses original fill/extrapolation; check strict_core",
                        "direction split uses a common nonweak-direction mask",
                        "normal-wind diagnostics are endpoint/length averages, NOT absolute whole-boundary flux",
                        "spatial persistence mean is mean of pointwise ratios",
                        "finite-data summary masks may differ by season; inspect min_valid_hours",
                        "contrasts are conditional mean-vector sensitivities, not independent causal contributions"]}
        (out/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("Finished. Check processing_log.csv, boundary_audit.csv, and valid_hours first.",flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--means-only",action="store_true")
    p.add_argument("--max-days",type=int)
    a=p.parse_args()
    self_test() if a.self_test else main(a)
