# -*- coding: utf-8 -*-
"""Hourly spring transport with winter scalar-mean speed / circular-mean direction.
Run this file as a normal Python script in PyCharm. No embedded test entry.
Requires the existing spring_winter_transport_experiments.py helper (not modified).
"""
from pathlib import Path
import datetime as dt
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

HELPER = Path(r'C:\Users\zyd\PycharmProjects\spring_winter_transport_experiments.py')
WIND_DIR = Path(r'E:\wind_24h')
PM_DIR = Path(r'E:\pm25_1km_24h_new')
PBL_DIR = Path(r'C:\Users\zyd\PycharmProjects\pbl_height')
BOUNDARY_DIR = Path(r'C:\Users\zyd\PycharmProjects\province_to_province_boundary_grid')
OLD_LOG = Path(r'C:\Users\zyd\Desktop\spring_winter_transport_experiments\processing_log.csv')
USE_OLD_LOG = True
OUTPUT_ROOT = Path(__file__).resolve().parent / 'hourly_wind_replacement_70_results'
START, END = '2023-03-01', '2024-02-29'
# Only exclude calm hours when ESTIMATING winter mean direction.
CALM_EPS = 0.05
# Diagnostic flag only: weak winter directions are retained, NOT excluded.
MIN_DIRECTION_RESULTANT = 0.05
MIN_WINTER_COVERAGE = 0.95
ZERO_EPS = 1e-12
EXPECTED_BOUNDARIES = 70
FACTOR = 86400e-12  # ug/s -> equivalent instantaneous rate t/day, NOT hourly mass
NAMES = ['actual_spring', 'winter_direction_only', 'winter_speed_only']


def winter_summary(u, v):
    """Synthetic-array helper for tests; axis 0 is time."""
    s = np.hypot(u, v)
    ok = np.isfinite(s)
    direction_ok = ok & (s >= CALM_EPS)
    cu = np.divide(u, s, out=np.full_like(s, np.nan), where=direction_ok)
    cv = np.divide(v, s, out=np.full_like(s, np.nan), where=direction_ok)
    a, b = np.nanmean(cu, axis=0), np.nanmean(cv, axis=0)
    r = np.hypot(a, b)
    dx = np.divide(a, r, out=np.full_like(r, np.nan), where=r > 0)
    dy = np.divide(b, r, out=np.full_like(r, np.nan), where=r > 0)
    return np.nanmean(np.where(ok, s, np.nan), axis=0), dx, dy, r


def replace(u, v, winter_speed, dx, dy, valid):
    speed = np.hypot(u, v)
    # Weak but nonzero spring winds still have a defined direction.
    # Exact zero has no direction: invalidate that hour globally, never invent one.
    mask = valid & np.isfinite(speed) & (speed > ZERO_EPS)
    def masked(a):
        return np.where(mask, a, np.nan)
    return [(masked(u), masked(v)),
            (masked(speed*dx), masked(speed*dy)),
            (masked(winter_speed*u/np.where(mask, speed, 1)),
             masked(winter_speed*v/np.where(mask, speed, 1)))]


def main():
    spec = importlib.util.spec_from_file_location('_transport_helper', HELPER)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    exec(compile(HELPER.read_text(encoding='utf-8-sig').replace(
        '@njit(cache=True)', '@njit(cache=False)'), str(HELPER), 'exec'), m.__dict__)
    m.BOUNDARY_DIR = BOUNDARY_DIR
    # Match the previous 70-boundary experiments; these are explicit assumptions.
    m.PBL_OUTSIDE_FILL = 500.0
    m.FALLBACK_TO_LOWEST_WIND = True
    m.MIN_BOUNDARY_LENGTH_COVERAGE = 1.0
    bs, points = m.load_boundaries()
    if len(bs) != EXPECTED_BOUNDARIES:
        raise ValueError(f'Expected 70 boundaries, found {len(bs)}; check BOUNDARY_DIR')
    allowed = None
    if USE_OLD_LOG:
        log = pd.read_csv(OLD_LOG)
        allowed = set(pd.to_datetime(log.loc[log.status == 'processed', 'time_cst']))
    out = OUTPUT_ROOT / dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    print('Output:', out, flush=True)
    logs, grid, interp = [], None, None
    def hours(months):
        for day in pd.date_range(START, END, freq='D'):
            if day.month in months:
                for h in range(24):
                    yield day.to_pydatetime()+dt.timedelta(hours=h)
    def wind(t):
        nonlocal grid, interp
        h = pd.read_csv(PBL_DIR / f'pbl_height_{t:%Y%m%d}.csv')
        hp = h[['lon', 'lat']].to_numpy(float)
        hv = h.pbl_heightchem.to_numpy(float)*m.PBL_SCALE
        u, v, dz, gp = m.read_wind(WIND_DIR / f'{t-dt.timedelta(hours=8):%Y%m%d%H}.nc', True)
        if grid is None:
            grid = gp
            interp = m.LinearWeights(grid, points, True)
        else:
            m.check_grid(grid, gp, 'wind')
        hw = m.interpolate_pbl(hp, hv, grid)
        hb = m.interpolate_pbl(hp, hv, points)
        a, b = m.weight(u, v, dz, hw)
        ub, vb = interp(a), interp(b)
        valid_h = np.isfinite(hb) & (hb > 0)
        return np.where(valid_h, ub, np.nan), np.where(valid_h, vb, np.nan), hb

    n = len(points)
    sums = np.zeros((3, n))
    counts = np.zeros((2, n), int)
    winter_hours = list(hours({12, 1, 2}))
    for i, t in enumerate(winter_hours):
        if allowed is not None and pd.Timestamp(t) not in allowed:
            logs.append([t, 'winter_excluded_by_old_log', ''])
            continue
        try:
            u, v, _ = wind(t)
            s = np.hypot(u, v)
            ok = np.isfinite(s)
            d = ok & (s >= CALM_EPS)
            sums[0] += np.where(ok, s, 0)
            sums[1] += np.divide(u, s, out=np.zeros(n), where=d)
            sums[2] += np.divide(v, s, out=np.zeros(n), where=d)
            counts[0] += ok
            counts[1] += d
            logs.append([t, 'winter_ok', ''])
        except Exception as e:
            logs.append([t, 'winter_error', repr(e)])
        if i % 24 == 0:
            print('Winter', t.date(), flush=True)
    mean_speed = sums[0]/np.where(counts[0] > 0, counts[0], np.nan)
    a = sums[1]/np.where(counts[1] > 0, counts[1], np.nan)
    b = sums[2]/np.where(counts[1] > 0, counts[1], np.nan)
    resultant = np.hypot(a, b)
    dx, dy = a/np.where(resultant > 0, resultant, np.nan), b/np.where(resultant > 0, resultant, np.nan)
    # Coverage and weak direction are audit flags, not reasons to drop boundaries.
    valid = np.isfinite(mean_speed) & np.isfinite(dx) & np.isfinite(dy) & (resultant > ZERO_EPS)
    weak_direction = np.isfinite(resultant) & (resultant < MIN_DIRECTION_RESULTANT)
    low_coverage = counts[0] < MIN_WINTER_COVERAGE*len(winter_hours)
    pd.DataFrame({'lon': points[:, 0], 'lat': points[:, 1],
        'winter_mean_scalar_speed_m_s': mean_speed, 'winter_direction_unit_east': dx,
        'winter_direction_unit_north': dy, 'circular_resultant': resultant,
        'winter_valid_hours': counts[0], 'winter_direction_hours': counts[1],
        'eligible': valid, 'weak_direction_flag': weak_direction,
        'low_coverage_flag': low_coverage}).to_csv(out/'winter_boundary_means.csv', index=False)
    core = np.array([valid[x.start:x.stop].all() for x in bs])
    pd.DataFrame({'boundary': [x.name for x in bs], 'winter_eligible': core,
        'retained': np.ones(len(bs), dtype=bool),
        'weak_direction_nodes': [int(weak_direction[x.start:x.stop].sum()) for x in bs],
        'low_coverage_nodes': [int(low_coverage[x.start:x.stop].sum()) for x in bs],
        'undefined_direction_nodes': [int((~valid[x.start:x.stop]).sum()) for x in bs],
        'min_winter_valid_hours': [int(counts[0,x.start:x.stop].min()) for x in bs]
        }).to_csv(out/'boundary_audit.csv', index=False)
    if not valid.all():
        pd.DataFrame(logs, columns=['time_cst', 'status', 'detail']).to_csv(out/'processing_log.csv', index=False)
        raise ValueError('Some winter directions are undefined (zero resultant or missing data). See boundary_audit.csv. No boundary was dropped and no arbitrary direction was assigned.')
    rows, pp, ip = [], None, None
    try:
        for i, t in enumerate(hours({3, 4, 5})):
            if allowed is not None and pd.Timestamp(t) not in allowed:
                logs.append([t, 'spring_excluded_by_old_log', ''])
                continue
            try:
                u, v, hb = wind(t)
                c, cp = m.read_pm(PM_DIR / f'{t:%Y%m%d%H}.nc', True)
                if pp is None:
                    pp, ip = cp, m.LinearWeights(cp, points, True)
                else:
                    m.check_grid(pp, cp, 'PM')
                cb = ip(c)
                fields = {name: (cb, hb, x, y) for name, (x, y) in zip(
                    NAMES, replace(u, v, mean_speed, dx, dy, valid))}
                _, flux, _ = m.integrate_scenarios(bs, fields)
                # All boundary and scenario means use the SAME complete hours.
                hour_valid = np.isfinite(flux).all()
                if not hour_valid:
                    logs.append([t, 'spring_incomplete_70_boundaries',
                                 'Common hour excluded: missing field or zero spring wind direction'])
                    flux[:] = np.nan
                for k, name in enumerate(NAMES):
                    for j, boundary in enumerate(bs):
                        f = flux[k, j]*FACTOR
                        rows.append([t, boundary.name, name, f, abs(f)])
                    for label, mask in [('all_boundaries', np.ones(len(bs), bool))]:
                        f = flux[k, mask]*FACTOR
                        value = np.abs(f).sum() if len(f) and np.isfinite(f).all() else np.nan
                        rows.append([t, label, name, np.nan, value])
                if hour_valid:
                    logs.append([t, 'spring_ok', ''])
            except Exception as e:
                logs.append([t, 'spring_error', repr(e)])
            if i % 24 == 0:
                print('Spring', t.date(), flush=True)
    finally:
        pd.DataFrame(logs, columns=['time_cst', 'status', 'detail']).to_csv(out/'processing_log.csv', index=False)
        f = pd.DataFrame(rows, columns=['time_cst', 'scope', 'scenario', 'signed_flux_t_day', 'abs_flux_t_day'])
        f.to_csv(out/'hourly_boundary_flux.csv', index=False)
        summary = f.groupby(['scope', 'scenario'], sort=False).agg(
            valid_hours=('abs_flux_t_day', 'count'), mean_abs_flux_t_day=('abs_flux_t_day', 'mean'),
            mean_signed_flux_t_day=('signed_flux_t_day', 'mean')).reset_index()
        base = summary[summary.scenario == NAMES[0]].set_index('scope').mean_abs_flux_t_day
        summary['change_t_day'] = summary.mean_abs_flux_t_day-summary.scope.map(base)
        summary['change_percent_actual_spring'] = 100*summary.change_t_day/summary.scope.map(base).replace(0, np.nan)
        summary.to_csv(out/'seasonal_comparisons.csv', index=False)
        (out/'metadata.json').write_text(json.dumps({'start': START, 'end': END,
            'direction': 'equal-hour circular mean of noncalm unit vectors at boundary points',
            'speed': 'mean hourly scalar speed, calm hours included',
            'calm_threshold_m_s': CALM_EPS, 'min_resultant': MIN_DIRECTION_RESULTANT,
            'min_winter_coverage': MIN_WINTER_COVERAGE, 'old_log_used': USE_OLD_LOG,
            'strict_spatial_domain': False, 'helper': str(HELPER),
            'n_boundaries': len(bs), 'weak_directions_retained': True,
            'pbl_outside_fill_m': 500.0, 'nearest_outside_wind_pm': True,
            'lowest_wind_fallback': True, 'spring_zero_direction_epsilon': ZERO_EPS,
            'common_complete_hours_for_all_70_boundaries': True,
            'warning': 'All 70 boundaries retained; weak winter directions flagged, not deleted. Means use identical complete spring hours. Legacy spatial fill/extrapolation assumptions apply. t/day is a rate, not hourly mass. No independent causal attribution.'}, indent=2), encoding='utf-8')
    print('Finished. Check processing_log.csv and valid_hours before interpretation.', flush=True)



if __name__ == '__main__':
    main()
