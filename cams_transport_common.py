"""CAMS surface PM2.5 + original multilayer winds/daily PBL boundary transport.

No temporal interpolation or spatial extrapolation. See CAMS_transport_README.md.
"""
from pathlib import Path
import datetime as dt
import json
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from scipy.spatial import Delaunay
from scipy.interpolate import griddata, RegularGridInterpolator

RATE_TO_T_DAY = 86400.0e-12


def array(var):
    return np.asarray(np.ma.filled(var[:].astype(float), np.nan))


class LinearWeights:
    """Fixed-grid barycentric interpolation; outside convex hull stays NaN."""
    def __init__(self, source, target):
        source, target = np.asarray(source), np.asarray(target)
        self.mask = np.isfinite(source).all(axis=1)
        tri = Delaunay(source[self.mask])
        simplex = tri.find_simplex(target)
        self.inside = simplex >= 0
        safe = np.maximum(simplex, 0)
        transform = tri.transform[safe]
        bary = np.einsum('nij,nj->ni', transform[:, :2], target-transform[:, 2])
        self.weights = np.column_stack((bary, 1-bary.sum(axis=1)))
        self.vertices = tri.simplices[safe]

    def __call__(self, values):
        values = np.asarray(values).ravel()[self.mask]
        sampled = values[self.vertices]
        # A missing vertex with exactly zero weight does not contaminate a node.
        active = np.abs(self.weights) > 1e-12
        valid = self.inside & np.all(np.isfinite(sampled) | ~active, axis=1)
        result = np.full(len(valid), np.nan)
        result[valid] = np.sum(np.where(active[valid], sampled[valid], 0)
                               * self.weights[valid], axis=1)
        return result


class CAMSReader:
    def __init__(self, path, variable, utc_offset=0):
        self.nc = Dataset(str(path))
        try:
            self.var = self.nc.variables[variable]
            self.lon = array(self.nc.variables['longitude'])
            self.lat = array(self.nc.variables['latitude'])
            if self.lon.ndim != 1 or self.lat.ndim != 1:
                raise ValueError('CAMS latitude/longitude must be 1D.')
            self.ix, self.iy = np.argsort(self.lon), np.argsort(self.lat)
            self.lon, self.lat = self.lon[self.ix], self.lat[self.iy]
            if np.any(np.diff(self.lon) <= 0) or np.any(np.diff(self.lat) <= 0):
                raise ValueError('Duplicate/non-increasing CAMS coordinates.')
            self.time_name = 'valid_time' if 'valid_time' in self.nc.variables else 'time'
            tv = self.nc.variables[self.time_name]
            if tv.ndim != 1:
                raise ValueError('Expected a 1D validity-time coordinate.')
            if self.time_name == 'time' and 'step' in self.nc.variables:
                if np.any(array(self.nc.variables['step']) != 0):
                    raise ValueError('Nonzero forecast step requires valid_time.')
            times = num2date(tv[:], tv.units, calendar=getattr(tv, 'calendar', 'standard'))
            self.times = [dt.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
                          - dt.timedelta(hours=utc_offset) for t in times]
            if len(self.times) < 2 or any(b <= a for a, b in zip(self.times, self.times[1:])):
                raise ValueError('CAMS validity times must be unique and increasing.')
            gaps = np.array([(b-a).total_seconds() for a,b in zip(self.times,self.times[1:])])
            self.interval_seconds = float(np.min(gaps))
            self.regular = bool(np.all(gaps == self.interval_seconds))
            if not self.regular or 86400 % self.interval_seconds:
                raise ValueError('Expected regular CAMS sampling dividing 24 hours.')
            self.time_dim = tv.dimensions[0]
            self.lat_dim = self.nc.variables['latitude'].dimensions[0]
            self.lon_dim = self.nc.variables['longitude'].dimensions[0]
            self.units = getattr(self.var, 'units', '')
            unit = self.units.lower().replace('μ','u').replace('µ','u')
            unit = unit.replace(' ', '').replace('**','').replace('^','').replace('−','-')
            if unit in ('ugm-3', 'ug/m3'):
                self.factor = 1.0
            elif unit in ('kgm-3', 'kg/m3'):
                self.factor = 1e9
            else:
                raise ValueError(f'Unrecognized concentration units: {self.units!r}')
        except Exception:
            self.nc.close()
            raise

    def interpolate(self, index, points):
        keys, dims = [], []
        for dim, size in zip(self.var.dimensions, self.var.shape):
            if dim == self.time_dim:
                keys.append(index)
            elif dim in (self.lat_dim, self.lon_dim):
                keys.append(slice(None)); dims.append(dim)
            elif size == 1:
                keys.append(0)
            else:
                raise ValueError(f'Unsupported CAMS dimension: {dim}')
        if self.time_dim not in self.var.dimensions:
            raise ValueError('PM2.5 variable has no validity-time dimension.')
        data = np.asarray(np.ma.filled(self.var[tuple(keys)].astype(float), np.nan))
        data = np.transpose(data, [dims.index(self.lat_dim), dims.index(self.lon_dim)])
        data = data[np.ix_(self.iy, self.ix)] * self.factor
        data[data < 0] = np.nan
        interpolator = RegularGridInterpolator((self.lat, self.lon), data,
                                               bounds_error=False, fill_value=np.nan)
        return interpolator(points[:, [1,0]])

    def close(self):
        self.nc.close()


def wind_data(path):
    with Dataset(str(path)) as nc:
        u, v, height = [array(nc.variables[name]) for name in ('u','v','height')]
        lon, lat = array(nc.variables['lon']), array(nc.variables['lat'])
        if u.ndim == 4 and u.shape[0] == 1:
            u, v = u[0], v[0]
        if height.ndim == 4 and height.shape[0] == 1:
            height = height[0]
        if u.ndim not in (2,3) or u.shape != v.shape:
            raise ValueError(f'Unsupported wind shape: {u.shape}, {v.shape}')
        if lon.ndim == lat.ndim == 1 and u.ndim == 3:
            lon, lat = np.meshgrid(lon, lat)
        if lon.shape != lat.shape or lon.size != np.prod(u.shape[1:]):
            raise ValueError('Wind coordinates and horizontal shape differ.')
        if height.ndim == 1 and len(height) == u.shape[0]:
            height = height.reshape((-1,) + (1,)*(u.ndim-1))
        height = np.broadcast_to(height, u.shape)
        points = np.column_stack((lon.ravel(), lat.ravel()))
        return points, u.reshape(u.shape[0],-1), v.reshape(v.shape[0],-1), height.reshape(u.shape[0],-1)


def weighted_wind(u, v, height, pbl, height_kind='thickness'):
    """Bottom-up layers; partial top layer; no lowest-layer fallback/truncation."""
    if height_kind == 'thickness':
        thickness = height
    elif height_kind == 'layer_top_agl':
        thickness = np.diff(height, axis=0, prepend=np.zeros_like(height[:1]))
    else:
        raise ValueError('HEIGHT_KIND must be thickness or layer_top_agl; not layer centers.')
    top = np.cumsum(thickness, axis=0)
    bottom = top-thickness
    weights = np.maximum(0, np.minimum(top, pbl[None,:])-bottom)
    active = weights > 0
    valid = (np.isfinite(pbl) & (pbl > 0)
             & np.all(np.isfinite(thickness) & (thickness > 0), axis=0)
             & (top[-1] >= pbl-1e-6)
             & np.all((np.isfinite(u) & np.isfinite(v)) | ~active, axis=0))
    uw, vw = np.full(pbl.shape,np.nan), np.full(pbl.shape,np.nan)
    total = weights[:,valid].sum(axis=0)
    uw[valid] = (np.where(active[:,valid],u[:,valid],0)*weights[:,valid]).sum(axis=0)/total
    vw[valid] = (np.where(active[:,valid],v[:,valid],0)*weights[:,valid]).sum(axis=0)/total
    return uw, vw


def geometry(points, closed):
    points = np.asarray(points, dtype=float)
    if not np.isfinite(points).all():
        raise ValueError('Boundary contains missing coordinates; not bridging gaps.')
    if closed and np.array_equal(points[0],points[-1]):
        points = points[:-1]
    if len(points) < (3 if closed else 2):
        raise ValueError('Too few boundary vertices.')
    i = np.arange(len(points) if closed else len(points)-1)
    j = (i+1) % len(points)
    delta = np.radians(points[j]-points[i])
    # Preserve the original segment-midpoint latitude conversion to metres.
    scale = np.cos(np.radians((points[i,1]+points[j,1])/2))
    dx, dy = 6371004*scale*delta[:,0], 6371004*delta[:,1]
    nx, ny = -dy, dx  # left normal, with segment length included exactly once
    if closed:
        xy = points-points[0]
        area2 = np.sum(xy[i,0]*xy[j,1]-xy[j,0]*xy[i,1])
        if abs(area2) < 1e-12:
            raise ValueError('Zero-area city boundary.')
        if area2 > 0:  # counterclockwise -> right normal is outward
            nx, ny = -nx, -ny
    return points, i, j, nx, ny, np.hypot(dx,dy)


def boundary_flux(geom, u, v, c, h):
    _, i,j,nx,ny,length = geom
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(c) & np.isfinite(h) & (h>0) & (c>=0)
    edges = valid[i] & valid[j]
    coverage = float(length[edges].sum()/length.sum())
    if not np.all(edges | (length == 0)):
        return np.nan, coverage
    flux = 0.5*(c[i]*h[i]*(u[i]*nx+v[i]*ny)+c[j]*h[j]*(u[j]*nx+v[j]*ny))
    return float(flux[length>0].sum()), coverage


def summarize_daily(frame, dates, names, expected, available):
    rows = []
    grouped = {(date,name): g for (date,name),g in frame.groupby(['date_cst','boundary_name'])}
    for date in dates:
        for name in names:
            g = grouped.get((str(date),name))
            values = np.array([]) if g is None else g['flux_ug_s'].dropna().to_numpy()
            n = len(values)
            rows.append(dict(date_cst=str(date), boundary_name=name,
                             mean_signed_flux_t_day=np.mean(values)*RATE_TO_T_DAY if n else np.nan,
                             mean_abs_flux_t_day=np.mean(np.abs(values))*RATE_TO_T_DAY if n else np.nan,
                             valid_samples=n, expected_samples=expected,
                             cams_samples_available=available.get(str(date),0),
                             complete_day=n==expected))
    return pd.DataFrame(rows)


def run(config):
    cfg = dict(config)
    out = Path(cfg['OUTPUT_ROOT']) / dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    print('Output:', out, flush=True)
    reader = CAMSReader(cfg['CAMS_FILE'],cfg['CAMS_VARIABLE'],cfg.get('CAMS_UTC_OFFSET',0))
    try:
        closed = cfg['CLOSED_BOUNDARY']
        files = sorted(Path(cfg['BOUNDARY_DIR']).glob('*.csv'))
        targets = cfg.get('TARGET_CITIES')
        if targets:
            files = [Path(cfg['BOUNDARY_DIR'])/f'{city}_boundary.csv' for city in targets]
        geoms = {}
        for file in files:
            df = pd.read_csv(file,index_col=0)
            pts = df[['lon','lat']].to_numpy(float) if {'lon','lat'} <= set(df) else df.iloc[:,[1,0]].to_numpy(float)
            name = file.stem.removesuffix('_boundary') if closed else file.stem
            geoms[name] = geometry(pts,closed)
        if not geoms:
            raise ValueError('No boundary files found.')
        points = np.vstack([g[0] for g in geoms.values()])
        slices, offset = {}, 0
        for name,g in geoms.items():
            slices[name] = slice(offset, offset+len(g[0])); offset += len(g[0])
        inside = ((points[:,0]>=reader.lon.min()) & (points[:,0]<=reader.lon.max())
                  & (points[:,1]>=reader.lat.min()) & (points[:,1]<=reader.lat.max()))
        audit = pd.DataFrame([dict(boundary_name=name, points=len(g[0]),
                                  cams_outside_points=int((~inside[slices[name]]).sum()))
                              for name,g in geoms.items()])
        audit.to_csv(out/'boundary_audit.csv',index=False,encoding='utf-8-sig')
        if (audit.cams_outside_points > 0).any():
            print('WARNING: Some boundaries exceed CAMS coverage; their flux will be NaN. See boundary_audit.csv.',flush=True)
        start, end = dt.date.fromisoformat(cfg['START_DATE']), dt.date.fromisoformat(cfg['END_DATE'])
        selected = [(k,t,t+dt.timedelta(hours=8)) for k,t in enumerate(reader.times)
                    if start <= (t+dt.timedelta(hours=8)).date() <= end]
        if not selected:
            raise ValueError('No CAMS times in selected CST date range.')
        metadata = dict(config=cfg, concentration_units=reader.units,
                        concentration_factor_to_ug_m3=reader.factor,
                        first_utc=str(reader.times[0]), last_utc=str(reader.times[-1]),
                        native_interval_hours=reader.interval_seconds/3600,
                        concentration_time_coordinate=reader.time_name,
                        daily_result='Mean instantaneous rate, t/day equivalent; NOT integrated daily mass.',
                        sign='Positive outward' if closed else 'Positive toward left of ordered polyline',
                        geometry='Original segment-midpoint latitude conversion to metres',
                        pbl='Daily CST height; repeated at native CAMS times; not hourly PBL observations',
                        missing='No extrapolation, no zero filling, complete spatial boundary required')
        (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        rows, logs, cache = [], [], None
        for k,utc,cst in selected:
            try:
                if utc.minute or utc.second:
                    raise ValueError('Wind filenames only support whole-hour validity times.')
                wind_path = Path(cfg['WIND_DIR']) / utc.strftime('%Y%m%d%H.nc')
                wp,u,v,h = wind_data(wind_path)
                key = str(cst.date())
                if cache is None or cache[0] != key or not np.array_equal(cache[1],wp,equal_nan=True):
                    df = pd.read_csv(Path(cfg['PBL_DIR'])/f'pbl_height_{cst:%Y%m%d}.csv')
                    p = df[['lon','lat','pbl_heightchem']].to_numpy(float)
                    p = p[np.isfinite(p).all(axis=1) & (p[:,2]>0)]
                    targets_all = np.vstack((wp,points))
                    target_valid = np.isfinite(targets_all).all(axis=1)
                    ph = np.full(len(targets_all),np.nan)
                    ph[target_valid] = griddata(p[:,:2],p[:,2],targets_all[target_valid],method='linear',fill_value=np.nan)
                    cache = key, wp.copy(), ph[:len(wp)], ph[len(wp):], LinearWeights(wp,points)
                    print(key, flush=True)
                uw,vw = weighted_wind(u,v,h,cache[2],cfg.get('HEIGHT_KIND','thickness'))
                ub,vb = cache[4](uw),cache[4](vw)
                cb = reader.interpolate(k,points)
                for name,geom in geoms.items():
                    s = slices[name]
                    flux,coverage = boundary_flux(geom,ub[s],vb[s],cb[s],cache[3][s])
                    rows.append(dict(time_utc=utc.isoformat(),time_cst=cst.isoformat(),date_cst=key,
                                     boundary_name=name,flux_ug_s=flux,
                                     flux_t_day_equivalent=flux*RATE_TO_T_DAY,
                                     spatial_coverage_fraction=coverage))
                logs.append(dict(time_utc=str(utc),time_cst=str(cst),status='processed',
                                 invalid_boundaries=sum(not np.isfinite(r['flux_ug_s']) for r in rows[-len(geoms):]),error=''))
            except Exception as exc:
                logs.append(dict(time_utc=str(utc),time_cst=str(cst),status='error',
                                 invalid_boundaries=len(geoms),error=f'{type(exc).__name__}: {exc}'))
                print(f'ERROR {utc}: {exc}',flush=True)
        frame = pd.DataFrame(rows,columns=['time_utc','time_cst','date_cst','boundary_name','flux_ug_s','flux_t_day_equivalent','spatial_coverage_fraction'])
        frame.to_csv(out/'native_time_flux.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(logs).to_csv(out/'processing_log.csv',index=False,encoding='utf-8-sig')
        dates = [start+dt.timedelta(days=n) for n in range((end-start).days+1)]
        available = {}
        for _,_,cst in selected:
            available[str(cst.date())] = available.get(str(cst.date()),0)+1
        daily = summarize_daily(frame,dates,list(geoms),int(86400/reader.interval_seconds),available)
        daily.to_csv(out/'daily_flux_summary.csv',index=False,encoding='utf-8-sig')
        daily[daily.complete_day].to_csv(out/'complete_day_flux.csv',index=False,encoding='utf-8-sig')
        print(f'Finished: {len(selected)} native times; {int(daily.complete_day.sum())} complete boundary-days. Check processing_log.csv and boundary_audit.csv.',flush=True)
        if not len(frame) or not np.isfinite(frame.flux_ug_s).any():
            raise RuntimeError(f'No valid flux was calculated. Inspect audit/log files in {out}')
        return out
    finally:
        reader.close()
