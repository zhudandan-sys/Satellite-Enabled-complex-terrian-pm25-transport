# -*- coding: utf-8 -*-
"""直接运行：使用CAMS程序同一算法、同一时次重算卫星浓度输送；不修改旧结果。
需要E盘原始PM/风/CAMS数据。本脚本只替换浓度读取器，复用cams_transport_common。
"""
from pathlib import Path
import importlib.util,sys,json,ast,re,unicodedata
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

ROOT=Path(r'C:\Users\zyd\PycharmProjects')
SAT_PM=Path(r'E:\pm25_1km_24h_new')
OUTPUT_ROOT=Path(__file__).resolve().parent/'satellite_matched_cams_results'
spec=importlib.util.spec_from_file_location('_matched_cams_engine',ROOT/'cams_transport_common.py')
engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine)
OriginalReader=engine.CAMSReader


def concentration_factor(units):
    """Normalize recognised unit notation without guessing an unknown unit."""
    u=unicodedata.normalize('NFKC',str(units)).lower()
    u=u.replace('µ','u').replace('μ','u').replace('−','-').replace('⁻','-')
    u=u.replace(r'\mu','u').replace(r'\micro','u')
    u=u.replace(r'\mathrm','').replace(r'\text','')
    u=u.replace(r'\cdot','').replace('·','')
    u=u.replace(r'\,','').replace(r'\;','').replace(r'\!','')
    u=re.sub(r'[\s$^{}()]','',u).replace('**','')
    match=re.fullmatch(r'(kg|mg|ug|ng)(?:/m3|m-3)',u)
    if match is None:
        raise ValueError(f'Unrecognised satellite concentration units {units!r} (normalised {u!r}); verify units before running.')
    return {'kg':1e9,'mg':1e3,'ug':1.0,'ng':1e-3}[match.group(1)]

class SatelliteReader(OriginalReader):
    def interpolate(self,index,points):
        import datetime as dt
        cst=self.times[index]+dt.timedelta(hours=8)
        with Dataset(str(SAT_PM/f'{cst:%Y%m%d%H}.nc')) as nc:
            c=np.asarray(np.ma.filled(nc['pm25'][:].astype(float),np.nan)).squeeze()
            lon=np.asarray(np.ma.filled(nc['lon'][:].astype(float),np.nan)).squeeze()
            lat=np.asarray(np.ma.filled(nc['lat'][:].astype(float),np.nan)).squeeze()
            units=getattr(nc['pm25'],'units','')
        c=c*concentration_factor(units)
        if lon.ndim==lat.ndim==1 and c.ndim==2 and c.shape==(len(lat),len(lon)):
            ix,iy=np.argsort(lon),np.argsort(lat);x,y=lon[ix],lat[iy];z=c[np.ix_(iy,ix)]
        else:
            if lon.size!=c.size or lat.size!=c.size:raise ValueError('PM coordinates/data size mismatch')
            x,y=np.unique(lon),np.unique(lat)
            if len(x)*len(y)!=c.size:raise ValueError('Expected regular rectangular satellite grid')
            ix=np.searchsorted(x,lon.ravel());iy=np.searchsorted(y,lat.ravel())
            if len(np.unique(iy*len(x)+ix))!=c.size:raise ValueError('Duplicate satellite coordinates')
            z=np.full((len(y),len(x)),np.nan);z[iy,ix]=c.ravel()
        z[z<0]=np.nan
        return RegularGridInterpolator((y,x),z,bounds_error=False,fill_value=np.nan)(points[:,[1,0]])

def describe(pairs,keys):
    rows=[]
    for key,g in pairs.groupby(keys):
        key=key if isinstance(key,tuple) else (key,)
        x,y=g.satellite,g.cams; d=y-x
        valid=(x.abs()>1e-6)&(y.abs()>1e-6)
        rows.append([*key,len(g),x.mean(),y.mean(),x.corr(y),x.corr(y,method='spearman'),d.mean(),np.sqrt((d*d).mean()),d.abs().mean(),float((np.sign(x[valid])==np.sign(y[valid])).mean()) if valid.any() else np.nan])
    return pd.DataFrame(rows,columns=keys+['n','satellite_mean_t_day','cams_mean_t_day','pearson_r','spearman_r','mean_difference_cams_minus_sat','rms_difference','mean_abs_difference','direction_agreement_nonzero'])

def main():
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    tree=ast.parse((ROOT/'workflow图片制作'/'城间.py').read_text(encoding='utf-8-sig'))
    groups=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='special_channels' for x in n.targets))
    arrows={x for g in groups.values() for x in g['channels']}
    for kind,folder in [('open','transport_open_cams_results'),('city','transport_city_cams_results')]:
        source=sorted(p for p in (ROOT/folder).iterdir() if (p/'metadata.json').exists())[-1]
        meta=json.loads((source/'metadata.json').read_text(encoding='utf-8'));cfg=meta['config'].copy()
        cfg['OUTPUT_ROOT']=str(OUTPUT_ROOT/kind)
        # Fail before any wind/annual calculation if units/grid/coordinates are wrong.
        first=pd.read_csv(source/'native_time_flux.csv',nrows=1)
        if first.empty: raise ValueError('CAMS reference has no native times')
        utc=pd.Timestamp(first.time_utc.iloc[0]).to_pydatetime()
        probe=SatelliteReader.__new__(SatelliteReader);probe.times=[utc]
        value=probe.interpolate(0,np.array([[110.0,35.0]]))
        import datetime as dt
        probe_file=SAT_PM/f'{utc+dt.timedelta(hours=8):%Y%m%d%H}.nc'
        with Dataset(str(probe_file)) as nc:
            probe_units=getattr(nc['pm25'],'units','')
        print(f'PREFLIGHT OK: {probe_file}; units={probe_units!r}; factor={concentration_factor(probe_units)}; sample={value}',flush=True)
        if not np.isfinite(value).all(): raise ValueError('Preflight point has missing concentration')
        engine.CAMSReader=SatelliteReader
        output=engine.run(cfg)
        note=json.loads((output/'metadata.json').read_text(encoding='utf-8'));note['actual_concentration_source']=str(SAT_PM);note['comparison_cams_source']=str(source);note['comparison']='CAMS times, identical geometry/wind/PBL/missing-data algorithm; satellite concentration sampled natively at 1km; not a source-tagging validation'
        (output/'metadata.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
        a=pd.read_csv(output/'native_time_flux.csv');b=pd.read_csv(source/'native_time_flux.csv')
        keys=['time_cst','boundary_name']
        p=a[keys+['flux_t_day_equivalent']].rename(columns={'flux_t_day_equivalent':'satellite'}).merge(b[keys+['flux_t_day_equivalent']].rename(columns={'flux_t_day_equivalent':'cams'}),on=keys,validate='one_to_one')
        if kind=='open':p=p[p.boundary_name.isin(arrows)]
        p=p.dropna(subset=['satellite','cams']);p['date_cst']=p.time_cst.str[:10]
        month=pd.to_datetime(p.time_cst).dt.month
        p['season']=month.map({12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'})
        p.to_csv(output/'matched_native_time_pairs.csv',index=False)
        describe(p,['season','boundary_name']).to_csv(output/'native_time_comparison_metrics.csv',index=False)
        daily=p.groupby(['date_cst','season','boundary_name'],as_index=False).agg(n=('satellite','size'),satellite=('satellite','mean'),cams=('cams','mean'))
        daily=daily[daily.n==8]
        daily.to_csv(output/'matched_complete_daily_pairs.csv',index=False)
        describe(daily,['season','boundary_name']).to_csv(output/'daily_comparison_metrics.csv',index=False)
        daily.groupby(['season','boundary_name'],as_index=False)[['satellite','cams']].mean().to_csv(output/'seasonal_signed_patterns.csv',index=False)
        print('MATCHED COMPARISON DONE:',output,flush=True)
    print('All done. Differences indicate dataset consistency, not error against ground truth.')

if __name__=='__main__':main()
