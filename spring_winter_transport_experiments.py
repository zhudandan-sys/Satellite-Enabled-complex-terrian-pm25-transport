"""春冬PM2.5开边界传输通量及单变量平均场替换试验。

安装：pip install numpy pandas scipy netCDF4 numba
运行：python spring_winter_transport_experiments.py
自检：python spring_winter_transport_experiments.py --self-test

数据约定与transport_24h.py一致：
  PM：YYYYMMDDHH.nc，北京时间；变量pm25、lon、lat。
  风：YYYYMMDDHH.nc，UTC；变量u、v、height、lon、lat。
      height为自下而上各层厚度(m)，不是高度或气压。
  PBL：pbl_height_YYYYMMDD.csv，日期按北京时间解释；
       列lon、lat、pbl_heightchem。
  边界CSV：沿用pd.read_csv(index_col=0)，余下第一列lat、第二列lon。
       点顺序定义左法向；开边界不闭合、不自动反向。
  平均场：读取上一版平均场程序输出的NC，若无NC则读取同名CSV。

重要：
  1. 所有替换仅用于春季小时；冬季实际作为固定对照。
  2. 春季全部试验使用每小时共同有效的原始相邻线段；不会跨缺测点连线。
  3. H_fixedU只改变通量式的H，保留原加权风；H_reweight同时重算
     对应高度内的加权风，二者不作独立贡献相加。
  4. 实际春冬1.27倍必须在本程序中重新核对，不能预先写死。
  5. 区域汇总为输入文件所列开边界的交换指标，不是研究区外边界净流出，
     更不等同于污染来源贡献。相同物理开边界不要重复放入两个CSV。
  6. 原脚本的Delaunay权重维度与缺测点跨越连线在这里已修正，
     因此新的实际基准可能与旧输出不同。边界积分本身不重复乘长度。
"""
from scipy.interpolate import griddata
from pathlib import Path
from dataclasses import dataclass
import datetime as dt
import json
import sys
import time

import numpy as np
import pandas as pd
try:
    from netCDF4 import Dataset
except ImportError:
    Dataset = None
from scipy.spatial import Delaunay, cKDTree
try:
    from numba import njit
except ImportError:
    # 无Numba仍可自检或串行运行；正式全年计算建议安装以加速。
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


# ======================== 用户配置 ========================
WIND_DIR = Path(r'E:\wind_24h')
PM25_DIR = Path(r'E:\pm25_1km_24h_new')
PBL_DIR = Path(r'C:\Users\zyd\PycharmProjects\pbl_height')
BOUNDARY_DIR = Path(r'C:\Users\zyd\PycharmProjects\province_to_province_boundary_grid')
MEAN_DIR = Path(r'E:\spring_winter_mean_fields')
OUTPUT_DIR = Path(r'E:\spring_winter_transport_experiments')

START_DATE = dt.date(2023, 3, 1)
END_DATE = dt.date(2024, 2, 29)       # 包括结束日
SPRING_MONTHS = {3, 4, 5}
WINTER_MONTHS = {12, 1, 2}

# 上一版平均PBLH提供两种口径；默认选择按成功共同小时加权的均值。
PBLH_MEAN_COLUMN = 'pblh_common_hour_weighted_mean_m'
# 若需要严格日均替换，改为'pblh_daily_mean_m'，并记录这一选择。

PBL_OUTSIDE_FILL = 500.0            # 沿用原代码；设np.nan可关闭固定值填充
ALLOW_NEAREST_OUTSIDE = True       # PM、风场插值域外沿用最近点
RECOMPUTE_WIND_FOR_REPLACED_H = True
FALLBACK_TO_LOWEST_WIND = True
REQUIRE_FULL_PBL_COVERAGE = False   # False沿用原代码的已覆盖层加权
MIN_VERTICAL_COVERAGE = 0.99

# 严格要求完整边界：1.0。若下调，不对缺失线段外推补足质量。
MIN_BOUNDARY_LENGTH_COVERAGE = 1.0
CHECK_COORDINATES_EVERY_FILE = False
SAVE_HOURLY_FILES = True
EARTH_RADIUS_M = 6371004.0
UG_S_TO_T_DAY = 86400.0e-12

# 按原脚本单位；如原数据单位不同，必须修改。
PM_SCALE = 1.0                     # ug/m3
WIND_SCALE = 1.0                   # m/s
THICKNESS_SCALE = 1.0              # m
PBL_SCALE = 1.0                    # m

SCENARIOS = [
    'S_actual',
    'S_C_spring_mean', 'S_C_winter_mean',
    'S_U_spring_mean', 'S_U_winter_mean',
    'S_H_spring_mean_fixedU', 'S_H_winter_mean_fixedU',
    'S_all_spring_mean', 'S_all_winter_mean',
]
if RECOMPUTE_WIND_FOR_REPLACED_H:
    SCENARIOS += ['S_H_spring_mean_reweight', 'S_H_winter_mean_reweight']


# ======================== 读取工具 ========================
def arr(nc, name):
    variable = nc.variables[name]
    values = np.asarray(np.ma.filled(variable[:], np.nan), dtype=float)
    for axis in range(len(variable.dimensions) - 1, -1, -1):
        if variable.dimensions[axis].lower() in ('time', 'times', 'valid_time'):
            if values.shape[axis] != 1:
                raise ValueError(f'{name}一个文件包含多个时次：{values.shape}')
            values = np.take(values, 0, axis=axis)
    return values


def points_from_nc(nc, spatial_shape):
    lon, lat = np.squeeze(arr(nc, 'lon')), np.squeeze(arr(nc, 'lat'))
    n = int(np.prod(spatial_shape))
    if lon.shape == lat.shape and lon.size == n:
        points = np.column_stack([lon.ravel(), lat.ravel()])
    elif (len(spatial_shape) == 2 and lon.ndim == lat.ndim == 1
          and lon.size == spatial_shape[1] and lat.size == spatial_shape[0]):
        x, y = np.meshgrid(lon, lat)
        points = np.column_stack([x.ravel(), y.ravel()])
    else:
        raise ValueError(f'坐标和数据维度不匹配：{lon.shape}, {lat.shape}, {spatial_shape}')
    if not np.isfinite(points).all():
        raise ValueError('经纬度存在无效值')
    return points


def read_pm(path, coordinates=False):
    with Dataset(str(path)) as nc:
        c = arr(nc, 'pm25') * PM_SCALE
        if c.ndim not in (1, 2):
            raise ValueError(f'pm25维度异常：{c.shape}')
        points = points_from_nc(nc, c.shape) if coordinates else None
    return c.ravel(), points


def read_wind(path, coordinates=False):
    with Dataset(str(path)) as nc:
        u, v, thickness = arr(nc, 'u'), arr(nc, 'v'), arr(nc, 'height')
        if u.shape != v.shape or u.shape != thickness.shape:
            raise ValueError('u、v、height形状必须一致')
        if u.ndim not in (2, 3):
            raise ValueError(f'多层风维度异常：{u.shape}')
        points = points_from_nc(nc, u.shape[1:]) if coordinates else None
    return (np.ascontiguousarray(u.reshape(u.shape[0], -1) * WIND_SCALE),
            np.ascontiguousarray(v.reshape(v.shape[0], -1) * WIND_SCALE),
            np.ascontiguousarray(thickness.reshape(thickness.shape[0], -1) * THICKNESS_SCALE),
            points)


def check_grid(reference, current, name):
    if reference.shape != current.shape or not np.allclose(reference, current, atol=1e-6, rtol=0):
        raise ValueError(f'{name}格点顺序或位置发生变化')


def load_mean(stem, columns):
    nc_path, csv_path = MEAN_DIR / f'{stem}.nc', MEAN_DIR / f'{stem}.csv'
    if nc_path.exists():
        with Dataset(str(nc_path)) as nc:
            points = np.column_stack([arr(nc, 'lon').ravel(), arr(nc, 'lat').ravel()])
            fields = {name: arr(nc, name).ravel() for name in columns}
    elif csv_path.exists():
        frame = pd.read_csv(csv_path)
        points = frame[['lon', 'lat']].to_numpy(float)
        fields = {name: frame[name].to_numpy(float) for name in columns}
    else:
        raise FileNotFoundError(f'未找到季节平均场：{nc_path}或{csv_path}')
    return points, fields


def reorder_exact(source_points, values, target_points):
    """平均场与原生网格只允许同点重排，不在这里偷偷重网格。"""
    if source_points.shape == target_points.shape and np.allclose(source_points, target_points, atol=1e-7, rtol=0):
        return values
    distance, index = cKDTree(source_points).query(target_points)
    if np.any(distance > 1e-6):
        raise ValueError('平均风场与小时风场不是同一网格，不能直接替换')
    return values[index]


# ======================== 空间插值 ========================
class LinearWeights:
    """二维Delaunay重心插值；正确使用transform[:, :2, :]。"""
    def __init__(self, source, target, nearest_outside=True):
        source, target = np.asarray(source), np.asarray(target)
        if len(np.unique(source, axis=0)) != len(source):
            raise ValueError('插值源点存在重复坐标')
        tri = Delaunay(source)
        simplex = tri.find_simplex(target)
        inside = simplex >= 0
        self.vertices = np.zeros((len(target), 3), dtype=int)
        self.weights = np.zeros((len(target), 3), dtype=float)
        self.usable = inside.copy()
        self.n_outside = int((~inside).sum())
        if inside.any():
            transform = tri.transform[simplex[inside]]
            delta = target[inside] - transform[:, 2, :]
            first_two = np.einsum('nij,nj->ni', transform[:, :2, :], delta)
            weights = np.column_stack([first_two, 1.0 - first_two.sum(axis=1)])
            weights = np.clip(weights, 0.0, 1.0)
            self.weights[inside] = weights / weights.sum(axis=1, keepdims=True)
            self.vertices[inside] = tri.simplices[simplex[inside]]
        if nearest_outside and (~inside).any():
            _, index = cKDTree(source).query(target[~inside])
            self.vertices[~inside, :] = index[:, None]
            self.weights[~inside, 0] = 1.0
            self.usable[~inside] = True

    def __call__(self, values):
        selected = np.asarray(values)[self.vertices]
        active = self.weights > 1e-12
        valid = self.usable & np.all(~active | np.isfinite(selected), axis=1)
        output = np.full(len(self.weights), np.nan)
        terms = np.where(active & np.isfinite(selected), selected, 0.0) * self.weights
        output[valid] = terms[valid].sum(axis=1)
        return output


def interpolate_pbl(points, values, target):
    valid = np.isfinite(points).all(axis=1) & np.isfinite(values)
    if valid.sum() < 3:
        raise ValueError('PBLH有效插值点不足')
    return griddata(points[valid], values[valid], target,
                    method='linear', fill_value=PBL_OUTSIDE_FILL)


# ======================== 累积层厚加权 ========================
@njit(cache=True)
def weighted_wind(u, v, thickness, h, fallback, require_full, min_coverage):
    nz, n = u.shape
    out_u, out_v = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(h[i]) or h[i] <= 0:
            if fallback:
                out_u[i], out_v[i] = u[0, i], v[0, i]
            continue
        cumulative = 0.0
        total = 0.0
        su, sv = 0.0, 0.0
        bad = False
        for k in range(nz):
            if cumulative >= h[i]:
                break
            dz = thickness[k, i]
            if not np.isfinite(dz) or dz <= 0:
                bad = True
                break
            portion = min(dz, h[i] - cumulative)
            if portion > 0:
                if not np.isfinite(u[k, i]) or not np.isfinite(v[k, i]):
                    bad = True
                    break
                su += u[k, i] * portion
                sv += v[k, i] * portion
                total += portion
            cumulative += dz
        if bad:
            continue
        if total > 0:
            if require_full and total / h[i] < min_coverage:
                continue
            out_u[i], out_v[i] = su / total, sv / total
        elif fallback:
            out_u[i], out_v[i] = u[0, i], v[0, i]
    return out_u, out_v


def weight(u, v, thickness, h):
    return weighted_wind(u, v, thickness, h, FALLBACK_TO_LOWEST_WIND,
                         REQUIRE_FULL_PBL_COVERAGE, MIN_VERTICAL_COVERAGE)


# ======================== 开边界积分 ========================
@dataclass
class Boundary:
    name: str
    start: int
    stop: int
    nx_length: np.ndarray
    ny_length: np.ndarray
    lengths: np.ndarray


def geometry(points):
    """法向(-dy,dx)已含线段长度，积分时不能再乘长度。"""
    delta = np.diff(points, axis=0)
    latitude = np.deg2rad(0.5 * (points[1:, 1] + points[:-1, 1]))
    dx = EARTH_RADIUS_M * np.deg2rad(delta[:, 0]) * np.cos(latitude)
    dy = EARTH_RADIUS_M * np.deg2rad(delta[:, 1])
    return -dy, dx, np.hypot(dx, dy)


def load_boundaries():
    boundaries, arrays, offset = [], [], 0
    for path in sorted(BOUNDARY_DIR.glob('*.csv')):
        frame = pd.read_csv(path, index_col=0)
        if frame.shape[1] < 2:
            raise ValueError(f'{path.name}在读取索引后不足两列')
        points = np.column_stack([frame.iloc[:, 1].to_numpy(float),
                                  frame.iloc[:, 0].to_numpy(float)])
        if len(points) < 2 or not np.isfinite(points).all():
            raise ValueError(f'{path.name}边界点不足或包含无效坐标')
        nx, ny, lengths = geometry(points)
        if lengths.sum() <= 0:
            raise ValueError(f'{path.name}边界总长度为零')
        boundaries.append(Boundary(path.stem, offset, offset + len(points), nx, ny, lengths))
        arrays.append(points)
        offset += len(points)
    if not arrays:
        raise FileNotFoundError(f'未找到边界CSV：{BOUNDARY_DIR}')
    return boundaries, np.vstack(arrays)


def integrate_scenarios(boundaries, fields):
    """fields[name]=(C,H,U,V)，全部已在边界点上。
    同一小时所有试验共享有效端点及相邻线段，避免掩膜差异伪装替换效应。
    """
    names = list(fields)
    npoints = len(next(iter(fields.values()))[0])
    common = np.ones(npoints, dtype=bool)
    for c, h, u, v in fields.values():
        common &= (np.isfinite(c) & np.isfinite(h) & np.isfinite(u)
                   & np.isfinite(v) & (c >= 0) & (h > 0))
    flux = np.full((len(names), len(boundaries)), np.nan)
    coverage = np.zeros(len(boundaries))
    for j, boundary in enumerate(boundaries):
        sl = slice(boundary.start, boundary.stop)
        mask = common[sl]
        valid_segments = mask[:-1] & mask[1:]
        coverage[j] = boundary.lengths[valid_segments].sum() / boundary.lengths.sum()
        if coverage[j] + 1e-12 < MIN_BOUNDARY_LENGTH_COVERAGE or not valid_segments.any():
            continue
        for k, name in enumerate(names):
            c, h, u, v = (x[sl] for x in fields[name])
            dot0 = boundary.nx_length * u[:-1] + boundary.ny_length * v[:-1]
            dot1 = boundary.nx_length * u[1:] + boundary.ny_length * v[1:]
            segments = 0.5 * (c[:-1] * h[:-1] * dot0 + c[1:] * h[1:] * dot1)
            flux[k, j] = segments[valid_segments].sum()
    return names, flux, coverage


# ======================== 主过程及输出 ========================
def dates():
    day = START_DATE
    while day <= END_DATE:
        if day.month in SPRING_MONTHS | WINTER_MONTHS:
            yield day
        day += dt.timedelta(days=1)


def paths(day, hour):
    cst = dt.datetime.combine(day, dt.time(hour))
    utc = cst - dt.timedelta(hours=8)
    return cst, PM25_DIR / f'{cst:%Y%m%d%H}.nc', WIND_DIR / f'{utc:%Y%m%d%H}.nc'


def initialize(boundaries, allpoints):
    for day in dates():
        for hour in range(24):
            _, pm_path, wind_path = paths(day, hour)
            if pm_path.exists() and wind_path.exists():
                _, pm_points = read_pm(pm_path, True)
                _, _, _, wind_points = read_wind(wind_path, True)
                break
        else:
            continue
        break
    else:
        raise FileNotFoundError('找不到共同存在的PM及风场小时文件')

    pm_interp = LinearWeights(pm_points, allpoints, ALLOW_NEAREST_OUTSIDE)
    wind_interp = LinearWeights(wind_points, allpoints, ALLOW_NEAREST_OUTSIDE)
    means = {}
    for season in ('spring', 'winter'):
        cp, cf = load_mean(f'{season}_pm25_mean', ['pm25_mean_ug_m3'])
        wp, wf = load_mean(f'{season}_pbl_weighted_wind_mean', ['u_mean_m_s', 'v_mean_m_s'])
        hp, hf = load_mean(f'{season}_pblh_mean_native_grid', [PBLH_MEAN_COLUMN])
        # 平均浓度与小时浓度使用同一原生网格及插值权重，避免重复剖分百万格点。
        cb = pm_interp(reorder_exact(cp, cf['pm25_mean_ug_m3'], pm_points))
        ub = wind_interp(reorder_exact(wp, wf['u_mean_m_s'], wind_points))
        vb = wind_interp(reorder_exact(wp, wf['v_mean_m_s'], wind_points))
        hb = interpolate_pbl(hp, hf[PBLH_MEAN_COLUMN], allpoints)
        hw = interpolate_pbl(hp, hf[PBLH_MEAN_COLUMN], wind_points)
        means[season] = {'C': cb, 'H': hb, 'U': ub, 'V': vb, 'H_wind': hw}

    audit = {'pm_interp_outside_points': pm_interp.n_outside,
             'wind_interp_outside_points': wind_interp.n_outside}
    return pm_points, wind_points, pm_interp, wind_interp, means, audit


def add_stat(store, season, name, values):
    key = (season, name)
    if key not in store:
        store[key] = {'sum': np.zeros(len(values)), 'abs_sum': np.zeros(len(values)),
                      'n': np.zeros(len(values), dtype=np.int64)}
    valid = np.isfinite(values)
    store[key]['sum'][valid] += values[valid]
    store[key]['abs_sum'][valid] += np.abs(values[valid])
    store[key]['n'][valid] += 1


def main():
    if Dataset is None:
        raise ImportError('缺少netCDF4，请先运行：pip install netCDF4')
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_HOURLY_FILES:
        (OUTPUT_DIR / 'hourly').mkdir(exist_ok=True)
    boundaries, allpoints = load_boundaries()
    pm_points, wind_points, pm_interp, wind_interp, means, interp_audit = initialize(boundaries, allpoints)
    spring_mean, winter_mean = means['spring'], means['winter']
    store, total_store, logs, daily_rows = {}, {}, [], []
    expected = {'spring': 0, 'winter': 0}
    for day in dates():
        season = 'spring' if day.month in SPRING_MONTHS else 'winter'
        expected[season] += 24
        print(f'处理北京时间日期：{day} ({season})', flush=True)
        pbl_path = PBL_DIR / f'pbl_height_{day:%Y%m%d}.csv'
        try:
            frame = pd.read_csv(pbl_path)
            p = frame[['lon', 'lat']].to_numpy(float)
            h = frame['pbl_heightchem'].to_numpy(float) * PBL_SCALE
            daily_h_boundary = interpolate_pbl(p, h, allpoints)
            daily_h_wind = interpolate_pbl(p, h, wind_points)
        except Exception as error:
            logs.append({'time_cst': str(day), 'status': 'pbl_error', 'detail': str(error)})
            continue

        hourly_rows, day_store = [], {}
        for hour in range(24):
            cst, pm_path, wind_path = paths(day, hour)
            if not pm_path.exists() or not wind_path.exists():
                logs.append({'time_cst': str(cst), 'status': 'missing_file', 'detail': f'PM={pm_path.exists()}, wind={wind_path.exists()}'})
                continue
            try:
                c_raw, cp = read_pm(pm_path, CHECK_COORDINATES_EVERY_FILE)
                u, v, thickness, wp = read_wind(wind_path, CHECK_COORDINATES_EVERY_FILE)
                if CHECK_COORDINATES_EVERY_FILE:
                    check_grid(pm_points, cp, 'PM')
                    check_grid(wind_points, wp, '风场')
                if len(c_raw) != len(pm_points) or u.shape[1] != len(wind_points):
                    raise ValueError('小时数据网格尺寸改变')
                cb = pm_interp(c_raw)
                uw, vw = weight(u, v, thickness, daily_h_wind)
                ub, vb = wind_interp(uw), wind_interp(vw)
                hb = daily_h_boundary
                actual = (cb, hb, ub, vb)

                if season == 'winter':
                    fields = {'W_actual': actual}
                else:
                    fields = {
                        'S_actual': actual,
                        'S_C_spring_mean': (spring_mean['C'], hb, ub, vb),
                        'S_C_winter_mean': (winter_mean['C'], hb, ub, vb),
                        'S_U_spring_mean': (cb, hb, spring_mean['U'], spring_mean['V']),
                        'S_U_winter_mean': (cb, hb, winter_mean['U'], winter_mean['V']),
                        'S_H_spring_mean_fixedU': (cb, spring_mean['H'], ub, vb),
                        'S_H_winter_mean_fixedU': (cb, winter_mean['H'], ub, vb),
                        'S_all_spring_mean': tuple(spring_mean[x] for x in ('C', 'H', 'U', 'V')),
                        'S_all_winter_mean': tuple(winter_mean[x] for x in ('C', 'H', 'U', 'V')),
                    }
                    if RECOMPUTE_WIND_FOR_REPLACED_H:
                        for s, m in [('spring', spring_mean), ('winter', winter_mean)]:
                            uh, vh = weight(u, v, thickness, m['H_wind'])
                            fields[f'S_H_{s}_mean_reweight'] = (cb, m['H'], wind_interp(uh), wind_interp(vh))

                names, flux, cover = integrate_scenarios(boundaries, fields)
                # 拒绝空结果，不把异常当作零通量。
                if not np.isfinite(flux).any():
                    raise ValueError('本小时所有边界均未达到共同有效长度阈值')
                for k, name in enumerate(names):
                    add_stat(store, season, name, flux[k])
                    add_stat(day_store, season, name, flux[k])
                    # 区域统计只使用全部边界均有效的小时，不变更边界集合。
                    aggregate = (np.array([np.sum(np.abs(flux[k])), np.sum(flux[k])])
                                 if np.isfinite(flux[k]).all() else np.array([np.nan, np.nan]))
                    add_stat(total_store, season, name, aggregate)
                    if SAVE_HOURLY_FILES:
                        for j, boundary in enumerate(boundaries):
                            hourly_rows.append({'time_cst': str(cst), 'season': season, 'scenario': name,
                                                'boundary_name': boundary.name, 'flux_ug_s': flux[k, j],
                                                'flux_rate_t_day': flux[k, j] * UG_S_TO_T_DAY,
                                                'valid_length_fraction': cover[j]})
                logs.append({'time_cst': str(cst), 'status': 'processed',
                             'detail': f'min_length_coverage={cover.min():.6f}'})
            except Exception as error:
                logs.append({'time_cst': str(cst), 'status': 'hour_error', 'detail': str(error)})
                print(f'  小时{hour:02d}未使用：{error}', flush=True)

        for (_, name), stat in day_store.items():
            for j, boundary in enumerate(boundaries):
                n = int(stat['n'][j])
                daily_rows.append({'date_cst': str(day), 'season': season, 'scenario': name,
                                   'boundary_name': boundary.name, 'valid_hours': n,
                                   'observed_signed_mass_t': stat['sum'][j] * 3600e-12 if n else np.nan,
                                   'mean_signed_rate_t_day': stat['sum'][j] / n * UG_S_TO_T_DAY if n else np.nan,
                                   'mean_abs_hourly_rate_t_day': stat['abs_sum'][j] / n * UG_S_TO_T_DAY if n else np.nan})
        if SAVE_HOURLY_FILES and hourly_rows:
            pd.DataFrame(hourly_rows).to_csv(OUTPUT_DIR / 'hourly' / f'flux_{day:%Y%m%d}.csv', index=False, encoding='utf-8-sig')
        # 每日更新日志，长任务中断后也能看到错误原因。
        pd.DataFrame(logs).to_csv(OUTPUT_DIR / 'processing_log.csv', index=False, encoding='utf-8-sig')

    rows = []
    for (season, name), stat in store.items():
        for j, boundary in enumerate(boundaries):
            n = int(stat['n'][j])
            rows.append({'season': season, 'scenario': name, 'boundary_name': boundary.name,
                         'valid_hours': n, 'expected_hours': expected[season],
                         'hour_coverage': n / expected[season] if expected[season] else np.nan,
                         'mean_signed_rate_t_day': stat['sum'][j] / n * UG_S_TO_T_DAY if n else np.nan,
                         'mean_abs_hourly_rate_t_day': stat['abs_sum'][j] / n * UG_S_TO_T_DAY if n else np.nan,
                         'observed_signed_mass_t': stat['sum'][j] * 3600e-12 if n else np.nan})
    for (season, name), stat in total_store.items():
        n = int(stat['n'][0])
        rows.append({'season': season, 'scenario': name, 'boundary_name': '__ALL_BOUNDARIES__',
                     'valid_hours': n, 'expected_hours': expected[season],
                     'hour_coverage': n / expected[season] if expected[season] else np.nan,
                     'mean_signed_rate_t_day': stat['sum'][1] / n * UG_S_TO_T_DAY if n else np.nan,
                     'mean_abs_hourly_rate_t_day': stat['sum'][0] / n * UG_S_TO_T_DAY if n else np.nan,
                     'observed_signed_mass_t': stat['sum'][1] * 3600e-12 if n else np.nan})
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError('无有效结果，请查看processing_log.csv')
    summary.to_csv(OUTPUT_DIR / 'seasonal_flux_summary.csv', index=False, encoding='utf-8-sig')
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(OUTPUT_DIR / 'daily_flux_summary.csv', index=False, encoding='utf-8-sig')
    export_comparisons(summary, daily)

    metadata = {'start_date_cst': str(START_DATE), 'end_date_cst': str(END_DATE),
                'pbl_mean_column': PBLH_MEAN_COLUMN, 'scenarios': SCENARIOS + ['W_actual'],
                'boundary_normal': 'left normal determined by original point order; not necessarily city outward',
                'units': 'C ug/m3; H and layer thickness m; U,V m/s; integral ug/s',
                'pbl_outside_fill': float(PBL_OUTSIDE_FILL) if np.isfinite(PBL_OUTSIDE_FILL) else None,
                'minimum_boundary_length_coverage': MIN_BOUNDARY_LENGTH_COVERAGE,
                'common_spring_mask': 'all spring scenarios use the same original adjacent valid segments each hour',
                'interpolation_audit': interp_audit,
                'notes': [
                    'mean_abs_hourly_rate是先对边界小时积分取绝对值再平均，不是逐线段取绝对值后积分。',
                    '__ALL_BOUNDARIES__先加总各开边界小时通量的绝对值，再对完整小时平均。',
                    'daily_abs_signed_rate是每日有符号平均率取绝对值后再跨完整日平均。',
                    '不同统计口径可能得到不同春冬比，需明确原论文1.27倍对应哪一种。',
                    'H_fixedU仅改变通量式H；H_reweight同时改变H和PBL内加权风。',
                    '实际基准已采用正确重心插值及不跨缺测点的梯形积分，可能不同于原代码输出。',
                    '本程序不提供独立因果贡献率或来源贡献率，也不含置信区间。',
                ]}
    with open(OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f'完成。输出：{OUTPUT_DIR}；耗时{(time.time()-started)/60:.1f}分钟', flush=True)


def safe_ratio(numerator, denominator):
    return numerator / denominator if np.isfinite(denominator) and denominator > 1e-12 else np.nan


def export_comparisons(summary, daily):
    indexed = summary.set_index(['boundary_name', 'scenario'])
    baseline, replacements = [], []
    pairs = [
        ('C', 'S_C_spring_mean', 'S_C_winter_mean'),
        ('U', 'S_U_spring_mean', 'S_U_winter_mean'),
        ('H_fixedU', 'S_H_spring_mean_fixedU', 'S_H_winter_mean_fixedU'),
    ]
    if RECOMPUTE_WIND_FOR_REPLACED_H:
        pairs += [('H_reweight', 'S_H_spring_mean_reweight', 'S_H_winter_mean_reweight')]
    for boundary_name in summary.boundary_name.unique():
        if (boundary_name, 'S_actual') not in indexed.index or (boundary_name, 'W_actual') not in indexed.index:
            continue
        a = indexed.loc[(boundary_name, 'S_actual')]
        w = indexed.loc[(boundary_name, 'W_actual')]
        baseline.append({'boundary_name': boundary_name,
                         'spring_mean_signed_rate_t_day': a.mean_signed_rate_t_day,
                         'winter_mean_signed_rate_t_day': w.mean_signed_rate_t_day,
                         'spring_mean_abs_hourly_rate_t_day': a.mean_abs_hourly_rate_t_day,
                         'winter_mean_abs_hourly_rate_t_day': w.mean_abs_hourly_rate_t_day,
                         'actual_spring_winter_ratio': safe_ratio(a.mean_abs_hourly_rate_t_day, w.mean_abs_hourly_rate_t_day),
                         'spring_valid_hours': a.valid_hours, 'winter_valid_hours': w.valid_hours})
        for variable, own, swapped in pairs:
            b = indexed.loc[(boundary_name, own)]
            c = indexed.loc[(boundary_name, swapped)]
            qa, qb, qc, qw = [x.mean_abs_hourly_rate_t_day for x in (a, b, c, w)]
            replacements.append({'boundary_name': boundary_name, 'variable': variable,
                                 'A_actual_spring_t_day': qa, 'B_own_mean_t_day': qb,
                                 'C_winter_mean_replacement_t_day': qc, 'D_actual_winter_t_day': qw,
                                 'actual_ratio_A_over_D': safe_ratio(qa, qw),
                                 'own_mean_ratio_B_over_D': safe_ratio(qb, qw),
                                 'replacement_ratio_C_over_D': safe_ratio(qc, qw),
                                 'averaging_difference_A_minus_B_t_day': qa - qb,
                                 'seasonal_mean_difference_B_minus_C_t_day': qb - qc,
                                 'total_difference_A_minus_C_t_day': qa - qc,
                                 'replacement_reduction_percent_relative_B': 100 * safe_ratio(qb-qc, qb),
                                 'net_signed_difference_C_minus_B_t_day': c.mean_signed_rate_t_day - b.mean_signed_rate_t_day})
    pd.DataFrame(baseline).to_csv(OUTPUT_DIR / 'baseline_spring_winter.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(replacements).to_csv(OUTPUT_DIR / 'replacement_comparisons.csv', index=False, encoding='utf-8-sig')

    # 额外口径：仅使用24小时完整日，先求每天有符号平均，再取绝对值跨日平均。
    full = daily[daily.valid_hours == 24].copy()
    if not full.empty:
        full['daily_abs_signed_rate_t_day'] = full.mean_signed_rate_t_day.abs()
        daily_stats = full.groupby(['season', 'scenario', 'boundary_name'], as_index=False).agg(
            daily_abs_signed_rate_t_day=('daily_abs_signed_rate_t_day', 'mean'),
            complete_days=('date_cst', 'nunique'))
        daily_stats.to_csv(OUTPUT_DIR / 'complete_day_statistics.csv', index=False, encoding='utf-8-sig')


def self_test():
    # 正确重心权重应精确插值线性函数。
    source = np.array([[0., 0.], [1., 0.], [0., 1.]])
    target = np.array([[.2, .3]])
    assert np.allclose(LinearWeights(source, target)(1+2*source[:, 0]+3*source[:, 1]), [2.3])

    # PBL=1000m，层厚300/500/400m，最后层只取200m。
    u = np.array([[1.], [3.], [5.]])
    v = np.zeros_like(u)
    thick = np.array([[300.], [500.], [400.]])
    wu, wv = weighted_wind(u, v, thick, np.array([1000.]), False, True, .99)
    assert np.allclose(wu, [2.8]) and np.allclose(wv, [0.])

    # 东西向开边界左法向朝北，均匀向北风应得到正通量。
    points = np.array([[105., 35.], [106., 35.], [107., 35.]])
    nx, ny, lengths = geometry(points)
    b = Boundary('test', 0, 3, nx, ny, lengths)
    c, h = np.full(3, 10.), np.full(3, 1000.)
    _, flux, coverage = integrate_scenarios([b], {'test': (c, h, np.zeros(3), np.ones(3))})
    assert np.allclose(flux[0, 0], 10*1000*lengths.sum())
    assert np.allclose(coverage, [1.])

    # 中间点缺测必须使两条相邻线段均无效，不能跨点重新连接。
    c[1] = np.nan
    _, flux, _ = integrate_scenarios([b], {'test': (c, h, np.zeros(3), np.ones(3))})
    assert np.isnan(flux[0, 0])
    print('自检通过：重心插值、部分层厚加权、开边界单位及缺测线段处理。')


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        self_test()
    else:
        main()
