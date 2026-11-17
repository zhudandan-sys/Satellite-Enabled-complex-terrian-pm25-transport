"""
参照 transport_24h.py 的实际数据格式和累积层厚加权方法，
计算春季、冬季三个变量的逐格点平均场。

数据格式：
1. PM2.5：
   E:/pm25_1km_24h_new/YYYYMMDDHH.nc
   文件名时间为北京时间CST。
   变量：pm25、lon、lat。

2. 风场：
   E:/wind_24h/YYYYMMDDHH.nc
   文件名时间为UTC。
   变量：u、v、height、lon、lat。
   注意：height在原代码中表示各层厚度，单位m，
         不是层中心高度，也不是累计高度。
   风层必须从近地面向高空排列。

3. PBLH：
   C:/Users/zyd/PycharmProjects/pbl_height/pbl_height_YYYYMMDD.csv
   按原代码，将文件日期解释为北京时间日期。
   列：date、pbl_heightchem、lon、lat。
   同一天的24个小时使用同一PBLH。

计算顺序：
   当日PBLH线性插值到风场网格
       → 每小时累积层厚并截取PBL内有效厚度
       → 分别加权u、v
       → 对春季、冬季分别进行时间平均。

依赖：
    pip install numpy pandas scipy netCDF4 numba

说明：
- 不计算边界通量，所以这里不需要城市边界。
- PM2.5、风场和PBLH保留各自原生网格。
- 输出平均风矢量u_mean、v_mean，而不是只输出风速大小。
- 春冬均值按有效样本计算，同时输出有效样本数和覆盖率。
- PBLH缺测后的最低层风回退、插值域外填500m等规则，
  默认沿用原代码，并在配置和日志中明确记录。
"""

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from numba import njit
from scipy.interpolate import griddata


# ============================================================
# 1. 路径与参数
# ============================================================

PM25_DIR = Path(r"E:\pm25_1km_24h_new")
WIND_DIR = Path(r"E:\wind_24h")
PBL_DIR = Path(r"C:\Users\zyd\PycharmProjects\pbl_height")

OUTPUT_DIR = Path(r"E:\spring_winter_mean_fields")

PM25_FILE_TEMPLATE = "{}{}.nc"
WIND_FILE_TEMPLATE = "{}{}.nc"
PBL_FILE_TEMPLATE = "pbl_height_{}.csv"

# 与原代码一致：北京时间日期，起止日期均包括。
START_DATE = dt.date(2023, 3, 1)
END_DATE = dt.date(2024, 2, 29)

SPRING_MONTHS = {3, 4, 5}
WINTER_MONTHS = {12, 1, 2}

# 与原代码一致：PM文件名为CST，风场文件名为UTC。
CST_MINUS_UTC_HOURS = 8

# 插值域外的PBLH填充值，原代码为500m。
# 如不希望使用固定值，可改为np.nan。
PBL_OUTSIDE_FILL_VALUE = 500.0

# 沿用原代码：PBLH无效或无法获得有效层厚时，使用最低层风。
# False：相应加权风设为NaN，不纳入平均。
FALLBACK_TO_LOWEST_WIND = True

# 原代码在模式有效层没有完全覆盖PBLH时，仍用已覆盖部分加权。
# False保持原行为；True则要求覆盖比例达到阈值。
REQUIRE_FULL_PBL_COVERAGE = False
MIN_PBL_COVERAGE = 0.99

# 两类小时文件均存在且读取成功，才同时纳入PM与风场平均。
USE_COMMON_HOURS = True

# 原代码假设风层自下向上排列。
# 如果文件明确是自上向下排列，改为False。
LAYERS_BOTTOM_TO_TOP = True

# 是否逐文件检查经纬度；开启更严格，但增加I/O。
CHECK_COORDINATES_EVERY_FILE = False

# 输出CSV及NetCDF。
SAVE_CSV = True
SAVE_NETCDF = True

# 单位转换；按原代码默认：
# pm25为ug/m3，u/v为m/s，height及PBLH为m。
PM25_SCALE = 1.0
WIND_SCALE = 1.0
LAYER_THICKNESS_SCALE = 1.0
PBLH_SCALE = 1.0


# ============================================================
# 2. 日期与文件名
# ============================================================

def season_of(date):
    if date.month in SPRING_MONTHS:
        return "spring"
    if date.month in WINTER_MONTHS:
        return "winter"
    return None


def iter_dates():
    current = START_DATE
    while current <= END_DATE:
        if season_of(current) is not None:
            yield current
        current += dt.timedelta(days=1)


def hourly_paths(cst_datetime):
    utc_datetime = cst_datetime - dt.timedelta(
        hours=CST_MINUS_UTC_HOURS
    )

    pm_path = PM25_DIR / PM25_FILE_TEMPLATE.format(
        cst_datetime.strftime("%Y%m%d"),
        f"{cst_datetime.hour:02d}",
    )
    wind_path = WIND_DIR / WIND_FILE_TEMPLATE.format(
        utc_datetime.strftime("%Y%m%d"),
        f"{utc_datetime.hour:02d}",
    )
    return pm_path, wind_path


# ============================================================
# 3. NetCDF读取与坐标处理
# ============================================================

def read_array(nc, name):
    if name not in nc.variables:
        raise KeyError(
            f"缺少变量{name}；实际变量：{list(nc.variables.keys())}"
        )

    variable = nc.variables[name]
    values = np.asarray(
        np.ma.filled(variable[:], np.nan),
        dtype=np.float64,
    )
    dimensions = list(variable.dimensions)

    # 每个文件只包含一个小时；允许存在长度为1的时间维。
    for axis in range(len(dimensions) - 1, -1, -1):
        if dimensions[axis].lower() in {
            "time", "times", "valid_time"
        }:
            if values.shape[axis] != 1:
                raise ValueError(
                    f"{name}时间维长度不为1：{values.shape}。"
                    "当前程序要求每个NC对应一个小时。"
                )
            values = np.take(values, 0, axis=axis)

    return values


def coordinate_points(nc, spatial_shape):
    lon = read_array(nc, "lon")
    lat = read_array(nc, "lat")

    lon = np.squeeze(lon)
    lat = np.squeeze(lat)
    n_points = int(np.prod(spatial_shape))

    # 二维经纬度或一维成对点坐标。
    if lon.shape == lat.shape and lon.size == n_points:
        points = np.column_stack([lon.ravel(), lat.ravel()])

    # 规则网格的一维经纬度轴。
    elif (
        len(spatial_shape) == 2
        and lon.ndim == 1
        and lat.ndim == 1
        and lon.size == spatial_shape[1]
        and lat.size == spatial_shape[0]
    ):
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        points = np.column_stack([
            lon_grid.ravel(), lat_grid.ravel()
        ])
    else:
        raise ValueError(
            f"坐标与数据网格不匹配：lon={lon.shape}，"
            f"lat={lat.shape}，空间维度={spatial_shape}"
        )

    if not np.isfinite(points).all():
        raise ValueError("网格经纬度存在NaN或无穷值。")

    return points


def check_points(reference, current, label):
    if reference.shape != current.shape:
        raise ValueError(f"{label}网格点数发生变化。")
    if not np.allclose(reference, current, atol=1e-6, rtol=0):
        raise ValueError(
            f"{label}网格位置或点排列发生变化，"
            "不能按数组位置直接累加平均。"
        )


def read_pm25(path, need_coordinates=False):
    with Dataset(str(path), "r") as nc:
        pm = read_array(nc, "pm25") * PM25_SCALE

        if pm.ndim not in (1, 2):
            raise ValueError(f"PM2.5维度异常：{pm.shape}")

        points = (
            coordinate_points(nc, pm.shape)
            if need_coordinates else None
        )

    return pm.ravel(), points


def read_wind(path, need_coordinates=False):
    with Dataset(str(path), "r") as nc:
        u = read_array(nc, "u") * WIND_SCALE
        v = read_array(nc, "v") * WIND_SCALE
        thickness = (
            read_array(nc, "height") * LAYER_THICKNESS_SCALE
        )

        if u.shape != v.shape or u.shape != thickness.shape:
            raise ValueError(
                f"u、v、height形状不一致："
                f"{u.shape}, {v.shape}, {thickness.shape}"
            )

        if u.ndim == 3:
            nz, ny, nx = u.shape
            spatial_shape = (ny, nx)
            u = u.reshape(nz, -1)
            v = v.reshape(nz, -1)
            thickness = thickness.reshape(nz, -1)

        elif u.ndim == 2:
            nz, n_points = u.shape
            spatial_shape = (n_points,)

        else:
            raise ValueError(f"风场维度异常：{u.shape}")

        points = (
            coordinate_points(nc, spatial_shape)
            if need_coordinates else None
        )

    if not LAYERS_BOTTOM_TO_TOP:
        u = u[::-1]
        v = v[::-1]
        thickness = thickness[::-1]

    return (
        np.ascontiguousarray(u),
        np.ascontiguousarray(v),
        np.ascontiguousarray(thickness),
        points,
    )


# ============================================================
# 4. 沿用原代码的累积层厚加权
# ============================================================

@njit(cache=True)
def compute_weighted_wind(
    u, v, thickness, pblh,
    fallback_to_lowest, require_full_coverage, min_coverage
):
    """
    u、v、thickness：(nz, N)
    pblh：(N,)

    累积高度：
        第k层底部 = 前面各层厚度之和；
        第k层顶部 = 第k层底部 + 第k层厚度。

    加权：
        U = sum(u_k * PBL内有效层厚) / sum(PBL内有效层厚)
        V = sum(v_k * PBL内有效层厚) / sum(PBL内有效层厚)

    返回：
        加权U、加权V、垂直覆盖率、是否使用最低层回退。
    """
    nz, n_points = u.shape

    weighted_u = np.full(n_points, np.nan)
    weighted_v = np.full(n_points, np.nan)
    coverage = np.full(n_points, np.nan)
    fallback_used = np.zeros(n_points, dtype=np.int64)

    for i in range(n_points):
        h = pblh[i]

        if not np.isfinite(h) or h <= 0:
            if fallback_to_lowest:
                weighted_u[i] = u[0, i]
                weighted_v[i] = v[0, i]
                fallback_used[i] = 1
            continue

        cumulative_height = 0.0
        total_thickness = 0.0
        sum_u = 0.0
        sum_v = 0.0
        bad_profile = False

        for k in range(nz):
            if cumulative_height >= h:
                break

            dz = thickness[k, i]

            # 无法确定层厚时，不跳过该层继续累加，
            # 否则会错误改变后续层的垂直位置。
            if not np.isfinite(dz) or dz <= 0:
                bad_profile = True
                break

            effective_dz = min(dz, h - cumulative_height)

            if effective_dz > 0:
                if not np.isfinite(u[k, i]) or not np.isfinite(v[k, i]):
                    bad_profile = True
                    break

                sum_u += u[k, i] * effective_dz
                sum_v += v[k, i] * effective_dz
                total_thickness += effective_dz

            cumulative_height += dz

        coverage[i] = total_thickness / h

        if bad_profile:
            # 有效边界层范围内出现缺失风层时，不生成虚假的完整加权风。
            continue

        if total_thickness > 0:
            if require_full_coverage and coverage[i] < min_coverage:
                continue

            weighted_u[i] = sum_u / total_thickness
            weighted_v[i] = sum_v / total_thickness

        elif fallback_to_lowest:
            weighted_u[i] = u[0, i]
            weighted_v[i] = v[0, i]
            fallback_used[i] = 1

    return weighted_u, weighted_v, coverage, fallback_used


# ============================================================
# 5. 逐格点累加器
# ============================================================

class MeanAccumulator:
    def __init__(self, variable_names, n_points):
        self.names = list(variable_names)
        self.sums = {
            name: np.zeros(n_points, dtype=np.float64)
            for name in self.names
        }
        self.count = np.zeros(n_points, dtype=np.int64)

    def add(self, values, weight=1):
        """
        多变量共同有效时才累加。
        风场中u、v、speed使用相同有效小时。
        weight可用于把日PBLH按成功处理的小时数加权。
        """
        valid = np.ones(self.count.size, dtype=bool)

        for name in self.names:
            array = np.asarray(values[name], dtype=np.float64)
            if array.shape != self.count.shape:
                raise ValueError(f"{name}数组尺寸变化。")
            valid &= np.isfinite(array)

        for name in self.names:
            array = np.asarray(values[name], dtype=np.float64)
            self.sums[name][valid] += array[valid] * weight

        self.count[valid] += int(weight)

    def means(self):
        return {
            name: np.divide(
                self.sums[name],
                self.count,
                out=np.full(self.count.shape, np.nan),
                where=self.count > 0,
            )
            for name in self.names
        }


def write_fields(path_stem, points, fields, units):
    if SAVE_CSV:
        frame = pd.DataFrame({
            "lon": points[:, 0],
            "lat": points[:, 1],
            **fields,
        })
        frame.to_csv(
            OUTPUT_DIR / f"{path_stem}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if SAVE_NETCDF:
        path = OUTPUT_DIR / f"{path_stem}.nc"
        with Dataset(str(path), "w", format="NETCDF4") as nc:
            nc.createDimension("point", len(points))

            lon_var = nc.createVariable("lon", "f8", ("point",))
            lat_var = nc.createVariable("lat", "f8", ("point",))
            lon_var[:] = points[:, 0]
            lat_var[:] = points[:, 1]
            lon_var.units = "degrees_east"
            lat_var.units = "degrees_north"

            for name, values in fields.items():
                values = np.asarray(values)

                if np.issubdtype(values.dtype, np.integer):
                    variable = nc.createVariable(
                        name, "i4", ("point",), zlib=True
                    )
                    variable[:] = values
                else:
                    variable = nc.createVariable(
                        name, "f8", ("point",),
                        zlib=True, fill_value=np.nan
                    )
                    variable[:] = values

                variable.units = units.get(name, "1")

            nc.start_date_cst = START_DATE.isoformat()
            nc.end_date_cst_inclusive = END_DATE.isoformat()
            nc.wind_filename_timezone = "UTC"
            nc.pm25_filename_timezone = "CST UTC+8"
            nc.pblh_time_resolution = "daily"
            nc.wind_weighting = (
                "hourly cumulative-layer-thickness weighting within daily PBLH"
            )


# ============================================================
# 6. 主程序
# ============================================================

def main():
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = list(iter_dates())
    expected_days = {
        season: sum(season_of(day) == season for day in dates)
        for season in ("spring", "winter")
    }
    expected_hours = {
        season: expected_days[season] * 24
        for season in expected_days
    }

    # 分别保留原生网格。
    pm_points = None
    wind_points = None
    pbl_points = None

    pm_acc = {}
    wind_acc = {}
    pbl_daily_acc = {}
    pbl_matched_hour_acc = {}
    pbl_on_wind_acc = {}

    file_log = []
    wind_quality_log = []

    for day in dates:
        season = season_of(day)
        day_text = day.strftime("%Y%m%d")
        pbl_path = PBL_DIR / PBL_FILE_TEMPLATE.format(day_text)

        print(f"\n[{season}] {day_text}")

        if not pbl_path.exists():
            file_log.append({
                "time_cst": f"{day} 00:00:00",
                "status": "missing_daily_pblh",
                "detail": str(pbl_path),
            })
            continue

        # ---------- 当日PBLH ----------
        try:
            frame = pd.read_csv(pbl_path)
            required = {"lon", "lat", "pbl_heightchem"}

            if not required.issubset(frame.columns):
                raise ValueError(
                    f"PBLH缺少列：{required - set(frame.columns)}"
                )

            frame = frame.sort_values(["lat", "lon"]).reset_index(drop=True)
            current_pbl_points = frame[["lon", "lat"]].to_numpy(
                dtype=np.float64
            )
            pbl_values = (
                frame["pbl_heightchem"].to_numpy(dtype=np.float64)
                * PBLH_SCALE
            )

            if frame.duplicated(["lon", "lat"]).any():
                raise ValueError("PBLH文件含重复经纬度。")

            if pbl_points is None:
                pbl_points = current_pbl_points
                for s in ("spring", "winter"):
                    pbl_daily_acc[s] = MeanAccumulator(
                        ["pblh_m"], len(pbl_points)
                    )
                    pbl_matched_hour_acc[s] = MeanAccumulator(
                        ["pblh_m"], len(pbl_points)
                    )
            else:
                check_points(
                    pbl_points, current_pbl_points, "PBLH"
                )

            # 日平均场使用所有成功读取的季节日PBLH。
            pbl_daily_acc[season].add({"pblh_m": pbl_values})

            source_valid = (
                np.isfinite(current_pbl_points).all(axis=1)
                & np.isfinite(pbl_values)
            )
            pbl_source_points = current_pbl_points[source_valid]
            pbl_source_values = pbl_values[source_valid]

            if len(pbl_source_values) < 3:
                raise ValueError("PBLH有效插值点不足。")

        except Exception as error:
            file_log.append({
                "time_cst": f"{day} 00:00:00",
                "status": "pblh_read_error",
                "detail": str(error),
            })
            print("  PBLH读取失败：", error)
            continue

        pbl_on_wind = None
        pbl_outside_count = 0
        successful_common_hours = 0

        for hour in range(24):
            cst_datetime = dt.datetime.combine(
                day, dt.time(hour=hour)
            )
            pm_path, wind_path = hourly_paths(cst_datetime)

            pm_exists = pm_path.exists()
            wind_exists = wind_path.exists()

            if USE_COMMON_HOURS and not (pm_exists and wind_exists):
                file_log.append({
                    "time_cst": str(cst_datetime),
                    "status": "missing_hourly_file",
                    "detail": (
                        f"PM_exists={pm_exists};"
                        f"wind_exists={wind_exists}"
                    ),
                })
                continue

            pm_values = None
            weighted_u = None
            weighted_v = None
            coverage = None
            fallback = None

            # ---------- 读取本小时PM2.5 ----------
            if pm_exists:
                try:
                    need_coords = (
                        pm_points is None or CHECK_COORDINATES_EVERY_FILE
                    )
                    pm_values, current_points = read_pm25(
                        pm_path, need_coordinates=need_coords
                    )

                    if pm_points is None:
                        pm_points = current_points
                        for s in ("spring", "winter"):
                            pm_acc[s] = MeanAccumulator(
                                ["pm25_ug_m3"], len(pm_points)
                            )
                    elif need_coords:
                        check_points(pm_points, current_points, "PM2.5")

                    if len(pm_values) != len(pm_points):
                        raise ValueError("PM2.5网格点数变化。")

                    pm_values[~np.isfinite(pm_values)] = np.nan

                except Exception as error:
                    pm_values = None
                    file_log.append({
                        "time_cst": str(cst_datetime),
                        "status": "pm_read_error",
                        "detail": str(error),
                    })

            # ---------- 读取风场并加权 ----------
            if wind_exists:
                try:
                    need_coords = (
                        wind_points is None or CHECK_COORDINATES_EVERY_FILE
                    )
                    u, v, thickness, current_points = read_wind(
                        wind_path, need_coordinates=need_coords
                    )

                    if wind_points is None:
                        wind_points = current_points
                        for s in ("spring", "winter"):
                            wind_acc[s] = MeanAccumulator(
                                ["u_m_s", "v_m_s", "speed_m_s"],
                                len(wind_points),
                            )
                            pbl_on_wind_acc[s] = MeanAccumulator(
                                ["pblh_m"], len(wind_points)
                            )
                    elif need_coords:
                        check_points(
                            wind_points, current_points, "风场"
                        )

                    if u.shape[1] != len(wind_points):
                        raise ValueError("风场网格点数变化。")

                    if pbl_on_wind is None:
                        # 与原代码保持一致：线性插值，域外填500m。
                        # 先获得NaN版本，记录域外点数量。
                        pbl_linear = griddata(
                            pbl_source_points,
                            pbl_source_values,
                            wind_points,
                            method="linear",
                            fill_value=np.nan,
                        )
                        outside = ~np.isfinite(pbl_linear)
                        pbl_outside_count = int(outside.sum())

                        pbl_on_wind = pbl_linear.copy()
                        pbl_on_wind[outside] = PBL_OUTSIDE_FILL_VALUE

                    (
                        weighted_u,
                        weighted_v,
                        coverage,
                        fallback,
                    ) = compute_weighted_wind(
                        u, v, thickness, pbl_on_wind,
                        FALLBACK_TO_LOWEST_WIND,
                        REQUIRE_FULL_PBL_COVERAGE,
                        MIN_PBL_COVERAGE,
                    )

                except Exception as error:
                    weighted_u = None
                    weighted_v = None
                    file_log.append({
                        "time_cst": str(cst_datetime),
                        "status": "wind_read_or_weighting_error",
                        "detail": str(error),
                    })

            pm_success = pm_values is not None
            wind_success = (
                weighted_u is not None and weighted_v is not None
            )

            if USE_COMMON_HOURS and not (pm_success and wind_success):
                continue

            # 所有读取和计算结束后再累加，避免半成功小时混入共同样本。
            if pm_success:
                pm_acc[season].add({"pm25_ug_m3": pm_values})

            if wind_success:
                wind_acc[season].add({
                    "u_m_s": weighted_u,
                    "v_m_s": weighted_v,
                    "speed_m_s": np.hypot(weighted_u, weighted_v),
                })

                valid_wind = (
                    np.isfinite(weighted_u) & np.isfinite(weighted_v)
                )
                pbl_for_valid_wind = np.where(
                    valid_wind, pbl_on_wind, np.nan
                )
                pbl_on_wind_acc[season].add({
                    "pblh_m": pbl_for_valid_wind
                })

                finite_coverage = coverage[np.isfinite(coverage)]
                wind_quality_log.append({
                    "time_cst": str(cst_datetime),
                    "n_wind_points": len(wind_points),
                    "n_valid_weighted_wind": int(valid_wind.sum()),
                    "n_lowest_layer_fallback": int(fallback.sum()),
                    "n_pblh_outside_interpolation": pbl_outside_count,
                    "n_coverage_below_threshold": int(
                        np.sum(
                            np.isfinite(coverage)
                            & (coverage < MIN_PBL_COVERAGE)
                        )
                    ),
                    "mean_vertical_coverage": (
                        float(finite_coverage.mean())
                        if finite_coverage.size else np.nan
                    ),
                })

            if pm_success and wind_success:
                successful_common_hours += 1

            file_log.append({
                "time_cst": str(cst_datetime),
                "status": "used",
                "detail": (
                    f"PM_used={pm_success};"
                    f"wind_used={wind_success}"
                ),
            })

        # 原生PBLH按共同成功小时加权，便于与小时输入口径比较。
        if successful_common_hours > 0:
            pbl_matched_hour_acc[season].add(
                {"pblh_m": pbl_values},
                weight=successful_common_hours,
            )

        print(f"  共同成功小时：{successful_common_hours}/24")

    # ========================================================
    # 7. 保存季节平均场
    # ========================================================

    for season in ("spring", "winter"):
        if season in pm_acc:
            mean = pm_acc[season].means()
            count = pm_acc[season].count

            write_fields(
                f"{season}_pm25_mean",
                pm_points,
                {
                    "pm25_mean_ug_m3": mean["pm25_ug_m3"],
                    "n_valid_hours": count,
                    "hour_coverage": count / max(
                        expected_hours[season], 1
                    ),
                },
                {"pm25_mean_ug_m3": "ug m-3"},
            )

        if season in wind_acc:
            mean = wind_acc[season].means()
            count = wind_acc[season].count

            vector_speed = np.hypot(
                mean["u_m_s"], mean["v_m_s"]
            )
            vector_direction = (
                np.degrees(
                    np.arctan2(-mean["u_m_s"], -mean["v_m_s"])
                ) + 360.0
            ) % 360.0
            vector_direction[vector_speed < 1e-8] = np.nan

            write_fields(
                f"{season}_pbl_weighted_wind_mean",
                wind_points,
                {
                    "u_mean_m_s": mean["u_m_s"],
                    "v_mean_m_s": mean["v_m_s"],
                    "mean_hourly_speed_m_s": mean["speed_m_s"],
                    "vector_mean_speed_m_s": vector_speed,
                    "vector_mean_direction_deg": vector_direction,
                    "n_valid_hours": count,
                    "hour_coverage": count / max(
                        expected_hours[season], 1
                    ),
                },
                {
                    "u_mean_m_s": "m s-1",
                    "v_mean_m_s": "m s-1",
                    "mean_hourly_speed_m_s": "m s-1",
                    "vector_mean_speed_m_s": "m s-1",
                    "vector_mean_direction_deg": "degree",
                },
            )

        if season in pbl_daily_acc:
            daily_mean = pbl_daily_acc[season].means()["pblh_m"]
            matched_mean = (
                pbl_matched_hour_acc[season].means()["pblh_m"]
            )

            write_fields(
                f"{season}_pblh_mean_native_grid",
                pbl_points,
                {
                    "pblh_daily_mean_m": daily_mean,
                    "n_valid_days": pbl_daily_acc[season].count,
                    "day_coverage": (
                        pbl_daily_acc[season].count
                        / max(expected_days[season], 1)
                    ),
                    "pblh_common_hour_weighted_mean_m": matched_mean,
                    "n_common_hours": (
                        pbl_matched_hour_acc[season].count
                    ),
                },
                {
                    "pblh_daily_mean_m": "m",
                    "pblh_common_hour_weighted_mean_m": "m",
                },
            )

        if season in pbl_on_wind_acc:
            mean = pbl_on_wind_acc[season].means()["pblh_m"]
            write_fields(
                f"{season}_pblh_mean_on_wind_grid",
                wind_points,
                {
                    "pblh_mean_m": mean,
                    "n_valid_wind_hours": pbl_on_wind_acc[season].count,
                },
                {"pblh_mean_m": "m"},
            )

    pd.DataFrame(file_log).to_csv(
        OUTPUT_DIR / "file_processing_log.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(wind_quality_log).to_csv(
        OUTPUT_DIR / "weighted_wind_quality_log.csv",
        index=False, encoding="utf-8-sig",
    )

    metadata = {
        "start_date_cst": START_DATE.isoformat(),
        "end_date_cst_inclusive": END_DATE.isoformat(),
        "expected_spring_days": expected_days["spring"],
        "expected_winter_days": expected_days["winter"],
        "expected_spring_hours": expected_hours["spring"],
        "expected_winter_hours": expected_hours["winter"],
        "pm25_filename_timezone": "CST UTC+8",
        "wind_filename_timezone": "UTC",
        "wind_height_variable_meaning": "layer thickness in metres",
        "pblh_resolution": "daily",
        "pblh_outside_fill_value": (
            None if np.isnan(PBL_OUTSIDE_FILL_VALUE)
            else PBL_OUTSIDE_FILL_VALUE
        ),
        "fallback_to_lowest_wind": FALLBACK_TO_LOWEST_WIND,
        "require_full_pbl_coverage": REQUIRE_FULL_PBL_COVERAGE,
        "use_common_hours": USE_COMMON_HOURS,
        "notes": [
            "每小时先按当日PBLH计算层厚加权风，再进行季节平均。",
            "后续平均风场替换使用u_mean_m_s与v_mean_m_s。",
            "mean_hourly_speed_m_s是小时加权风速模长的平均。",
            "vector_mean_speed_m_s是平均风矢量的模长。",
            "两种平均风速不是同一指标。",
            "PBLH原生网格同时输出日均与共同小时加权均值。",
            "hour_coverage为逐格点有效小时数占应有季节小时数的比例。",
            "各变量分别保留原生网格，后续通量试验需使用一致的空间采样。",
        ],
    }

    with open(
        OUTPUT_DIR / "calculation_metadata.json",
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            metadata, file, ensure_ascii=False,
            indent=2, allow_nan=False
        )

    print("\n计算完成。")
    print("输出目录：", OUTPUT_DIR)
    print(f"耗时：{(time.time() - start) / 60:.2f}分钟")
    print(
        "后续风场替换使用："
        "u_mean_m_s、v_mean_m_s，"
        "不要仅使用mean_hourly_speed_m_s。"
    )


if __name__ == "__main__":
    main()