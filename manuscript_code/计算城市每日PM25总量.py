#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算汾渭平原各城市边界层内每天的 PM2.5 总质量。

定义（逐网格计算后在城市内部求和）：
    M_day = sum(C_day * H_PBL * A_grid)

其中：
    C_day  : 当地时间一天内小时 PM2.5 浓度的算术平均值，单位 μg/m3
    H_PBL  : 当日边界层高度，单位 m
    A_grid : 经纬度网格的实际面积，单位 m2
    M_day  : 每日平均状态下边界层内的 PM2.5 总质量，单位 μg

换算：1 μg = 1e-9 kg = 1e-12 t。

注意：该结果是污染物“存量/负荷”，不是排放量，也不是传输通量；因此不把
24 个小时的质量直接相加。若小时文件采用 UTC，请在调用时设置 --utc-offset 8，
程序会用北京时间（UTC+8）划分自然日；若文件时间本身已经是北京时间，保持默认 0。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from netCDF4 import Dataset


DEFAULT_PM25_DIR = Path(r"E:\pm25_1km_24h_new")
DEFAULT_PBL_DIR = Path(r"C:\Users\zyd\PycharmProjects\pbl_height")
DEFAULT_BOUNDARY_DIR = Path(r"C:\Users\zyd\PycharmProjects\boundary_grid_province")
DEFAULT_OUTPUT_XLSX = Path(
    r"C:\Users\zyd\PycharmProjects\城市每日PM25总量_20230301_20240229.xlsx"
)

CITY_NAMES = {
    "xian": "西安",
    "yuncheng": "运城",
    "lishi": "吕梁",
    "baoji": "宝鸡",
    "xianyang": "咸阳",
    "tongchuan": "铜川",
    "linfen": "临汾",
    "luoyang": "洛阳",
    "sanmenxia": "三门峡",
    "weinan": "渭南",
    "yuci": "晋中",
}


@dataclass(frozen=True)
class GridSubset:
    lat: np.ndarray
    lon: np.ndarray
    lat_slice: slice
    lon_slice: slice
    area_m2: np.ndarray
    city_masks: Dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算汾渭平原各城市边界层内每天的 PM2.5 总质量"
    )
    parser.add_argument("--start", default="2023-03-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2024-02-29", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--pm25-dir", type=Path, default=DEFAULT_PM25_DIR)
    parser.add_argument("--pbl-dir", type=Path, default=DEFAULT_PBL_DIR)
    parser.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="逐日明细 CSV；默认与 Excel 同名",
    )
    parser.add_argument(
        "--min-valid-hours",
        type=int,
        default=18,
        help="一天至少需要的有效小时文件数，默认 18",
    )
    parser.add_argument(
        "--min-grid-coverage",
        type=float,
        default=0.90,
        help="城市内有效网格比例阈值，默认 0.90",
    )
    parser.add_argument(
        "--utc-offset",
        type=int,
        default=0,
        help="文件时间到当地时间的小时偏移；UTC 文件在中国设为 8，已是北京时间设为 0",
    )
    return parser.parse_args()


def date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def read_boundary(path: Path) -> np.ndarray:
    """读取城市边界，返回按 [lon, lat] 排列的坐标。"""
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] < 2:
        raise ValueError(f"边界文件至少应有两列（lat、lon）：{path}")
    lat = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float)
    valid = np.isfinite(lat) & np.isfinite(lon)
    polygon = np.column_stack([lon[valid], lat[valid]])
    if polygon.shape[0] < 3:
        raise ValueError(f"边界有效点少于 3 个：{path}")
    return polygon


def load_boundaries(boundary_dir: Path) -> Dict[str, np.ndarray]:
    boundaries: Dict[str, np.ndarray] = {}
    for city in CITY_NAMES:
        path = boundary_dir / f"{city}_boundary.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少城市边界文件：{path}")
        boundaries[city] = read_boundary(path)
    return boundaries


def find_first_pm25_file(pm25_dir: Path, start: date, end: date, utc_offset: int) -> Path:
    """找到一个样例文件，用于读取固定网格坐标。"""
    for local_day in date_range(start, end):
        for local_hour in range(24):
            local_dt = datetime.combine(local_day, datetime.min.time()) + timedelta(hours=local_hour)
            file_dt = local_dt - timedelta(hours=utc_offset)
            path = pm25_dir / f"{file_dt:%Y%m%d%H}.nc"
            if path.exists():
                return path
    raise FileNotFoundError(f"日期范围内未找到 PM2.5 NetCDF 文件：{pm25_dir}")


def cell_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """计算规则经纬度网格面积，返回 shape=(lat, lon) 的 m2 数组。"""
    if lat.size < 2 or lon.size < 2:
        raise ValueError("经纬度坐标至少需要两个点")
    dlat = float(np.median(np.abs(np.diff(lat))))
    dlon = float(np.median(np.abs(np.diff(lon))))
    radius = 6_371_000.0
    lat_low = np.deg2rad(lat - dlat / 2.0)
    lat_high = np.deg2rad(lat + dlat / 2.0)
    dlon_rad = math.radians(dlon)
    row_area = radius**2 * dlon_rad * np.abs(np.sin(lat_high) - np.sin(lat_low))
    return np.broadcast_to(row_area[:, None], (lat.size, lon.size)).copy()


def prepare_grid(
    sample_file: Path, boundaries: Dict[str, np.ndarray]
) -> GridSubset:
    with Dataset(sample_file) as ds:
        lat_all = np.asarray(ds.variables["lat"][:], dtype=float)
        lon_all = np.asarray(ds.variables["lon"][:], dtype=float)

    all_points = np.vstack(list(boundaries.values()))
    min_lon, max_lon = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    min_lat, max_lat = float(all_points[:, 1].min()), float(all_points[:, 1].max())

    lat_idx = np.flatnonzero((lat_all >= min_lat) & (lat_all <= max_lat))
    lon_idx = np.flatnonzero((lon_all >= min_lon) & (lon_all <= max_lon))
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("城市边界与 PM2.5 网格没有空间重叠")

    # 各向外扩一个网格，避免边界附近点因浮点误差被裁掉。
    i0, i1 = max(0, int(lat_idx[0]) - 1), min(lat_all.size, int(lat_idx[-1]) + 2)
    j0, j1 = max(0, int(lon_idx[0]) - 1), min(lon_all.size, int(lon_idx[-1]) + 2)
    lat_slice, lon_slice = slice(i0, i1), slice(j0, j1)
    lat, lon = lat_all[lat_slice], lon_all[lon_slice]

    lon2d, lat2d = np.meshgrid(lon, lat)
    grid_points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    masks: Dict[str, np.ndarray] = {}
    for city, polygon in boundaries.items():
        masks[city] = MplPath(polygon, closed=True).contains_points(
            grid_points, radius=1e-12
        ).reshape(lat.size, lon.size)
        if not masks[city].any():
            raise ValueError(f"城市 {CITY_NAMES[city]} 的边界内没有 PM2.5 网格中心点")

    return GridSubset(
        lat=lat,
        lon=lon,
        lat_slice=lat_slice,
        lon_slice=lon_slice,
        area_m2=cell_areas(lat, lon),
        city_masks=masks,
    )


def hourly_paths(pm25_dir: Path, local_day: date, utc_offset: int) -> List[Path]:
    paths: List[Path] = []
    for local_hour in range(24):
        local_dt = datetime.combine(local_day, datetime.min.time()) + timedelta(hours=local_hour)
        file_dt = local_dt - timedelta(hours=utc_offset)
        path = pm25_dir / f"{file_dt:%Y%m%d%H}.nc"
        if path.exists():
            paths.append(path)
    return paths


def read_daily_pm25(paths: Sequence[Path], grid: GridSubset) -> Tuple[np.ndarray, np.ndarray]:
    """逐网格计算小时浓度的日平均，同时返回各网格有效小时数。"""
    shape = (grid.lat.size, grid.lon.size)
    value_sum = np.zeros(shape, dtype=np.float64)
    value_count = np.zeros(shape, dtype=np.int16)

    for path in paths:
        with Dataset(path) as ds:
            raw = ds.variables["pm25"][grid.lat_slice, grid.lon_slice]
            values = np.ma.filled(raw, np.nan).astype(np.float64, copy=False)
        valid = np.isfinite(values) & (values >= 0.0)
        value_sum[valid] += values[valid]
        value_count[valid] += 1

    daily_mean = np.full(shape, np.nan, dtype=np.float64)
    np.divide(value_sum, value_count, out=daily_mean, where=value_count > 0)
    return daily_mean, value_count


def read_pbl_grid(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = {"lat", "lon", "pbl_heightchem"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"PBL 文件缺少列 {sorted(missing)}：{path}")

    table = df.pivot_table(
        index="lat", columns="lon", values="pbl_heightchem", aggfunc="mean"
    ).sort_index().sort_index(axis=1)
    lat = table.index.to_numpy(float)
    lon = table.columns.to_numpy(float)
    values = table.to_numpy(float)
    if lat.size < 2 or lon.size < 2:
        raise ValueError(f"PBL 网格坐标不足：{path}")
    return lat, lon, values


def bilinear_interpolate(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    src: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """把规则 PBL 网格双线性插值到 PM2.5 网格中心。"""
    if src_lat[0] > src_lat[-1]:
        src_lat, src = src_lat[::-1], src[::-1, :]
    if src_lon[0] > src_lon[-1]:
        src_lon, src = src_lon[::-1], src[:, ::-1]

    iy = np.searchsorted(src_lat, target_lat, side="right") - 1
    ix = np.searchsorted(src_lon, target_lon, side="right") - 1
    outside_y = (target_lat < src_lat[0]) | (target_lat > src_lat[-1])
    outside_x = (target_lon < src_lon[0]) | (target_lon > src_lon[-1])
    iy = np.clip(iy, 0, src_lat.size - 2)
    ix = np.clip(ix, 0, src_lon.size - 2)

    wy = (target_lat - src_lat[iy]) / (src_lat[iy + 1] - src_lat[iy])
    wx = (target_lon - src_lon[ix]) / (src_lon[ix + 1] - src_lon[ix])

    v00 = src[np.ix_(iy, ix)]
    v10 = src[np.ix_(iy + 1, ix)]
    v01 = src[np.ix_(iy, ix + 1)]
    v11 = src[np.ix_(iy + 1, ix + 1)]
    result = (
        v00 * (1.0 - wy[:, None]) * (1.0 - wx[None, :])
        + v10 * wy[:, None] * (1.0 - wx[None, :])
        + v01 * (1.0 - wy[:, None]) * wx[None, :]
        + v11 * wy[:, None] * wx[None, :]
    )
    result[outside_y, :] = np.nan
    result[:, outside_x] = np.nan
    return result


def empty_rows(local_day: date, available_hours: int, flag: str) -> List[dict]:
    return [
        {
            "日期": local_day.isoformat(),
            "城市": CITY_NAMES[city],
            "城市英文标识": city,
            "PM2.5总质量_t": np.nan,
            "PM2.5总质量_kg": np.nan,
            "面积加权日均浓度_ug_m3": np.nan,
            "面积加权PBLH_m": np.nan,
            "边界层空气体积_m3": np.nan,
            "小时文件数": available_hours,
            "城市总网格数": np.nan,
            "有效网格数": 0,
            "有效网格比例": 0.0,
            "质量控制": flag,
        }
        for city in CITY_NAMES
    ]


def calculate_day(
    local_day: date,
    pm25_dir: Path,
    pbl_dir: Path,
    grid: GridSubset,
    min_valid_hours: int,
    min_grid_coverage: float,
    utc_offset: int,
) -> List[dict]:
    paths = hourly_paths(pm25_dir, local_day, utc_offset)
    if len(paths) < min_valid_hours:
        return empty_rows(local_day, len(paths), "insufficient_hour_files")

    pbl_path = pbl_dir / f"pbl_height_{local_day:%Y%m%d}.csv"
    if not pbl_path.exists():
        return empty_rows(local_day, len(paths), "missing_pbl_file")

    daily_pm25, valid_hour_count = read_daily_pm25(paths, grid)
    pbl_lat, pbl_lon, pbl_values = read_pbl_grid(pbl_path)
    pbl = bilinear_interpolate(pbl_lat, pbl_lon, pbl_values, grid.lat, grid.lon)

    rows: List[dict] = []
    for city, city_mask in grid.city_masks.items():
        total_cells = int(city_mask.sum())
        # 一个网格只有在其小时浓度数达到阈值且 PBLH 有效时才参与质量积分。
        valid = (
            city_mask
            & (valid_hour_count >= min_valid_hours)
            & np.isfinite(daily_pm25)
            & np.isfinite(pbl)
            & (pbl > 0.0)
        )
        valid_cells = int(valid.sum())
        coverage = valid_cells / total_cells if total_cells else 0.0

        if valid_cells == 0:
            mass_t = mass_kg = mean_pm25 = mean_pbl = air_volume = np.nan
            flag = "no_valid_grid"
        else:
            area = grid.area_m2[valid]
            concentration = daily_pm25[valid]
            height = pbl[valid]
            mass_ug = float(np.sum(concentration * height * area))
            mass_kg = mass_ug * 1e-9
            mass_t = mass_ug * 1e-12
            mean_pm25 = float(np.sum(concentration * area) / np.sum(area))
            mean_pbl = float(np.sum(height * area) / np.sum(area))
            air_volume = float(np.sum(height * area))
            flag = "ok" if coverage >= min_grid_coverage else "low_grid_coverage"

        rows.append(
            {
                "日期": local_day.isoformat(),
                "城市": CITY_NAMES[city],
                "城市英文标识": city,
                "PM2.5总质量_t": mass_t,
                "PM2.5总质量_kg": mass_kg,
                "面积加权日均浓度_ug_m3": mean_pm25,
                "面积加权PBLH_m": mean_pbl,
                "边界层空气体积_m3": air_volume,
                "小时文件数": len(paths),
                "城市总网格数": total_cells,
                "有效网格数": valid_cells,
                "有效网格比例": coverage,
                "质量控制": flag,
            }
        )
    return rows


def build_summary(daily: pd.DataFrame) -> pd.DataFrame:
    valid = daily[daily["PM2.5总质量_t"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "城市",
                "有效天数",
                "平均每日总质量_t",
                "中位数_t",
                "最小值_t",
                "最大值_t",
            ]
        )
    summary = (
        valid.groupby("城市", sort=False)["PM2.5总质量_t"]
        .agg(
            有效天数="count",
            平均每日总质量_t="mean",
            中位数_t="median",
            最小值_t="min",
            最大值_t="max",
        )
        .reset_index()
    )
    return summary


def write_outputs(daily: pd.DataFrame, output_xlsx: Path, output_csv: Path) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_csv, index=False, encoding="utf-8-sig", float_format="%.10g")

    explanation = pd.DataFrame(
        {
            "项目": [
                "计算对象",
                "计算公式",
                "日浓度",
                "网格面积",
                "质量换算",
                "时间说明",
                "缺测处理",
                "科学含义",
            ],
            "说明": [
                "各城市行政边界内、边界层高度以下的每日平均 PM2.5 总质量",
                "M = Σ(C_day × H_PBL × A_grid)",
                "每个 PM2.5 网格当天有效小时浓度的算术平均，单位 μg/m3",
                "根据规则经纬度网格边界和地球半径计算，单位 m2",
                "1 μg = 1e-9 kg = 1e-12 t",
                "--utc-offset=0 表示文件名时间已是当地时间；UTC 文件在中国应设为 8",
                "有效小时或有效网格不足时保留日期和城市，并在质量控制列标记；不对缺失面积外推",
                "这是污染物质量存量/负荷，不是排放量、日累计量或跨边界传输通量",
            ],
        }
    )
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        daily.to_excel(writer, sheet_name="每日城市总量", index=False)
        build_summary(daily).to_excel(writer, sheet_name="城市汇总", index=False)
        explanation.to_excel(writer, sheet_name="说明", index=False)


def main() -> None:
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if start > end:
        raise ValueError("开始日期不能晚于结束日期")
    if not 1 <= args.min_valid_hours <= 24:
        raise ValueError("--min-valid-hours 必须在 1 到 24 之间")
    if not 0.0 <= args.min_grid_coverage <= 1.0:
        raise ValueError("--min-grid-coverage 必须在 0 到 1 之间")

    output_csv = args.output_csv or args.output_xlsx.with_suffix(".csv")
    boundaries = load_boundaries(args.boundary_dir)
    sample = find_first_pm25_file(args.pm25_dir, start, end, args.utc_offset)
    grid = prepare_grid(sample, boundaries)

    all_rows: List[dict] = []
    dates = list(date_range(start, end))
    for number, local_day in enumerate(dates, start=1):
        try:
            rows = calculate_day(
                local_day=local_day,
                pm25_dir=args.pm25_dir,
                pbl_dir=args.pbl_dir,
                grid=grid,
                min_valid_hours=args.min_valid_hours,
                min_grid_coverage=args.min_grid_coverage,
                utc_offset=args.utc_offset,
            )
        except Exception as exc:
            rows = empty_rows(local_day, 0, f"calculation_error: {exc}")
        all_rows.extend(rows)
        print(f"[{number:03d}/{len(dates):03d}] {local_day}: 完成")

    daily = pd.DataFrame(all_rows)
    daily["日期"] = pd.to_datetime(daily["日期"])
    write_outputs(daily, args.output_xlsx, output_csv)
    print(f"逐日明细：{output_csv}")
    print(f"Excel 结果：{args.output_xlsx}")


if __name__ == "__main__":
    main()
