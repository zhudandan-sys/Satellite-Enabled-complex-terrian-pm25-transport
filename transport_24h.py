import numpy as np
import os
import datetime
import time
import glob
import pandas as pd
from math import *
from netCDF4 import Dataset
from scipy.spatial import Delaunay, KDTree
from scipy.interpolate import griddata
import warnings
from numba import jit
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ================== 可配置路径 ==================
WIND_DIR = r"E:\wind_24h"
PM25_DIR = r"E:\pm25_1km_24h_new"
PBL_DIR = r"C:\Users\zyd\PycharmProjects\pbl_height"
BOUNDARY_DIR = r"C:\Users\zyd\PycharmProjects\province_to_province_boundary_grid"

WIND_FILE_TEMPLATE = "{}{}.nc"
PM25_FILE_TEMPLATE = "{}{}.nc"
PBL_FILE_TEMPLATE = "pbl_height_{}.csv"

OUTPUT_DIR = "transport_flux24h_open"

START_DATE = datetime.date(2023, 3, 1)
END_DATE = datetime.date(2024, 2, 29)

LON_MIN = 105.0
LON_MAX = 115.0
LAT_MIN = 31.0
LAT_MAX = 41.0
RES = 0.01


# ------------------ 加权风计算（累积高度法，Numba加速） ------------------
@jit(nopython=True)
def compute_weighted_wind(u_flat, v_flat, thick_flat, pbl_flat):
    """
    计算PBL内的加权风（权重为各层在PBL内的厚度）
    u_flat, v_flat, thick_flat: shape (nz, N)
    pbl_flat: shape (N,)
    返回 u_weighted, v_weighted (shape (N,))
    """
    nz, N = u_flat.shape
    u_weighted = np.zeros(N, dtype=np.float64)
    v_weighted = np.zeros(N, dtype=np.float64)

    for i in range(N):
        pbl = pbl_flat[i]
        if np.isnan(pbl) or pbl <= 0:
            u_weighted[i] = u_flat[0, i]
            v_weighted[i] = v_flat[0, i]
            continue

        cum_h = 0.0
        total_thick = 0.0
        u_sum = 0.0
        v_sum = 0.0

        for k in range(nz):
            thick = thick_flat[k, i]
            top_h = cum_h + thick
            if top_h <= pbl:
                total_thick += thick
                u_sum += u_flat[k, i] * thick
                v_sum += v_flat[k, i] * thick
            else:
                if cum_h < pbl:
                    partial = pbl - cum_h
                    if partial > 0:
                        total_thick += partial
                        u_sum += u_flat[k, i] * partial
                        v_sum += v_flat[k, i] * partial
                break
            cum_h = top_h

        if total_thick > 0:
            u_weighted[i] = u_sum / total_thick
            v_weighted[i] = v_sum / total_thick
        else:
            u_weighted[i] = u_flat[0, i]
            v_weighted[i] = v_flat[0, i]

    return u_weighted, v_weighted


# ------------------ 预计算插值权重类 ------------------
class InterpWeights:
    """针对固定源点集和目标点集，预先计算插值权重（Delaunay + 重心坐标）"""

    def __init__(self, src_points, dst_points, fill_value=0.0):
        self.fill_value = fill_value
        self.weights = []
        src_points = np.asarray(src_points)
        dst_points = np.asarray(dst_points)
        tri = Delaunay(src_points)
        kdt = KDTree(src_points)
        for pt in dst_points:
            simplex = tri.find_simplex(pt)
            if simplex >= 0:
                transform = tri.transform[simplex]
                delta = pt - transform[2]
                bary = transform[1] @ delta
                bary = np.append(bary, 1.0 - bary.sum())
                bary = np.clip(bary, 0, 1)
                if bary.sum() > 0:
                    bary = bary / bary.sum()
                else:
                    bary = np.ones(3) / 3.0
                vertices = tri.simplices[simplex]
                self.weights.append((vertices.tolist(), bary.tolist()))
            else:
                dist, idx = kdt.query(pt, k=1)
                self.weights.append(([int(idx), int(idx), int(idx)], [1.0, 0.0, 0.0]))

    def interp(self, values):
        result = np.full(len(self.weights), self.fill_value, dtype=values.dtype)
        for i, (vert_indices, bary_weights) in enumerate(self.weights):
            if len(vert_indices) != len(bary_weights):
                min_len = min(len(vert_indices), len(bary_weights))
                vert_indices = vert_indices[:min_len]
                bary_weights = bary_weights[:min_len]
                bary_weights = np.array(bary_weights) / np.sum(bary_weights)
            val = np.sum(values[vert_indices] * bary_weights)
            result[i] = val
        return result


# ------------------ 处理单小时的函数（修改为开边界通量计算） ------------------
def process_hour(args):
    (date_str, hour, wind_file, pm25_file,
     all_boundary_info,
     pbl_on_wind_grid, pbl_boundary_dict,
     wind_interp_weights, pm25_interp_weights) = args

    r_earth = 6371004
    boundary_names = list(all_boundary_info.keys())
    hourly_trans = {name: 0.0 for name in boundary_names}

    try:
        # ----- 1. 读取风场数据 -----
        with Dataset(wind_file, 'r') as nc:
            u_raw = nc.variables['u'][:]
            v_raw = nc.variables['v'][:]
            height_raw = nc.variables['height'][:]  # 厚度

        u = np.ma.filled(u_raw, np.nan).astype(np.float32)
        v = np.ma.filled(v_raw, np.nan).astype(np.float32)
        thickness = np.ma.filled(height_raw, np.nan).astype(np.float32)

        if u.ndim == 3:
            nz, ny, nx = u.shape
            N = ny * nx
            u_flat = u.reshape(nz, N)
            v_flat = v.reshape(nz, N)
            thick_flat = thickness.reshape(nz, N)
        elif u.ndim == 2:
            nz, N = u.shape
            u_flat = u
            v_flat = v
            thick_flat = thickness
        else:
            raise ValueError(f"风场维度异常: {u.shape}")

        pbl_flat = pbl_on_wind_grid.ravel().astype(np.float32)

        u_weighted_flat, v_weighted_flat = compute_weighted_wind(
            u_flat, v_flat, thick_flat, pbl_flat
        )

        # ----- 2. 插值到边界点 -----
        u_all = wind_interp_weights.interp(u_weighted_flat)
        v_all = wind_interp_weights.interp(v_weighted_flat)

        u_boundary = {}
        v_boundary = {}
        offset = 0
        for name, pts in all_boundary_info.items():
            n = len(pts)
            u_boundary[name] = u_all[offset:offset + n]
            v_boundary[name] = v_all[offset:offset + n]
            offset += n

        # ----- 3. PM2.5插值 -----
        with Dataset(pm25_file, 'r') as nc:
            pm25_raw = nc.variables['pm25'][:]
        pm25 = np.ma.filled(pm25_raw, np.nan).astype(np.float32).ravel()
        pm25_all = pm25_interp_weights.interp(pm25)

        pm25_boundary = {}
        offset = 0
        for name, pts in all_boundary_info.items():
            n = len(pts)
            pm25_boundary[name] = pm25_all[offset:offset + n]
            offset += n

        # ----- 4. 计算通量（开边界：相邻点对，不闭合）-----
        for name in boundary_names:
            pts = all_boundary_info[name]
            u_vals = u_boundary[name]
            v_vals = v_boundary[name]
            pm_vals = pm25_boundary[name]
            pbl_vals = pbl_boundary_dict[name]

            # 剔除无效点（如含NaN）
            valid = ~(np.isnan(u_vals) | np.isnan(v_vals) | np.isnan(pm_vals) | np.isnan(pbl_vals))
            if np.sum(valid) < 2:
                continue

            # 提取有效点（保持原始顺序）
            pts_valid = pts[valid]
            u_valid = u_vals[valid]
            v_valid = v_vals[valid]
            pm_valid = pm_vals[valid]
            pbl_valid = pbl_vals[valid]

            n_pts = len(pts_valid)
            total_flux = 0.0

            # 对每段相邻边界点（i, i+1），不闭合
            for i in range(n_pts - 1):
                lon_cur, lat_cur = pts_valid[i]
                lon_nxt, lat_nxt = pts_valid[i + 1]

                dlon = lon_nxt - lon_cur
                dlat = lat_nxt - lat_cur
                avg_lat = (lat_nxt + lat_cur) / 2.0

                # 经纬度差转换为弧长（米）
                dx = 2 * r_earth * np.pi * dlon * np.cos(np.pi * avg_lat / 180.0) / 360.0
                dy = 2 * r_earth * np.pi * dlat / 360.0

                # 左侧法向量（与 transport.py 一致：n = (-dy, dx)）
                nx = -dy
                ny = dx

                # 两端点法向风速分量（已隐含线段长度）
                dot_cur = nx * u_valid[i] + ny * v_valid[i]
                dot_nxt = nx * u_valid[i + 1] + ny * v_valid[i + 1]

                # 梯形积分贡献（μg/s）
                seg_flux = (pbl_valid[i] * pm_valid[i] * dot_cur +
                            pbl_valid[i + 1] * pm_valid[i + 1] * dot_nxt) / 2.0
                total_flux += seg_flux

            hourly_trans[name] = total_flux

    except Exception as e:
        print(f"  小时处理异常 {date_str}{hour:02d}(CST): {e}")

    return hourly_trans


def main():
    start_time = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    boundary_path = os.path.join(script_dir, BOUNDARY_DIR)
    output_path = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)

    # ================== 1. 加载边界文件（不修正方向） ==================
    print("加载边界文件（开边界，不进行方向修正）...")
    city_boundary_files = glob.glob(os.path.join(boundary_path, '*.csv'))
    all_boundary_info = {}
    for boundary_file in city_boundary_files:
        boundary_name = os.path.splitext(os.path.basename(boundary_file))[0]
        df = pd.read_csv(boundary_file, index_col=0)
        # 假设 CSV 第一列为纬度，第二列为经度
        lat_vals = df.iloc[:, 0].values
        lon_vals = df.iloc[:, 1].values
        all_boundary_info[boundary_name] = np.column_stack([lon_vals, lat_vals])

    boundary_names = list(all_boundary_info.keys())
    if not boundary_names:
        print("未找到边界文件")
        return
    all_boundary_points = np.vstack(list(all_boundary_info.values()))

    # ================== 2. 提取源点坐标（风场和PM2.5网格点） ==================
    print("提取NC文件网格坐标...")
    # 风场
    test_date = START_DATE
    found = False
    while test_date <= END_DATE and not found:
        for cst_hour in range(24):
            cst_dt = datetime.datetime.combine(test_date, datetime.time(hour=cst_hour))
            utc_dt = cst_dt - datetime.timedelta(hours=8)
            utc_file = os.path.join(WIND_DIR,
                                    WIND_FILE_TEMPLATE.format(utc_dt.strftime("%Y%m%d"), f"{utc_dt.hour:02d}"))
            if os.path.exists(utc_file):
                with Dataset(utc_file, 'r') as nc:
                    lat = np.ma.filled(nc.variables['lat'][:], np.nan).ravel()
                    lon = np.ma.filled(nc.variables['lon'][:], np.nan).ravel()
                    wind_points = np.column_stack([lon, lat])
                found = True
                break
        test_date += datetime.timedelta(days=1)
    if not found:
        print("未找到风场文件")
        return

    # PM2.5
    test_date = START_DATE
    found = False
    while test_date <= END_DATE and not found:
        for cst_hour in range(24):
            cst_file = os.path.join(PM25_DIR,
                                    PM25_FILE_TEMPLATE.format(test_date.strftime("%Y%m%d"), f"{cst_hour:02d}"))
            if os.path.exists(cst_file):
                with Dataset(cst_file, 'r') as nc:
                    lat = np.ma.filled(nc.variables['lat'][:], np.nan).ravel()
                    lon = np.ma.filled(nc.variables['lon'][:], np.nan).ravel()
                    pm25_points = np.column_stack([lon, lat])
                found = True
                break
        test_date += datetime.timedelta(days=1)
    if not found:
        print("未找到PM2.5文件")
        return

    # ================== 3. 预计算插值权重 ==================
    print("预计算插值权重...")
    wind_interp_weights = InterpWeights(wind_points, all_boundary_points, fill_value=0.0)
    pm25_interp_weights = InterpWeights(pm25_points, all_boundary_points, fill_value=0.0)

    # ================== 4. 主循环 ==================
    current_date = START_DATE
    while current_date <= END_DATE:
        date_str = current_date.strftime("%Y%m%d")
        print(f"\n处理日期: {date_str}")

        # 读取PBL（每天一次）
        pbl_file = os.path.join(PBL_DIR, PBL_FILE_TEMPLATE.format(date_str))
        try:
            pbl_df = pd.read_csv(pbl_file)
            pbl_lon = pbl_df['lon'].values
            pbl_lat = pbl_df['lat'].values
            pbl_val = pbl_df['pbl_heightchem'].values
            valid = ~(np.isnan(pbl_lon) | np.isnan(pbl_lat) | np.isnan(pbl_val))
            pbl_src = np.column_stack([pbl_lon[valid], pbl_lat[valid]])
            pbl_src_val = pbl_val[valid]

            pbl_on_wind_grid = griddata(pbl_src, pbl_src_val, wind_points, method='linear', fill_value=500.0)
            pbl_boundary_dict = {}
            for name, pts in all_boundary_info.items():
                pbl_boundary_dict[name] = griddata(pbl_src, pbl_src_val, pts, method='linear', fill_value=500.0)
        except Exception as e:
            print(f"PBL处理失败: {e}")
            current_date += datetime.timedelta(days=1)
            continue

        # 准备小时任务
        hour_args = []
        for cst_hour in range(24):
            cst_dt = datetime.datetime.combine(current_date, datetime.time(hour=cst_hour))
            utc_dt = cst_dt - datetime.timedelta(hours=8)
            wind_file = os.path.join(WIND_DIR,
                                     WIND_FILE_TEMPLATE.format(utc_dt.strftime("%Y%m%d"), f"{utc_dt.hour:02d}"))
            pm25_file = os.path.join(PM25_DIR,
                                     PM25_FILE_TEMPLATE.format(current_date.strftime("%Y%m%d"), f"{cst_hour:02d}"))
            if not (os.path.exists(wind_file) and os.path.exists(pm25_file)):
                print(f"  跳过缺失: {date_str}{cst_hour:02d}")
                continue
            hour_args.append((date_str, cst_hour, wind_file, pm25_file,
                              all_boundary_info, pbl_on_wind_grid, pbl_boundary_dict,
                              wind_interp_weights, pm25_interp_weights))

        # 保存日结果（吨/天）
        # 注意：开边界通量的符号取决于法向量方向（左侧），正值/负值表示输送方向
        # 并行处理部分 —— 修正累加逻辑
        daily_trans = {name: 0.0 for name in boundary_names}
        max_workers = min(8, len(hour_args))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_hour, arg) for arg in hour_args]
            for future in as_completed(futures):
                hourly = future.result()  # 单位: μg/s
                for name in boundary_names:
                    # 每小时总质量(μg) = 每秒通量 × 3600 秒
                    daily_trans[name] += hourly.get(name, 0.0) * 3600

        # 保存结果 —— 修正转换因子
        result_df = pd.DataFrame({
            'boundary_name': boundary_names,
            'transport_ug_day': [daily_trans[name] for name in boundary_names],
            'transport_t_day': [daily_trans[name] * 1e-12 for name in boundary_names]  # 1 μg = 1e-12 t
        })
        out_file = os.path.join(output_path, f"open_boundary_flux_{date_str}.csv")
        result_df.to_csv(out_file, index=False)
        print(f"  已保存: {out_file}")

        current_date += datetime.timedelta(days=1)

    print(f"\n总耗时: {(time.time() - start_time) / 60:.2f} 分钟")
    print("处理完成")


if __name__ == '__main__':
    main()