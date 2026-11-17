import numpy as np
import os
import datetime
import time
import pandas as pd
from netCDF4 import Dataset
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, KDTree
import warnings
import gc
from numba import jit

warnings.filterwarnings('ignore')

# ================== 可配置路径（根据实际情况修改） ==================
WIND_DIR = r"E:\wind_24h"  # 存放多层风场NC文件的目录（UTC时间）
PM25_DIR = r"E:\pm25_1km_24h_new"  # 存放PM2.5 NC文件的目录（CST时间，已修改）
PBL_DIR = r"C:\Users\zyd\PycharmProjects\pbl_height"  # 存放PBL CSV文件的目录
BOUNDARY_DIR = r"C:\Users\zyd\PycharmProjects\boundary_grid_province"  # 边界文件夹

# 文件命名模板
WIND_FILE_TEMPLATE = "{}{}.nc"
PM25_FILE_TEMPLATE = "{}{}.nc"  # 使用CST时间命名
PBL_FILE_TEMPLATE = "pbl_height_{}.csv"

# 输出目录
OUTPUT_DIR = "city_net_flux24h"

# 时间范围（北京时间）
START_DATE = datetime.date(2023, 12, 27)
END_DATE = datetime.date(2024, 2, 29)

# 研究区域（用于过滤，非必须）
LON_MIN = 105.0
LON_MAX = 115.0
LAT_MIN = 31.0
LAT_MAX = 41.0
RES = 0.01

# 目标城市
target_cities = ['xian', 'yuncheng', 'lishi', 'baoji', 'xianyang',
                 'tongchuan', 'linfen', 'luoyang', 'sanmenxia', 'weinan', 'yuci']


# ------------------ 加权风计算（Numba加速） ------------------
@jit(nopython=True)
def compute_weighted_wind(u_flat, v_flat, h_flat, pbl_flat):
    """垂直积分：边界层内按PBL高度下的实际层厚度加权平均风
    核心修改：
    1. 针对每个网格点的PBL高度，计算其下各层的实际厚度
    2. 用PBL约束下的层厚度作为权重计算风速加权平均
    """
    nz, N = u_flat.shape
    u_weighted = np.zeros(N, dtype=np.float64)
    v_weighted = np.zeros(N, dtype=np.float64)

    for i in range(N):
        # 该网格点的所有层高度和PBL高度
        h_levels = h_flat[:, i]
        pbl_i = pbl_flat[i]

        # 筛选PBL高度以下的有效层
        mask = h_levels <= pbl_i
        valid_h = h_levels[mask]
        n_valid = len(valid_h)

        if n_valid >= 1:
            # 计算PBL高度下各层的实际厚度
            thick_valid = np.zeros(n_valid, dtype=np.float64)

            if n_valid == 1:
                # 仅1层时，厚度为PBL高度（该层覆盖整个PBL）
                thick_valid[0] = pbl_i
            else:
                # 多层时：
                # 第一层厚度 = 第二层高度 - 第一层高度
                thick_valid[0] = valid_h[1] - valid_h[0]
                # 中间层厚度 = (下下层高度 - 上上一层高度) / 2
                if n_valid > 2:
                    thick_valid[1:-1] = (valid_h[2:] - valid_h[:-2]) / 2.0
                # 最后一层厚度 = PBL高度 - 最后一层高度（修正到PBL顶）
                thick_valid[-1] = pbl_i - valid_h[-1]

            # 确保厚度非负（避免0或负数导致加权异常）
            thick_valid = np.clip(thick_valid, 1e-6, None)

            # 提取对应层的风速
            u_valid = u_flat[mask, i]
            v_valid = v_flat[mask, i]

            # 按厚度加权计算平均风速
            total_w = np.sum(thick_valid)
            if total_w > 0:
                u_weighted[i] = np.sum(u_valid * thick_valid) / total_w
                v_weighted[i] = np.sum(v_valid * thick_valid) / total_w
            else:
                # 兜底：使用第一层风速
                u_weighted[i] = u_flat[0, i]
                v_weighted[i] = v_flat[0, i]
        else:
            # 无有效层（PBL高度低于所有层），使用第一层风速
            u_weighted[i] = u_flat[0, i]
            v_weighted[i] = v_flat[0, i]

    return u_weighted, v_weighted


# ------------------ 边界积分计算净通量（修正版） ------------------
def calculate_net_flux(pts, u_vals, v_vals, pm_vals, pbl_vals, r_earth):
    """沿闭合多边形积分：Σ (PM2.5 * PBL * (u,v)·法向)   （注意：法向量已隐含边长，不再乘线段长度）"""
    valid_mask = ~(np.isnan(u_vals) | np.isnan(v_vals) | np.isnan(pm_vals) | np.isnan(pbl_vals))
    if np.sum(valid_mask) < 2:
        return 0.0

    pts_valid = pts[valid_mask]
    u_valid = u_vals[valid_mask]
    v_valid = v_vals[valid_mask]
    pm_valid = pm_vals[valid_mask]
    pbl_valid = pbl_vals[valid_mask]

    n = len(pts_valid)
    net_flux = 0.0
    for i in range(n):
        j = (i + 1) % n  # 闭合多边形，最后一点连回第一点
        lon_cur, lat_cur = pts_valid[i]
        lon_nxt, lat_nxt = pts_valid[j]

        dlon = lon_nxt - lon_cur
        dlat = lat_nxt - lat_cur
        avg_lat = (lat_nxt + lat_cur) / 2.0

        # 经纬度差转换为弧长（米）
        dx = 2 * r_earth * np.pi * dlon * np.cos(np.radians(avg_lat)) / 360.0
        dy = 2 * r_earth * np.pi * dlat / 360.0

        # 法向量（保持原符号，即假设多边形为顺时针时指向外部；若方向相反则净通量符号反转，但方向判断逻辑不变）
        nx = -dy
        ny = dx

        # 梯形积分：线段两端点的通量贡献平均
        dot_cur = nx * u_valid[i] + ny * v_valid[i]
        dot_nxt = nx * u_valid[j] + ny * v_valid[j]
        seg_flux = (pbl_valid[i] * pm_valid[i] * dot_cur + pbl_valid[j] * pm_valid[j] * dot_nxt) / 2.0
        net_flux += seg_flux

    return net_flux


# ------------------ 预计算风场插值权重（基于Delaunay三角剖分，使用有效点集） ------------------
def precompute_wind_weights(wind_points_valid, boundary_pts):
    """
    为每个边界点预计算风场插值所需的顶点索引和权重
    参数：
        wind_points_valid : (N_valid, 2) 有效风场网格点坐标（已剔除NaN）
        boundary_pts: (M, 2) 城市边界点坐标
    返回：
        weights_list: 列表，每个元素为 (indices, coeffs)
            - indices: 顶点索引数组（在 wind_points_valid 中的位置）
            - coeffs : 对应的权重（和为1）
    """
    print("  预计算风场插值权重（Delaunay + 备选KDTree）...")
    tri = Delaunay(wind_points_valid)  # 构建三角剖分
    tree = KDTree(wind_points_valid)  # 备选：最近邻搜索

    weights_list = []
    outliers = 0
    for pt in boundary_pts:
        simplex = tri.find_simplex(pt)  # 查找所在三角形索引
        if simplex >= 0:  # 点在凸包内
            vertices = tri.simplices[simplex]
            transform = tri.transform[simplex]
            c = transform[:2].dot(pt - transform[2])
            coeffs = np.array([c[0], c[1], 1 - c[0] - c[1]])
            if np.any(coeffs < 0):
                coeffs = np.clip(coeffs, 0, 1)
                coeffs /= coeffs.sum()
            weights_list.append((vertices, coeffs))
        else:  # 点在凸包外 → 使用最近邻
            outliers += 1
            dist, idx = tree.query(pt)
            weights_list.append(([idx], np.array([1.0])))
    if outliers > 0:
        print(f"    警告：{outliers} 个边界点位于风场网格凸包外，使用最近邻插值")
    return weights_list


# ------------------ 预计算PM2.5插值权重（双线性，规则网格） ------------------
def precompute_pm25_weights(lon_1d, lat_1d, boundary_pts):
    """
    为每个边界点预计算PM2.5双线性插值的网格索引和权重
    参数：
        lon_1d : 一维经度数组（升序）
        lat_1d : 一维纬度数组（升序）
        boundary_pts: (M, 2) 边界点坐标
    返回：
        weights_list: 列表，每个元素为 (indices, coeffs)
            - indices: 四个网格点的二维索引 (i0,j0), (i1,j0), (i0,j1), (i1,j1)
            - coeffs : 对应的四个权重 (w00, w01, w10, w11)
    """
    print("  预计算PM2.5双线性插值权重...")
    lon = np.asarray(lon_1d)
    lat = np.asarray(lat_1d)
    nlon = len(lon)
    nlat = len(lat)

    weights_list = []
    for (plon, plat) in boundary_pts:
        i = np.searchsorted(lon, plon) - 1
        i = np.clip(i, 0, nlon - 2)
        j = np.searchsorted(lat, plat) - 1
        j = np.clip(j, 0, nlat - 2)

        lon0, lon1 = lon[i], lon[i + 1]
        lat0, lat1 = lat[j], lat[j + 1]

        dx = (plon - lon0) / (lon1 - lon0) if lon1 > lon0 else 0.5
        dy = (plat - lat0) / (lat1 - lat0) if lat1 > lat0 else 0.5

        w00 = (1 - dx) * (1 - dy)
        w01 = dx * (1 - dy)
        w10 = (1 - dx) * dy
        w11 = dx * dy

        indices = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
        coeffs = np.array([w00, w01, w10, w11])
        weights_list.append((indices, coeffs))
    return weights_list


# ------------------ 应用预计算权重快速插值 ------------------
def interpolate_wind_from_weights(u_valid, v_valid, weights_list):
    """根据预计算权重快速插值风场到边界点（基于有效点集）"""
    u_vals = np.zeros(len(weights_list))
    v_vals = np.zeros(len(weights_list))
    for k, (indices, coeffs) in enumerate(weights_list):
        if len(indices) == 3:  # 三角形插值
            u_vals[k] = np.dot(u_valid[indices], coeffs)
            v_vals[k] = np.dot(v_valid[indices], coeffs)
        else:  # 最近邻
            u_vals[k] = u_valid[indices[0]]
            v_vals[k] = v_valid[indices[0]]
    return u_vals, v_vals


def interpolate_pm25_from_weights(pm25_2d, weights_list):
    """根据预计算权重快速插值PM2.5到边界点"""
    vals = np.zeros(len(weights_list))
    for k, (indices, coeffs) in enumerate(weights_list):
        v00 = pm25_2d[indices[0][1], indices[0][0]]
        v01 = pm25_2d[indices[1][1], indices[1][0]]
        v10 = pm25_2d[indices[2][1], indices[2][0]]
        v11 = pm25_2d[indices[3][1], indices[3][0]]
        vals[k] = v00 * coeffs[0] + v01 * coeffs[1] + v10 * coeffs[2] + v11 * coeffs[3]
    return vals


def main():
    start_time = time.time()
    r_earth = 6371004

    script_dir = os.path.dirname(os.path.abspath(__file__))
    boundary_path = os.path.join(script_dir, BOUNDARY_DIR)
    output_path = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)

    # ================== 加载目标城市边界 ==================
    print("正在加载目标城市边界...")
    all_boundary_info = {}
    for city in target_cities:
        filename = f"{city}_boundary.csv"
        file_path = os.path.join(boundary_path, filename)
        if not os.path.exists(file_path):
            print(f"⚠️  警告：未找到 {city} 边界文件：{filename}，跳过")
            continue
        try:
            df = pd.read_csv(file_path, index_col=0)
            lat_vals = df.iloc[:, 0].values
            lon_vals = df.iloc[:, 1].values
            all_boundary_info[city] = np.column_stack([lon_vals, lat_vals])
            print(f"✅ 已加载：{city}")
        except Exception as e:
            print(f"❌ 读取 {city} 失败：{e}")

    city_names = list(all_boundary_info.keys())
    if not city_names:
        print("没有成功加载任何城市边界，退出")
        return
    print(f"\n🎯 最终参与计算的城市：{city_names}")

    # ================== 预提取NC文件坐标并预计算插值权重 ==================
    print("\n预提取NC文件网格坐标...")

    # --- 风场网格坐标 (不规则网格，可能含掩码/NaN) ---
    lon_wind_raw, lat_wind_raw = None, None
    test_date = START_DATE
    found = False
    while test_date <= END_DATE and not found:
        for hour in range(24):
            # 北京时间转UTC时间（测试文件查找）
            cst_dt = datetime.datetime.combine(test_date, datetime.time(hour=hour))
            utc_dt = cst_dt - datetime.timedelta(hours=8)
            utc_date_str = utc_dt.strftime("%Y%m%d")
            utc_hour_str = f"{utc_dt.hour:02d}"

            test_file = os.path.join(WIND_DIR, WIND_FILE_TEMPLATE.format(utc_date_str, utc_hour_str))
            if os.path.exists(test_file):
                with Dataset(test_file, 'r') as nc:
                    lat_wind_raw = nc.variables['lat'][:].astype(np.float64)
                    lon_wind_raw = nc.variables['lon'][:].astype(np.float64)
                    # 处理掩码数组
                    if isinstance(lat_wind_raw, np.ma.MaskedArray):
                        lat_wind_raw = lat_wind_raw.filled(np.nan)
                    if isinstance(lon_wind_raw, np.ma.MaskedArray):
                        lon_wind_raw = lon_wind_raw.filled(np.nan)
                found = True
                break
        test_date += datetime.timedelta(days=1)
    if not found:
        print("未找到风场NC文件，退出")
        return

    # 保留原始形状，并构建有效点集用于三角剖分
    wind_shape = lon_wind_raw.shape
    lon_flat = lon_wind_raw.ravel()
    lat_flat = lat_wind_raw.ravel()
    valid_mask = ~(np.isnan(lon_flat) | np.isnan(lat_flat))
    wind_points_valid = np.column_stack([lon_flat[valid_mask], lat_flat[valid_mask]])
    wind_valid_mask = valid_mask  # 保存用于筛选风场值

    # 为每个城市预计算风场插值权重（基于有效点集）
    print("\n预计算风场插值权重（每个城市）...")
    wind_weights = {}
    for city, pts in all_boundary_info.items():
        wind_weights[city] = precompute_wind_weights(wind_points_valid, pts)

    # --- PM2.5网格坐标 (规则网格，使用CST时间查找) ---
    lon_pm25, lat_pm25 = None, None
    test_date = START_DATE
    found = False
    while test_date <= END_DATE and not found:
        for hour in range(24):
            # 直接使用CST日期和小时（PM2.5文件按CST命名）
            cst_date_str = test_date.strftime("%Y%m%d")
            cst_hour_str = f"{hour:02d}"
            test_file = os.path.join(PM25_DIR, PM25_FILE_TEMPLATE.format(cst_date_str, cst_hour_str))
            if os.path.exists(test_file):
                with Dataset(test_file, 'r') as nc:
                    lat_pm25 = nc.variables['lat'][:].astype(np.float64)
                    lon_pm25 = nc.variables['lon'][:].astype(np.float64)
                    if isinstance(lat_pm25, np.ma.MaskedArray):
                        lat_pm25 = lat_pm25.filled(np.nan)
                    if isinstance(lon_pm25, np.ma.MaskedArray):
                        lon_pm25 = lon_pm25.filled(np.nan)
                found = True
                break
        test_date += datetime.timedelta(days=1)
    if not found:
        print("未找到PM2.5 NC文件，退出")
        return

    # 确保为一维（规则网格），若为二维则取第一行/列
    if lon_pm25.ndim == 2:
        lon_pm25 = lon_pm25[0, :]
    if lat_pm25.ndim == 2:
        lat_pm25 = lat_pm25[:, 0]
    # 确保升序（通常已经是）
    lon_pm25 = np.sort(lon_pm25)
    lat_pm25 = np.sort(lat_pm25)

    # 为每个城市预计算PM2.5插值权重
    print("预计算PM2.5双线性插值权重...")
    pm25_weights = {}
    for city, pts in all_boundary_info.items():
        pm25_weights[city] = precompute_pm25_weights(lon_pm25, lat_pm25, pts)

    # ================== 主循环 ==================
    current_date = START_DATE
    while current_date <= END_DATE:
        date_str = current_date.strftime("%Y%m%d")
        print(f"\n========== 处理日期: {date_str} (CST) ==========")

        # ---------- 读取并插值PBL（每天一次，保持原日期逻辑） ----------
        pbl_file = os.path.join(PBL_DIR, PBL_FILE_TEMPLATE.format(date_str))
        try:
            pbl_df = pd.read_csv(pbl_file)
            pbl_lon = pbl_df['lon'].values
            pbl_lat = pbl_df['lat'].values
            pbl_val = pbl_df['pbl_heightchem'].values
            valid = ~(np.isnan(pbl_lon) | np.isnan(pbl_lat) | np.isnan(pbl_val))
            pbl_points = np.column_stack([pbl_lon[valid], pbl_lat[valid]])
            pbl_values = pbl_val[valid]

            pbl_boundary = {}
            for city, pts in all_boundary_info.items():
                pbl_interp = griddata(pbl_points, pbl_values, pts, method='linear', fill_value=500.0)
                pbl_boundary[city] = pbl_interp

            # 将PBL插值到风场网格的所有点上（原始形状，包括无效点）
            pbl_on_wind_valid = griddata(pbl_points, pbl_values, wind_points_valid, method='linear', fill_value=500.0)
            # 创建与原始网格形状相同的数组，用fill_value填充，然后将有效点位置的值填入
            pbl_on_wind_grid = np.full(wind_shape, 500.0, dtype=np.float64)
            pbl_on_wind_grid.ravel()[wind_valid_mask] = pbl_on_wind_valid
        except Exception as e:
            print(f"PBL文件处理失败: {e}")
            current_date += datetime.timedelta(days=1)
            continue

        # 初始化每日净通量
        daily_net_flux = {name: 0.0 for name in city_names}

        # ---------- 小时循环 ----------
        for hour in range(24):
            # 北京时间转UTC时间（仅用于风场文件）
            cst_dt = datetime.datetime.combine(current_date, datetime.time(hour=hour))
            utc_dt = cst_dt - datetime.timedelta(hours=8)
            utc_date_str = utc_dt.strftime("%Y%m%d")
            utc_hour_str = f"{utc_dt.hour:02d}"

            wind_file = os.path.join(WIND_DIR, WIND_FILE_TEMPLATE.format(utc_date_str, utc_hour_str))

            # PM2.5文件使用CST时间（已修改）
            cst_date_str = current_date.strftime("%Y%m%d")
            cst_hour_str = f"{hour:02d}"
            pm25_file = os.path.join(PM25_DIR, PM25_FILE_TEMPLATE.format(cst_date_str, cst_hour_str))

            if not (os.path.exists(wind_file) and os.path.exists(pm25_file)):
                print(f"  跳过缺失文件: UTC {utc_date_str}{utc_hour_str} (对应CST {date_str}{hour:02d})")
                continue

            try:
                # ----- 读取风场 -----
                with Dataset(wind_file, 'r') as nc:
                    u_raw = nc.variables['u'][:]
                    v_raw = nc.variables['v'][:]
                    height_raw = nc.variables['height'][:]

                # 处理掩膜数组
                if isinstance(u_raw, np.ma.MaskedArray):
                    u_raw = u_raw.filled(np.nan)
                if isinstance(v_raw, np.ma.MaskedArray):
                    v_raw = v_raw.filled(np.nan)
                if isinstance(height_raw, np.ma.MaskedArray):
                    height_raw = height_raw.filled(np.nan)

                u = u_raw.astype(np.float64)
                v = v_raw.astype(np.float64)
                height = height_raw.astype(np.float64)

                nz, ny, nx = u.shape

                # 展平以便垂直积分
                u_flat = u.reshape(nz, -1)
                v_flat = v.reshape(nz, -1)
                h_flat = height.reshape(nz, -1)
                pbl_flat = pbl_on_wind_grid.ravel()

                # 垂直积分（加权风）
                u_weighted_flat, v_weighted_flat = compute_weighted_wind(u_flat, v_flat, h_flat, pbl_flat)

                # 筛选有效点的风场值（与 wind_points_valid 对应）
                u_valid = u_weighted_flat[wind_valid_mask]
                v_valid = v_weighted_flat[wind_valid_mask]

                # ----- 读取PM2.5 -----
                with Dataset(pm25_file, 'r') as nc:
                    pm25_raw = nc.variables['pm25'][:]
                if isinstance(pm25_raw, np.ma.MaskedArray):
                    pm25_raw = pm25_raw.filled(np.nan)
                pm25 = pm25_raw.astype(np.float64)  # 应为二维 (lat, lon)

                # ----- 对每个城市计算净通量 -----
                for city in city_names:
                    # 风场插值
                    u_city, v_city = interpolate_wind_from_weights(u_valid, v_valid, wind_weights[city])
                    # PM2.5插值
                    pm25_city = interpolate_pm25_from_weights(pm25, pm25_weights[city])
                    # PBL已预插值
                    pbl_city = pbl_boundary[city]

                    # 计算小时净通量（修正后的函数）
                    hourly_flux = calculate_net_flux(all_boundary_info[city],
                                                     u_city, v_city,
                                                     pm25_city, pbl_city,
                                                     r_earth)
                    daily_net_flux[city] += hourly_flux

                gc.collect()

            except Exception as e:
                print(f"  小时处理异常 UTC {utc_date_str}{utc_hour_str} (对应CST {date_str}{hour:02d}): {e}")
                continue

        # 保存每日结果
        convert_factor = 8.64e-8  # μg/s → 吨/天
        result_data = []
        for city in city_names:
            net_flux_td = daily_net_flux[city] * convert_factor
            result_data.append({
                'city_name': city,
                'net_flux_ug_s': daily_net_flux[city],
                'net_flux_ton_day': net_flux_td,
                'flux_direction': '流出' if net_flux_td > 0 else '流入'
            })

        result_df = pd.DataFrame(result_data)
        out_file = os.path.join(output_path, f"city_net_flux_{date_str}.csv")
        result_df.to_csv(out_file, index=False, encoding='utf-8')
        print(f"✅ 已保存净通量结果：{out_file}")

        current_date += datetime.timedelta(days=1)
        gc.collect()

    total_minutes = (time.time() - start_time) / 60
    print(f"\n🎉 全部处理完成！总耗时：{total_minutes:.2f} 分钟")


if __name__ == '__main__':
    main()