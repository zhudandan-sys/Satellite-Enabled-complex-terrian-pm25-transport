import numpy as np
import pandas as pd
import datetime
import os
import warnings
from netCDF4 import Dataset
from math import pi
from scipy.interpolate import RegularGridInterpolator  # 用于降水重采样

warnings.filterwarnings("ignore")

# ==================== 常数定义 ====================
C_s = 0.075  # PM2.5浓度限值，单位：mg/m³ (24h二级限值)
W_r = 600000.0  # 雨洗常数（无量纲，公式中取值6e5）
sqrt_pi_half = np.sqrt(pi) / 2  # √π/2

# ==================== 网格参数 ====================
lon_begin, lon_end = 105.0, 115.0
lat_begin, lat_end = 31.0, 41.0
res = 0.1  # 分析网格分辨率（度）

lon_grid = np.arange(lon_begin, lon_end + res / 2, res)
lat_grid = np.arange(lat_begin, lat_end + res / 2, res)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
grid_shape = lon_mesh.shape  # (101, 101)

# ==================== 城市配置 ====================
target_cities = ['xian', 'yuncheng', 'lishi', 'baoji', 'xianyang',
                 'tongchuan', 'linfen', 'luoyang', 'sanmenxia', 'weinan', 'yuci']
BOUNDARY_DIR = r"C:\Users\zyd\PycharmProjects\boundary_grid_province"


# ==================== 辅助函数 ====================
def point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def load_city_boundary(city_name):
    boundary_file = os.path.join(BOUNDARY_DIR, f"{city_name}_boundary.csv")
    if not os.path.exists(boundary_file):
        print(f"警告: 未找到城市边界文件 {boundary_file}")
        return None
    try:
        df = pd.read_csv(boundary_file, index_col=0)
        lat_vals = df.iloc[:, 0].values
        lon_vals = df.iloc[:, 1].values
        poly_points = list(zip(lon_vals, lat_vals))
        return poly_points
    except Exception as e:
        print(f"读取城市边界失败 {city_name}: {e}")
        return None


def calc_ventilation_index(u_profile, v_profile, height_profile, pbl_height):
    if pbl_height <= 0:
        return 0.0
    nz = len(height_profile)
    V_E = 0.0
    for k in range(nz):
        z_center = height_profile[k]
        if z_center > pbl_height:
            break
        if k == 0:
            if nz > 1:
                thickness = (height_profile[0] + height_profile[1]) / 2.0
            else:
                thickness = height_profile[0]
        elif k == nz - 1:
            thickness = height_profile[k] - (height_profile[k - 1] + height_profile[k]) / 2.0
        else:
            thickness = (height_profile[k + 1] - height_profile[k - 1]) / 2.0
        upper_bound = z_center + thickness / 2.0
        if upper_bound > pbl_height:
            lower_bound = z_center - thickness / 2.0
            thickness = pbl_height - lower_bound
            if thickness <= 0:
                continue
        speed = np.sqrt(u_profile[k] ** 2 + v_profile[k] ** 2)
        V_E += speed * thickness
    return V_E


def get_pbl_grid(date, pbl_dir):
    time_str = date.strftime("%Y%m%d")
    pbl_file = os.path.join(pbl_dir, f"pbl_height_{time_str}.csv")
    if not os.path.exists(pbl_file):
        print(f"警告: 边界层文件不存在 {pbl_file}")
        return np.full(grid_shape, np.nan)
    try:
        pbl_df = pd.read_csv(pbl_file)
        pbl_grid = np.zeros(grid_shape)
        nlon = round((lon_end - lon_begin) * 10) + 1
        for j in range(grid_shape[0]):
            for i in range(grid_shape[1]):
                idx = nlon * round(10 * (lat_mesh[j, i] - lat_begin)) + \
                      round(10 * (lon_mesh[j, i] - lon_begin))
                try:
                    pbl_grid[j, i] = float(pbl_df.loc[idx, 'pbl_heightchem'])
                except:
                    pbl_grid[j, i] = np.nan
        return pbl_grid
    except Exception as e:
        print(f"读取边界层数据失败 {time_str}: {e}")
        return np.full(grid_shape, np.nan)


def get_precip_grid(date, precip_nc_path, var_name='prec'):
    """
    从NetCDF文件中读取指定日期的降水场（mm/day），并重采样到分析网格（0.1°）
    """
    try:
        with Dataset(precip_nc_path, 'r') as nc:
            time_var = nc.variables['time']
            time_units = time_var.units
            from netCDF4 import num2date
            times = num2date(time_var[:], time_units)

            # 修复：cftime 对象没有 .date() 方法，直接比较年、月、日
            target_date = date
            idx = None
            for i, t in enumerate(times):
                if t.year == target_date.year and t.month == target_date.month and t.day == target_date.day:
                    idx = i
                    break
            if idx is None:
                print(f"降水数据中未找到日期 {target_date}")
                return np.full(grid_shape, 0.0)

            # 读取经纬度
            lon_nc = nc.variables['lon'][:]
            lat_nc = nc.variables['lat'][:]
            precip_2d = nc.variables[var_name][idx, :, :]  # (lat, lon)

        # 创建插值器（双线性插值）
        if lat_nc[0] > lat_nc[-1]:
            lat_nc = lat_nc[::-1]
            precip_2d = precip_2d[::-1, :]
        if lon_nc[0] > lon_nc[-1]:
            lon_nc = lon_nc[::-1]
            precip_2d = precip_2d[:, ::-1]

        interp_func = RegularGridInterpolator((lat_nc, lon_nc), precip_2d,
                                              bounds_error=False, fill_value=0.0)
        points = np.stack([lat_mesh.ravel(), lon_mesh.ravel()], axis=-1)
        precip_interp = interp_func(points).reshape(grid_shape)
        return precip_interp
    except Exception as e:
        print(f"读取降水数据失败: {e}")
        return np.full(grid_shape, 0.0)


def calculate_city_asi(date, wind_dir, pbl_dir, precip_nc_path, city_polygons):
    # 1. 获取边界层高度网格
    pbl_grid = get_pbl_grid(date, pbl_dir)
    if np.all(np.isnan(pbl_grid)):
        return {city: np.nan for city in city_polygons.keys()}

    # 2. 获取当日降水网格（mm/day）
    precip_grid = get_precip_grid(date, precip_nc_path, var_name='prec')

    # 3. 初始化累加器
    daily_ve_sum = np.zeros(grid_shape)
    valid_hours_grid = np.zeros(grid_shape, dtype=int)

    # 4. 逐小时处理风场
    time_str = date.strftime("%Y%m%d")
    for hour in range(24):
        hour_str = f"{hour:02d}"
        wind_file = os.path.join(wind_dir, f"{time_str}{hour_str}.nc")
        if not os.path.exists(wind_file):
            continue
        try:
            with Dataset(wind_file, 'r') as nc:
                u_data = nc.variables['u'][:]  # (level, lat, lon)
                v_data = nc.variables['v'][:]
                height_3d = nc.variables['height'][:]
                nlev, nlat, nlon = u_data.shape
        except Exception as e:
            print(f"读取风场文件失败 {wind_file}: {e}")
            continue

        for j in range(grid_shape[0]):
            for i in range(grid_shape[1]):
                if np.isnan(pbl_grid[j, i]):
                    continue
                lat_idx = int(round((lat_mesh[j, i] - lat_begin) / res))
                lon_idx = int(round((lon_mesh[j, i] - lon_begin) / res))
                lat_idx = max(0, min(lat_idx, nlat - 1))
                lon_idx = max(0, min(lon_idx, nlon - 1))
                u_profile = u_data[:, lat_idx, lon_idx]
                v_profile = v_data[:, lat_idx, lon_idx]
                height_profile = height_3d[:, lat_idx, lon_idx]
                V_E = calc_ventilation_index(u_profile, v_profile, height_profile, pbl_grid[j, i])
                daily_ve_sum[j, i] += V_E * 3600
                valid_hours_grid[j, i] += 1

    if np.sum(valid_hours_grid) == 0:
        print(f"日期 {time_str} 无有效小时数据")
        return {city: np.nan for city in city_polygons.keys()}

    # 5. 计算日平均通风量 (m²/s)
    daily_ve_full = np.full(grid_shape, np.nan)
    mask = valid_hours_grid > 0
    daily_ve_full[mask] = daily_ve_sum[mask] * (24.0 / valid_hours_grid[mask])

    # 6. 计算网格面积 (m²)
    R = 6371000.0
    dlon = np.radians(res)
    dlat = np.radians(res)
    lat_rad = np.radians(lat_mesh)
    s_grid = R ** 2 * np.cos(lat_rad) * dlon * dlat

    # 7. 计算通风项贡献 (吨/天·km²)
    ASI_vent = np.full(grid_shape, np.nan)
    valid_vent = ~np.isnan(daily_ve_full) & (s_grid > 0)
    ASI_vent[valid_vent] = (sqrt_pi_half * daily_ve_full[valid_vent] * C_s /
                            np.sqrt(s_grid[valid_vent]) * 0.001)

    # 8. 计算雨洗项贡献 (吨/天·km²)
    #    公式: W_r * R * C_s * 0.000001，其中 R 单位 mm/day
    ASI_rain = np.full(grid_shape, 0.0)
    valid_rain = ~np.isnan(precip_grid)
    ASI_rain[valid_rain] = W_r * precip_grid[valid_rain] * C_s * 0.000001

    # 9. 总 ASI
    ASI_grid = ASI_vent + ASI_rain

    # 10. 对每个城市求平均
    city_asi = {}
    for city, poly_points in city_polygons.items():
        if poly_points is None:
            city_asi[city] = np.nan
            continue
        inside = np.zeros(grid_shape, dtype=bool)
        for j in range(grid_shape[0]):
            for i in range(grid_shape[1]):
                if point_in_polygon(lon_mesh[j, i], lat_mesh[j, i], poly_points):
                    inside[j, i] = True
        asi_vals = ASI_grid[inside]
        asi_vals = asi_vals[~np.isnan(asi_vals)]
        if len(asi_vals) == 0:
            city_asi[city] = np.nan
        else:
            city_asi[city] = np.mean(asi_vals)
    return city_asi


def main():
    # ==================== 用户配置 ====================
    wind_data_dir = r"E:\wind_24h"  # 风场NC文件目录
    pbl_data_dir = r"C:\Users\zyd\PycharmProjects\pbl_height"
    # 降水文件路径模板，用 {year} 占位年份
    precip_nc_template = r"E:\CHM_PRE V2\daily\CHM_PRE_V2_daily_{year}.nc"
    output_file = "city_average_ASI_with_rain.xlsx"

    start_date = datetime.date(2023, 3, 1)
    end_date = datetime.date(2024, 2, 29)
    # ================================================

    # 加载城市边界
    print("加载城市边界...")
    city_polygons = {}
    for city in target_cities:
        poly = load_city_boundary(city)
        if poly is not None:
            city_polygons[city] = poly
            print(f"  ✅ {city}")
        else:
            print(f"  ❌ {city} 边界加载失败，跳过")
    if not city_polygons:
        print("没有成功加载任何城市边界，退出")
        return

    delta = datetime.timedelta(days=1)
    results = []
    current_date = start_date
    total_days = (end_date - start_date).days + 1
    print(f"\n开始计算 {total_days} 天的城市平均大气自净能力指数（含雨洗项，吨/天·km²）...")

    while current_date <= end_date:
        print(f"正在处理: {current_date.strftime('%Y-%m-%d')}")
        # 根据年份动态生成降水文件路径
        year = current_date.year
        precip_nc_file = precip_nc_template.format(year=year)
        # 如果对应年份文件不存在，可给出提示（但继续运行，get_precip_grid 会返回 0）
        if not os.path.exists(precip_nc_file):
            print(f"    警告: 降水文件不存在 {precip_nc_file}，雨洗项将置零")

        city_asi = calculate_city_asi(current_date, wind_data_dir, pbl_data_dir,
                                      precip_nc_file, city_polygons)
        for city, asi_val in city_asi.items():
            results.append({
                '日期': current_date,
                '城市': city,
                '平均ASI(吨/天·km²)': asi_val
            })
        current_date += delta

    df_result = pd.DataFrame(results)
    df_result.to_excel(output_file, index=False, float_format='%.6f')
    print(f"\n计算完成！结果已保存至: {output_file}")
    print(f"有效记录数: {df_result['平均ASI(吨/天·km²)'].notna().sum()} / {len(df_result)}")

if __name__ == "__main__":
    main()