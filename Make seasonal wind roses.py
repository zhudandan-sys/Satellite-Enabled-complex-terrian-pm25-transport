import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== 全局字体设置 ====================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 16

# ==================== 基础配置 ====================
TARGET_CITIES_CN = ['西安市', '运城市', '榆次市', '宝鸡市', '咸阳市',
                    '铜川市', '临汾市', '洛阳市', '三门峡市', '渭南市', '离石县']

# 区域定义
REGIONS = {
    "Weihe_Plain": {
        "name_cn": "Weihe_Plain",
        "cities": ["西安市","咸阳市","渭南市","宝鸡市","铜川市","渭南市"]
    },
    "Fenhe_Plain": {
        "name_cn": "Fenhe_Plain",
        "cities": ["临汾市","离石县","榆次市"]
    },
    "Yuncheng-Sanmenxia-Luoyang": {
        "name_cn": "Yuncheng-Sanmenxia-Luoyang",
        "cities": ["洛阳市","三门峡市","运城市"]
    }
}

CITY_TO_REGION = {}
for region_en, info in REGIONS.items():
    for city in info["cities"]:
        CITY_TO_REGION[city] = region_en

# 数据路径
WIND_DIR = r"E:\wind_24h"
PBL_DIR = r"C:\Users\zyd\PycharmProjects\pbl_height"
SHAPEFILE_PATH = r"D:\科研\中国ArcGIS数据(到县界、Lambert投影) 备份\Lambert\中国地州界.shp"

LOCAL_TZ_OFFSET = 8

# 风向分级 (16方位)
WIND_BINS = np.linspace(0, 360, 17)
WIND_LABELS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
               'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

# 风速分级 & 灰度配色
SPEED_BINS = [0, 2, 4, 6, 8, 30]
SPEED_LABELS = ['0-2', '2-4', '4-6', '6-8', '>8']
N_SPEED_BINS = len(SPEED_LABELS)
#SPEED_COLORS = plt.cm.gray_r(np.linspace(0.2, 0.9, N_SPEED_BINS))黑色系
SPEED_COLORS = plt.cm.Purples(np.linspace(0.2, 0.9, N_SPEED_BINS))

# 单季节最小有效样本数
MIN_SAMPLES_PER_SEASON = 10

# 季节判断
def get_season(month):
    if 3 <= month <= 5:
        return 'spring'
    elif 6 <= month <= 8:
        return 'summer'
    elif 9 <= month <= 11:
        return 'autumn'
    else:
        return 'winter'

SEASON_NAMES_CN = {
    'spring': '春季',
    'summer': '夏季',
    'autumn': '秋季',
    'winter': '冬季'
}

# 污染事件时间列表
pollution_events = [
    ("2023-10-10", "2023-10-18", "事件1"),
    ("2023-11-12", "2023-11-17", "事件2"),
    ("2023-11-26", "2023-11-29", "事件3"),
    ("2023-03-10", "2023-03-15", "事件4"),
    ("2024-02-25", "2024-02-29", "事件5"),
    ("2023-03-02", "2023-03-09", "事件6"),
    ("2023-03-18", "2023-03-20", "事件7"),
    ("2023-10-21", "2023-11-02", "事件8"),
    ("2023-12-18", "2023-12-26", "事件9"),
    ("2023-12-08", "2023-12-14", "事件10"),
    ("2023-04-06", "2023-04-09", "事件11"),
    ("2023-12-02", "2023-12-06", "事件12"),
    ("2023-03-21", "2023-03-28", "事件13"),
    ("2024-02-18", "2024-02-20", "事件14"),
    ("2024-01-11", "2024-01-21", "事件15"),
    ("2023-04-10", "2023-04-15", "事件16"),
    ("2024-01-28", "2024-02-14", "事件17"),
    ("2023-04-19", "2023-04-21", "事件18"),
    ("2023-12-27", "2024-01-10", "事件19")
]

# ==================== 工具函数 ====================
def get_city_center(city_name, gdf):
    mask = gdf['NAME'].str.contains(city_name, na=False)
    if not mask.any():
        raise ValueError(f"City not found: {city_name}")
    geom = gdf[mask].iloc[0].geometry
    center = geom.centroid
    return center.y, center.x

def find_nearest_grid_point_2d(lat, lon, lats, lons):
    dist = (lats - lat)**2 + (lons - lon)**2
    idx_flat = np.argmin(dist)
    return np.unravel_index(idx_flat, lats.shape)

def get_pbl_height(city_lat, city_lon, pbl_df):
    lat_vals = pd.to_numeric(pbl_df['lat'], errors='coerce')
    lon_vals = pd.to_numeric(pbl_df['lon'], errors='coerce')
    valid = lat_vals.notna() & lon_vals.notna()
    if not valid.any():
        raise ValueError("No valid lat/lon in PBL data")
    dist_sq = (lat_vals[valid] - city_lat)**2 + (lon_vals[valid] - city_lon)**2
    idx_min = dist_sq.idxmin()
    return pbl_df.loc[idx_min, 'pbl_heightchem']

def compute_weighted_wind(u_profile, v_profile, height_profile, pbl_height):
    if np.max(height_profile) < 10000:
        cum_thick = 0.0
        u_sum = 0.0
        v_sum = 0.0
        total_thick = 0.0
        for k, thick in enumerate(height_profile):
            if cum_thick + thick <= pbl_height:
                u_sum += u_profile[k] * thick
                v_sum += v_profile[k] * thick
                total_thick += thick
                cum_thick += thick
            else:
                frac = (pbl_height - cum_thick) / thick
                if frac > 0:
                    u_sum += u_profile[k] * thick * frac
                    v_sum += v_profile[k] * thick * frac
                    total_thick += thick * frac
                break
        if total_thick > 0:
            return u_sum / total_thick, v_sum / total_thick
        else:
            return u_profile[0], v_profile[0]
    else:
        mask = height_profile <= pbl_height
        if np.any(mask):
            return np.mean(u_profile[mask]), np.mean(v_profile[mask])
        else:
            return u_profile[0], v_profile[0]

def compute_wind_speed_dir(u, v):
    speed = np.sqrt(u**2 + v**2)
    wd_math = np.arctan2(v, u) * 180 / np.pi
    wd = (270 - wd_math) % 360
    return speed, wd

# ==================== 1. 风玫瑰绘图函数（移除标题、图例，仅保留图形） ====================
def plot_wind_rose(samples, region_en, region_cn, season_key, output_dir):
    if len(samples) < MIN_SAMPLES_PER_SEASON:
        print(f"    Not enough samples for {region_en} - {season_key} (n={len(samples)}), skipped")
        return

    total_n = len(samples)
    df = pd.DataFrame(samples)
    df['wind_bin'] = pd.cut(df['wind_dir'], bins=WIND_BINS, labels=WIND_LABELS, right=False)
    df['speed_bin'] = pd.cut(df['wind_speed'], bins=SPEED_BINS, labels=SPEED_LABELS, right=False)

    grouped = df.groupby(['wind_bin', 'speed_bin'], observed=True).size().unstack(fill_value=0)
    grouped = grouped.reindex(WIND_LABELS, fill_value=0)
    grouped = grouped.reindex(columns=SPEED_LABELS, fill_value=0)
    percent_table = grouped / total_n * 100

    y_max = 20.0
    bin_centers_deg = (WIND_BINS[:-1] + WIND_BINS[1:]) / 2
    bin_centers_rad = np.deg2rad(bin_centers_deg)
    width_rad = np.deg2rad(22.5)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # 隐藏所有网格、刻度、外框
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.set_ylabel('')

    # 绘制风玫瑰主体
    for i, wind_dir in enumerate(WIND_LABELS):
        speeds_percent = percent_table.loc[wind_dir].values
        bottom = 0.0
        for j, percent in enumerate(speeds_percent):
            if percent > 0:
                ax.bar(bin_centers_rad[i], percent, width=width_rad,
                       bottom=bottom, color=SPEED_COLORS[j],
                       edgecolor='white', linewidth=0.2, alpha=0.9)
            bottom += percent

    ax.set_ylim(0, y_max)

    # ========== 关键修改：删除标题、图例绘制代码 ==========

    # 保存图片（透明背景）
    out_filename = f"{season_key}.png"
    out_path = os.path.join(output_dir, out_filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"    Saved rose: {out_path}")

# ==================== 2. 新增：单独绘制风速图例（独立图片） ====================
def save_legend_only(output_root):
    """单独生成风速图例图片，全局仅生成1张"""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')  # 隐藏坐标轴

    # 构造图例色块
    legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=SPEED_COLORS[j],
                                    edgecolor='black', linewidth=0.5)
                      for j in range(N_SPEED_BINS)]
    leg = ax.legend(legend_patches, SPEED_LABELS, title='Wind Speed (m/s)',
                    loc='center', frameon=True, fontsize=16, title_fontsize=16)
    plt.setp(leg.get_title(), fontsize=16)

    # 保存图例图片
    legend_path = os.path.join(output_root, "wind_speed_legend.png")
    plt.savefig(legend_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"\n>>> 独立图例已保存至: {legend_path}")

# ==================== 时间序列生成函数 ====================
def get_all_datetimes_from_events(events):
    all_dt_set = set()
    for start_str, end_str, _ in events:
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        current = start
        while current <= end:
            for hour in range(24):
                local_dt = current.replace(hour=hour, minute=0, second=0, microsecond=0)
                if local_dt > end + timedelta(days=1) - timedelta(seconds=1):
                    break
                all_dt_set.add(local_dt)
            current += timedelta(days=1)
    return sorted(all_dt_set)

def get_summer_2023_datetimes():
    summer_dt_set = set()
    start = datetime(2023, 6, 1)
    end = datetime(2023, 8, 31)
    current = start
    while current <= end:
        for hour in range(24):
            local_dt = current.replace(hour=hour, minute=0, second=0, microsecond=0)
            summer_dt_set.add(local_dt)
        current += timedelta(days=1)
    return sorted(list(summer_dt_set))

# ==================== 主处理函数 ====================
def process_combined_events_by_season(events, output_root):
    print("\n========== 开始处理数据并绘图 ==========")
    os.makedirs(output_root, exist_ok=True)

    # 第一步：单独保存图例图片
    save_legend_only(output_root)

    # 加载行政区矢量
    gdf = gpd.read_file(SHAPEFILE_PATH, encoding='gbk')
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')

    # 获取城市中心点
    city_centers = {}
    for city_cn in TARGET_CITIES_CN:
        try:
            lat, lon = get_city_center(city_cn, gdf)
            city_centers[city_cn] = (lat, lon)
            print(f"{city_cn}: center ({lat:.4f}, {lon:.4f})")
        except Exception as e:
            print(f"Warning: skipping city {city_cn} - {e}")
    if not city_centers:
        raise RuntimeError("No valid city data")

    # 读取风场网格信息
    sample_nc = None
    for f in os.listdir(WIND_DIR):
        if f.endswith('.nc'):
            sample_nc = os.path.join(WIND_DIR, f)
            break
    if sample_nc is None:
        raise FileNotFoundError(f"No wind file found in {WIND_DIR}")
    ds_sample = xr.open_dataset(sample_nc)
    grid_lats = ds_sample['lat'].values
    grid_lons = ds_sample['lon'].values
    height_profile = ds_sample['height'].values
    ds_sample.close()
    print(f"Wind grid lat range: {grid_lats.min():.2f}~{grid_lats.max():.2f}, lon range: {grid_lons.min():.2f}~{grid_lons.max():.2f}")

    # 匹配网格点位
    city_wind_info = {}
    for city_cn, (lat, lon) in city_centers.items():
        if grid_lats.ndim == 1 and grid_lons.ndim == 1:
            lat_idx = np.argmin(np.abs(grid_lats - lat))
            lon_idx = np.argmin(np.abs(grid_lons - lon))
        else:
            lat_idx, lon_idx = find_nearest_grid_point_2d(lat, lon, grid_lats, grid_lons)
        if height_profile.ndim == 1:
            h_profile = height_profile
        elif height_profile.ndim == 3:
            h_profile = height_profile[:, lat_idx, lon_idx]
        else:
            raise ValueError(f"Unexpected height_profile dimension: {height_profile.ndim}")
        city_wind_info[city_cn] = {
            'lat_idx': lat_idx,
            'lon_idx': lon_idx,
            'height_profile': h_profile
        }

    # 初始化样本容器
    samples_by_region_season = {}
    for region_en in REGIONS.keys():
        samples_by_region_season[region_en] = {
            'spring': [],
            'summer': [],
            'autumn': [],
            'winter': []
        }

    # 1. 处理污染事件数据（春、秋、冬，跳过夏季）
    print("\n=== 处理污染事件数据（春/秋/冬） ===")
    event_datetimes = get_all_datetimes_from_events(events)
    print(f"污染事件总时长: {len(event_datetimes)} 小时")

    pbl_cache = {}
    processed = 0
    total = len(event_datetimes)
    for local_dt in event_datetimes:
        processed += 1
        if processed % 500 == 0:
            print(f"  已处理 {processed}/{total} 小时")
        if 6 <= local_dt.month <= 8:
            continue

        local_date_str = local_dt.strftime('%Y%m%d')
        local_hour = local_dt.hour
        utc_dt = local_dt - timedelta(hours=LOCAL_TZ_OFFSET)
        utc_time_str = utc_dt.strftime('%Y%m%d')
        utc_hour = utc_dt.hour

        wind_file = os.path.join(WIND_DIR, f"{utc_time_str}{utc_hour:02d}.nc")
        if not os.path.exists(wind_file):
            wind_file = os.path.join(WIND_DIR, f"{utc_time_str}_{utc_hour:02d}.nc")
        if not os.path.exists(wind_file):
            continue
        try:
            ds_wind = xr.open_dataset(wind_file)
            u = ds_wind['u'].values
            v = ds_wind['v'].values
            ds_wind.close()
        except Exception:
            continue

        date_str = local_date_str
        if date_str not in pbl_cache:
            pbl_file_hour = os.path.join(PBL_DIR, f"pbl_height_{date_str}_{local_hour:02d}.csv")
            pbl_file_day = os.path.join(PBL_DIR, f"pbl_height_{date_str}.csv")
            if os.path.exists(pbl_file_hour):
                pbl_file = pbl_file_hour
            elif os.path.exists(pbl_file_day):
                pbl_file = pbl_file_day
            else:
                pbl_cache[date_str] = None
            if pbl_cache.get(date_str) is None and os.path.exists(pbl_file):
                try:
                    pbl_df = pd.read_csv(pbl_file)
                    pbl_cache[date_str] = pbl_df
                except Exception:
                    pbl_cache[date_str] = None
        df_pbl = pbl_cache.get(date_str)
        if df_pbl is None:
            continue

        season_key = get_season(local_dt.month)
        for city_cn, (city_lat, city_lon) in city_centers.items():
            region_en = CITY_TO_REGION.get(city_cn)
            if region_en is None:
                continue
            info = city_wind_info[city_cn]
            lat_idx = info['lat_idx']
            lon_idx = info['lon_idx']
            h_profile = info['height_profile']

            if u.ndim == 3:
                u_prof = u[:, lat_idx, lon_idx]
                v_prof = v[:, lat_idx, lon_idx]
            else:
                u_prof = u[lat_idx, lon_idx]
                v_prof = v[lat_idx, lon_idx]

            try:
                pbl_h = get_pbl_height(city_lat, city_lon, df_pbl)
            except Exception:
                continue
            u_avg, v_avg = compute_weighted_wind(u_prof, v_prof, h_profile, pbl_h)
            ws, wd = compute_wind_speed_dir(u_avg, v_avg)
            if ws < 0.5:
                continue
            samples_by_region_season[region_en][season_key].append({'wind_dir': wd, 'wind_speed': ws})

    # 2. 单独处理2023年夏季(6-8月)数据
    print("\n=== 处理2023年夏季(6-8月)背景数据 ===")
    summer_datetimes = get_summer_2023_datetimes()
    print(f"夏季总时长: {len(summer_datetimes)} 小时")
    summer_pbl_cache = {}
    processed_summer = 0
    total_summer = len(summer_datetimes)
    for local_dt in summer_datetimes:
        processed_summer += 1
        if processed_summer % 500 == 0:
            print(f"  已处理 {processed_summer}/{total_summer} 小时")

        local_date_str = local_dt.strftime('%Y%m%d')
        local_hour = local_dt.hour
        utc_dt = local_dt - timedelta(hours=LOCAL_TZ_OFFSET)
        utc_time_str = utc_dt.strftime('%Y%m%d')
        utc_hour = utc_dt.hour

        wind_file = os.path.join(WIND_DIR, f"{utc_time_str}{utc_hour:02d}.nc")
        if not os.path.exists(wind_file):
            wind_file = os.path.join(WIND_DIR, f"{utc_time_str}_{utc_hour:02d}.nc")
        if not os.path.exists(wind_file):
            continue
        try:
            ds_wind = xr.open_dataset(wind_file)
            u = ds_wind['u'].values
            v = ds_wind['v'].values
            ds_wind.close()
        except Exception:
            continue

        date_str = local_date_str
        if date_str not in summer_pbl_cache:
            pbl_file_hour = os.path.join(PBL_DIR, f"pbl_height_{date_str}_{local_hour:02d}.csv")
            pbl_file_day = os.path.join(PBL_DIR, f"pbl_height_{date_str}.csv")
            if os.path.exists(pbl_file_hour):
                pbl_file = pbl_file_hour
            elif os.path.exists(pbl_file_day):
                pbl_file = pbl_file_day
            else:
                summer_pbl_cache[date_str] = None
            if summer_pbl_cache.get(date_str) is None and os.path.exists(pbl_file):
                try:
                    pbl_df = pd.read_csv(pbl_file)
                    summer_pbl_cache[date_str] = pbl_df
                except Exception:
                    summer_pbl_cache[date_str] = None
        df_pbl = summer_pbl_cache.get(date_str)
        if df_pbl is None:
            continue

        for city_cn, (city_lat, city_lon) in city_centers.items():
            region_en = CITY_TO_REGION.get(city_cn)
            if region_en is None:
                continue
            info = city_wind_info[city_cn]
            lat_idx = info['lat_idx']
            lon_idx = info['lon_idx']
            h_profile = info['height_profile']

            if u.ndim == 3:
                u_prof = u[:, lat_idx, lon_idx]
                v_prof = v[:, lat_idx, lon_idx]
            else:
                u_prof = u[lat_idx, lon_idx]
                v_prof = v[lat_idx, lon_idx]

            try:
                pbl_h = get_pbl_height(city_lat, city_lon, df_pbl)
            except Exception:
                continue
            u_avg, v_avg = compute_weighted_wind(u_prof, v_prof, h_profile, pbl_h)
            ws, wd = compute_wind_speed_dir(u_avg, v_avg)
            if ws < 0.5:
                continue
            samples_by_region_season[region_en]['summer'].append({'wind_dir': wd, 'wind_speed': ws})

    # 3. 批量绘制风玫瑰（无标题、无图例）
    print("\n=== 绘制风玫瑰图（仅图形主体） ===")
    for region_en, seasons_dict in samples_by_region_season.items():
        region_cn = REGIONS[region_en]["name_cn"]
        region_output_dir = os.path.join(output_root, region_en)
        os.makedirs(region_output_dir, exist_ok=True)
        print(f"  区域: {region_cn}")
        for season_key, samples in seasons_dict.items():
            plot_wind_rose(samples, region_en, region_cn, season_key, region_output_dir)

    print("\n========== 全部任务完成 ==========")

# ==================== 程序入口 ====================
if __name__ == "__main__":
    output_root = r"C:\Users\zyd\PycharmProjects\风玫瑰图_合并事件季节"
    process_combined_events_by_season(pollution_events, output_root)