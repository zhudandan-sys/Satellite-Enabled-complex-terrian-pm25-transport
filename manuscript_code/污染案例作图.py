import numpy as np
import os
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from math import *
from netCDF4 import Dataset
import warnings
import matplotlib.font_manager as fm
import geopandas as gpd
from shapely.geometry import Point, box
import cartopy.crs as ccrs

# 忽略警告
warnings.filterwarnings("ignore")

# 全局PM2.5颜色范围设置
PM25_MIN = 0
PM25_MAX = 120  # 根据实际数据调整此值

# 地球半径（米）
r_earth = 6371004

# ===================== 核心修改1：全局字体设置为Times New Roman =====================
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False


# 检查Times New Roman是否可用（调试用）
def check_times_new_roman():
    font_list = [f.name for f in fm.fontManager.ttflist]
    if 'Times New Roman' in font_list:
        print("✓ Times New Roman字体可用")
    else:
        print("⚠ 未找到Times New Roman字体，可能显示异常")


check_times_new_roman()


# 自定义颜色映射 - 仿照目标图片的colorbar
def create_target_colormap():
    """创建与目标图片颜色条一致的PM2.5颜色映射（适配最大值150，已修复顺序问题）"""
    color_nodes = [
        (0.0, (0.2, 0.0, 0.4)),  # 0μg/m³：深紫色
        (20 / PM25_MAX, (0.0, 0.6, 0.6)),  # 35μg/m³：蓝绿色
        (40 / PM25_MAX, (0.0, 0.8, 0.4)),  # 70μg/m³：绿色
        (60 / PM25_MAX, (1.0, 1.0, 0.2)),  # 105μg/m³：黄色
        (80 / PM25_MAX, (1.0, 0.6, 0.0)),  # 140μg/m³：橙色
        (100 / PM25_MAX, (1.0, 0.2, 0.0)),  # 175μg/m³：红色
        (120 / PM25_MAX, (0.8, 0.0, 0.0))  # 210μg/m³：深红色
    ]
    '''
    color_nodes = [
        (0.0, (0.2, 0.0, 0.4)),  # 0μg/m³：深紫色
        (35 / PM25_MAX, (0.0, 0.6, 0.6)),  # 35μg/m³：蓝绿色
        (70 / PM25_MAX, (0.0, 0.8, 0.4)),  # 70μg/m³：绿色
        (105 / PM25_MAX, (1.0, 1.0, 0.2)),  # 105μg/m³：黄色
        (140 / PM25_MAX, (1.0, 0.6, 0.0)),  # 140μg/m³：橙色
        (175 / PM25_MAX, (1.0, 0.2, 0.0)),  # 175μg/m³：红色
        (210 / PM25_MAX, (0.8, 0.0, 0.0))  # 210μg/m³：深红色
    ]
    '''
    cmap = LinearSegmentedColormap.from_list(
        'target_pm25',
        color_nodes,
        N=256
    )
    norm = Normalize(vmin=PM25_MIN, vmax=PM25_MAX)
    return cmap, norm


# 创建目标颜色映射
TARGET_CMAP, TARGET_NORM = create_target_colormap()


def load_city_boundary(city_file, target_cities=['西安市', '太原市', '郑州市'], target_crs='EPSG:4326'):
    """加载指定城市的边界"""
    try:
        cities = gpd.read_file(city_file)

        # 检查数据列名
        name_col = None
        for col in cities.columns:
            if 'name' in col.lower() or 'NAME' in col or '市' in col:
                name_col = col
                break

        if name_col is None:
            name_col = cities.columns[0]
            print(f"警告: 未找到明确的名称列，使用 {name_col} 作为城市名称列")

        # 坐标系转换
        if cities.crs is None:
            print("警告: 市级数据未定义坐标系，尝试强制设置为EPSG:4326")
            cities = cities.set_crs(target_crs, allow_override=True)

        cities = cities.to_crs(target_crs)

        # 筛选目标城市
        matched_cities = []
        for city in target_cities:
            exact_match = cities[cities[name_col] == city]
            if not exact_match.empty:
                matched_cities.append(city)
            else:
                fuzzy_match = cities[cities[name_col].str.contains(city[:2])]
                if not fuzzy_match.empty:
                    matched_cities.append(fuzzy_match[name_col].iloc[0])

        cities = cities[cities[name_col].isin(matched_cities)]

        if cities.empty:
            print(f"警告: 未找到目标城市 {target_cities}，请检查数据中的城市名称")
            return None

        return cities
    except Exception as e:
        print(f"市级数据加载失败: {str(e)}")
        return None


def calc_pbl_wind(u_data, v_data, heights, pbl_height, lat_idx, lon_idx):
    """计算边界层高度内的加权平均风场"""
    try:
        pbl_height = float(pbl_height)
        if pbl_height <= 0:
            return 0, 0

        max_h = len(heights) - 1
        if not (0 <= lat_idx < u_data.shape[1] and 0 <= lon_idx < u_data.shape[2]):
            return 0, 0

        weighted_u, weighted_v, total_weight = 0.0, 0.0, 0.0

        for h_idx in range(len(heights)):
            height = heights[h_idx]
            if height > pbl_height:
                break

            if h_idx == 0:
                thickness = (heights[1] - heights[0]) / 2 if len(heights) > 1 else heights[0]
            elif h_idx == len(heights) - 1:
                thickness = heights[h_idx] - heights[h_idx - 1]
            else:
                thickness = (heights[h_idx + 1] - heights[h_idx - 1]) / 2

            if h_idx == 0 and heights[0] > pbl_height:
                thickness = pbl_height
            elif height + thickness / 2 > pbl_height:
                thickness = pbl_height - max(0, (height - thickness / 2))

            weight = thickness
            weighted_u += u_data[h_idx, lat_idx, lon_idx] * weight
            weighted_v += v_data[h_idx, lat_idx, lon_idx] * weight
            total_weight += weight

        return (weighted_u / total_weight, weighted_v / total_weight) if total_weight > 0 else (0, 0)
    except Exception as e:
        print(f"calc_pbl_wind error: {str(e)}")
        return 0, 0


def calculate_transport_flux(lon_grid, lat_grid, u_avg, v_avg, pm25_data, pbl_height):
    """计算网格间的传输通量"""
    flux_lon = np.zeros_like(lon_grid)
    flux_lat = np.zeros_like(lat_grid)
    flux_magnitude = np.zeros_like(lon_grid)

    # 计算网格尺寸（米）
    d_lon = np.zeros_like(lon_grid)
    d_lat = np.zeros_like(lat_grid)

    for j in range(lon_grid.shape[0]):
        for i in range(lon_grid.shape[1]):
            d_lon[j, i] = 2 * r_earth * pi * abs(lon_grid[j, min(i + 1, lon_grid.shape[1] - 1)] - lon_grid[j, i]) * \
                          cos(pi * lat_grid[j, i] / 180) / 360
            d_lat[j, i] = 2 * r_earth * pi * abs(lat_grid[min(j + 1, lon_grid.shape[0] - 1), i] - lat_grid[j, i]) / 360

    for j in range(1, lon_grid.shape[0] - 1):
        for i in range(1, lon_grid.shape[1] - 1):
            u_now = u_avg[j, i]
            v_now = v_avg[j, i]
            pm25_now = pm25_data[j, i]
            pbl_now = pbl_height[j, i]

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            total_flux_x = 0
            total_flux_y = 0

            for dj, di in directions:
                nj, ni = j + dj, i + di
                if nj < 0 or nj >= lon_grid.shape[0] or ni < 0 or ni >= lon_grid.shape[1]:
                    continue

                if dj == 0:
                    boundary_length = (d_lat[j, i] + d_lat[nj, ni]) / 2
                else:
                    boundary_length = (d_lon[j, i] + d_lon[nj, ni]) / 2

                if dj == 0:
                    normal_vector = [0, 1] if di > 0 else [0, -1]
                else:
                    normal_vector = [1, 0] if dj > 0 else [-1, 0]

                wind_projection = u_now * normal_vector[0] + v_now * normal_vector[1]
                flux = pm25_now * pbl_now * wind_projection * boundary_length

                total_flux_x += flux * normal_vector[0]
                total_flux_y += flux * normal_vector[1]

            flux_lon[j, i] = total_flux_x / 1e9
            flux_lat[j, i] = total_flux_y / 1e9
            flux_magnitude[j, i] = sqrt(total_flux_x ** 2 + total_flux_y ** 2) / 1e9

    return flux_lon, flux_lat, flux_magnitude


def plot_daily_wind_flux(pm25_data, lon_vec, lat_vec,
                         lon_grid_lowres, lat_grid_lowres, u_avg, v_avg,
                         time_str, output_dir, city_file, pbl_height, static_status=None):
    """优化内存的每日风场通量绘图函数"""
    os.makedirs(output_dir, exist_ok=True)

    # 计算传输通量
    flux_lon, flux_lat, flux_magnitude = calculate_transport_flux(
        lon_grid_lowres, lat_grid_lowres, u_avg, v_avg, pm25_data, pbl_height
    )

    # 创建图形
    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # ===================== 核心修改2：标题设置为Times New Roman加粗 =====================
    # title = f'PM2.5 Concentration and Boundary Layer Transport Flux'
    # plt.title(title, fontsize=24, fontweight='bold', fontname='Times New Roman')

    # 1. 智能降采样PM2.5数据
    max_pixels = 2000
    if pm25_data.shape[0] > max_pixels or pm25_data.shape[1] > max_pixels:
        step = max(1, int(max(pm25_data.shape) / max_pixels))
        pm25_plot = pm25_data[::step, ::step]
        lon_plot = lon_vec[::step]
        lat_plot = lat_vec[::step]
        print(f"PM2.5数据降采样: 原始尺寸 {pm25_data.shape} -> 降采样后 {pm25_plot.shape}")
    else:
        pm25_plot = pm25_data
        lon_plot = lon_vec
        lat_plot = lat_vec

    # 2. 绘制PM2.5背景
    im = ax.pcolormesh(lon_plot, lat_plot, pm25_plot,
                       cmap=TARGET_CMAP, norm=TARGET_NORM,
                       shading='gouraud',
                       alpha=0.8,
                       transform=ccrs.PlateCarree())

    # 3. 绘制传输通量箭头
    max_flux = np.nanmax(flux_magnitude)
    skip = 2

    lon_sub = lon_grid_lowres[::skip, ::skip]
    lat_sub = lat_grid_lowres[::skip, ::skip]
    flux_lon_sub = flux_lon[::skip, ::skip]
    flux_lat_sub = flux_lat[::skip, ::skip]
    flux_magnitude_sub = flux_magnitude[::skip, ::skip]

    colors = (0.7, 0, 0)
    # 核心修改：使用全局统一的通量最大值计算比例
    if static_status and 'global_max_flux' in static_status:
        scale_factor = static_status['global_max_flux'] * 1.2 if static_status['global_max_flux'] > 0 else 1.0
    else:
        scale_factor = max_flux * 1.2 if max_flux > 0 else 1.0

    if flux_magnitude_sub.size > 0:
        q = ax.quiver(
            lon_sub, lat_sub,
            flux_lon_sub, flux_lat_sub,
            color=colors,
            width=0.0025,
            headwidth=2,
            headlength=4,
            scale=scale_factor,
            scale_units='inches',
            transform=ccrs.PlateCarree()
        )

    map_bounds = box(105, 31, 115, 41)
    # 绘制市级行政区边界
    if city_file and os.path.exists(city_file):
        target_cities = ['西安', '运城', '榆次', '宝鸡', '咸阳', '铜川', '临汾', '洛阳', '三门峡', '渭南', '离石']
        cities = load_city_boundary(city_file, target_cities=target_cities)

        if cities is not None and not cities.empty:
            name_col = [col for col in cities.columns if 'name' in col.lower() or 'NAME' in col or '市' in col][0]

            for idx, row in cities.iterrows():
                city_cn = row[name_col]
                clipped_city = gpd.clip(cities[cities[name_col] == city_cn], map_bounds)
                clipped_city.plot(
                    ax=ax,
                    edgecolor='white',
                    facecolor='none',
                    linewidth=1.2,
                    linestyle='-',
                    alpha=1.0,
                    label=city_cn,
                    transform=ccrs.PlateCarree()
                )

                centroid = clipped_city.geometry.centroid.iloc[0]
                # 城市英文名映射
                target_cities_cn = ['西安', '运城', '榆次', '宝鸡', '咸阳', '铜川', '临汾', '洛阳', '三门峡', '渭南',
                                    '离石']
                target_cities_en = ['Xian', 'Yuncheng', 'Jinzhong', 'Baoji', 'Xianyang',
                                    'Tongchuan', 'Linfen', 'Luoyang', 'Sanmenxia', 'Weinan', 'Lvliang']
                city_cn2en = dict(zip(target_cities_cn, target_cities_en))

                # 查找对应的英文名
                city_en = None
                if city_cn in city_cn2en:
                    city_en = city_cn2en[city_cn]
                else:
                    for cn_name, en_name in city_cn2en.items():
                        if cn_name in city_cn or city_cn in cn_name:
                            city_en = en_name
                            break

                if city_en is None:
                    city_en = city_cn
                    print(f"警告: 未找到城市 '{city_cn}' 的英文映射，使用中文名")

                x_offset = 0
                y_offset = 0
                if city_en == 'Luoyang': x_offset = 0.45
                if city_en == 'Sanmenxia': x_offset = -0.45
                if city_en == 'Xianyang': x_offset = -0.15
                if city_en == 'Weinan': y_offset = -0.15
                ax.text(centroid.x + x_offset, centroid.y+y_offset, city_en,
                        fontsize=27, ha='center', va='center', zorder=7,
                        fontname='Times New Roman',weight='black',color='black')
    # 添加图例元素
    if flux_magnitude_sub.size > 0 and max_flux > 0:
        legend_flux_value = 2000 / 86.4
        # 绘制风矢量的图例（箭头+文字）
        qk = ax.quiverkey(
            q, X=0.25, Y=0.95,
            U=legend_flux_value,
            label=f'{legend_flux_value * 86.4:.0f} t/d',
            labelpos='W',
            color='black',
            fontproperties={'size': 28, 'family': 'Times New Roman','weight':'black'},
            transform=ax.transAxes
        )

        # 添加半透明背景矩形（覆盖箭头和文字）
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (0.22, 0.91),  # 左下角坐标 (x, y)，轴坐标范围 0~1
            0.18, 0.08,  # 宽度，高度
            facecolor='white', alpha=0.7, edgecolor='none',
            transform=ax.transAxes, zorder=0
        )
        ax.add_patch(rect)
        qk.set_zorder(1)  # 让图例浮在矩形上面

    # ===================== 核心修改5：Colorbar字体设置为Times New Roman =====================
    cbar_pm25 = fig.colorbar(im, ax=ax, extend='max', pad=0.05)
    cbar_pm25.set_label('PM2.5 (μg/m^3)', fontsize=34, fontname='Times New Roman')

    cbar_pm25.set_ticks([0, 20, 40, 60, 80, 100, 120])
    cbar_pm25.set_ticklabels(['0', '20', '40', '60', '80', '100', '120'])

    '''
    cbar_pm25.set_ticks([0, 35, 75, 105, 140, 175, 210])
    cbar_pm25.set_ticklabels(['0', '35', '70', '105', '140', '175', '210'])
    '''

    for label in cbar_pm25.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(28)

    ax.set_xlabel('Longitude (°E)', fontsize=28, fontname='Times New Roman')
    ax.set_ylabel('Latitude (°N)', fontsize=28, fontname='Times New Roman')

    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
    gl.xlabel_style = {'fontsize': 28, 'fontname': 'Times New Roman'}
    gl.ylabel_style = {'fontsize': 28, 'fontname': 'Times New Roman'}
    gl.top_labels = False
    gl.right_labels = False

    ax.set_extent([105, 115, 31, 41], crs=ccrs.PlateCarree())

    plot_path = os.path.join(output_dir, f'transport_flux_{time_str}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"图像已保存: {plot_path}")


if __name__ == '__main__':
    start_time = time.time()

    # 区域参数
    area_name = 'prd'
    lon_begin, lon_end = 105.0, 115.0
    lat_begin, lat_end = 31.0, 41.0
    res = 0.1  # 低分辨率网格大小（度）
    print(f"使用分辨率: {res}度")

    # ===================== 新增：时区偏移设置 =====================
    UTC_OFFSET = 8  # 假设PM2.5为北京时间（UTC+8），风场为UTC时间。根据实际时区修改此值

    # 创建低分辨率网格（用于风场）
    lon_grid_lowres = np.arange(lon_begin, lon_end + 0.001, res)
    lat_grid_lowres = np.arange(lat_begin, lat_end + 0.001, res)
    lon_grid_lowres, lat_grid_lowres = np.meshgrid(lon_grid_lowres, lat_grid_lowres)
    grid_shape = lon_grid_lowres.shape

    # 时间范围
    begin = datetime.date(2023,10,10)
    end = datetime.date(2023,10,15)
    delta = datetime.timedelta(days=1)

    # 路径设置
    path = os.path.dirname(os.path.abspath(__file__))
    plot_dir = r"E:\污染案例图\new"
    os.makedirs(plot_dir, exist_ok=True)

    # 加载城市边界数据（只加载一次）
    city_file = r"D:\科研\中国ArcGIS数据(到县界、Lambert投影) 备份\Lambert\中国地州界.shp"
    target_cities = ['西安', '运城', '榆次', '宝鸡', '咸阳', '铜川', '临汾', '洛阳', '三门峡', '渭南', '离石']
    city_boundaries = load_city_boundary(city_file, target_cities=target_cities)

    global_max_pm25 = 0

    try:
        # 第一阶段：收集所有日期的通量数据，计算全局最大通量
        all_days_max_flux = []
        all_days_data = []
        print("第一阶段：收集所有日期的通量数据...")

        d = begin
        while d <= end:
            time_target = d.strftime("%Y%m%d")
            d += delta

            day_data = {
                'time_target': time_target,
                'daily_u_avg': None,
                'daily_v_avg': None,
                'daily_pm25_avg': None,
                'lon_highres': None,
                'lat_highres': None,
                'pbl_grid': None
            }

            try:
                # 读取边界层高度
                pbl_csv = pd.read_csv(
                    path + os.sep + 'pbl_height' + os.sep + f'pbl_height_{time_target}.csv',
                    index_col=False
                )

                # 初始化网格数据
                pbl_grid = np.zeros(grid_shape)
                for j in range(grid_shape[0]):
                    for i in range(grid_shape[1]):
                        try:
                            idx = round((10 * (lon_end - lon_begin) + 1) * (10 * (lat_grid_lowres[j, i] - lat_begin))) + \
                                  round(10 * (lon_grid_lowres[j, i] - lon_begin))
                            pbl_grid[j, i] = float(pbl_csv.loc[idx, 'pbl_heightchem'])
                        except:
                            pbl_grid[j, i] = np.nan

                # 初始化累加器
                daily_u = np.zeros(grid_shape)
                daily_v = np.zeros(grid_shape)
                daily_pm25_highres = None
                lon_highres = None
                lat_highres = None
                valid_hours = 0

                # ===================== 修改：小时循环中根据时区偏移计算UTC时间，用于风场文件 =====================
                for hour in range(24):
                    local_hour = hour
                    # 计算对应的UTC时间（日期和小时候）
                    utc_hour = (local_hour - UTC_OFFSET) % 24
                    utc_date = d - datetime.timedelta(days=1) if local_hour < UTC_OFFSET else d
                    utc_time_target = utc_date.strftime("%Y%m%d")
                    hour_str_utc = f"{utc_hour:02d}"

                    try:
                        # 读取风场数据（使用UTC时间）
                        wind_file = f"E:\\wind_24h\\{utc_time_target}{hour_str_utc}.nc"
                        with Dataset(wind_file, 'r') as nc:
                            u_data = nc.variables['u'][:]
                            v_data = nc.variables['v'][:]
                            heights = nc.variables['height'][:]
                            if heights.ndim == 3:
                                heights = heights[:, 0, 0]

                        # 读取PM2.5数据（使用本地时间，文件名不变）
                        pm25_file = f"E:\\pm25_1km_24h_new\\{time_target}{local_hour:02d}.nc"
                        with Dataset(pm25_file, 'r') as nc:
                            pm25_data = nc.variables['pm25'][:]
                            if lon_highres is None:
                                if 'lon' in nc.variables:
                                    lon_highres = nc.variables['lon'][:]
                                    lat_highres = nc.variables['lat'][:]
                                else:
                                    lon_highres = np.arange(pm25_data.shape[1]) * 0.01 + lon_begin
                                    lat_highres = np.arange(pm25_data.shape[0]) * 0.01 + lat_begin

                            if daily_pm25_highres is None:
                                daily_pm25_highres = np.zeros_like(pm25_data)
                            daily_pm25_highres += pm25_data

                        # 计算边界层平均风场（使用UTC风场）
                        u_avg = np.zeros(grid_shape)
                        v_avg = np.zeros(grid_shape)
                        pm25_lowres = np.zeros(grid_shape)

                        for j in range(grid_shape[0]):
                            for i in range(grid_shape[1]):
                                lat_idx = min(int(round((lat_grid_lowres[j, i] - lat_begin) / res)),
                                              u_data.shape[1] - 1)
                                lon_idx = min(int(round((lon_grid_lowres[j, i] - lon_begin) / res)),
                                              u_data.shape[2] - 1)

                                u_avg[j, i], v_avg[j, i] = calc_pbl_wind(
                                    u_data, v_data, heights, pbl_grid[j, i],
                                    lat_idx, lon_idx
                                )
                                pm25_lowres[j, i] = pm25_data[lat_idx, lon_idx]

                        # 累加小时数据
                        daily_u += u_avg
                        daily_v += v_avg
                        valid_hours += 1

                    except Exception as e:
                        print(f"\n小时数据异常: {e}")
                        continue

                # 计算日平均值
                if valid_hours > 0:
                    daily_u_avg = daily_u / valid_hours
                    daily_v_avg = daily_v / valid_hours
                    daily_pm25_avg = daily_pm25_highres / valid_hours
                    wind_speed = np.sqrt(daily_u_avg ** 2 + daily_v_avg ** 2)

                    current_max = np.nanmax(daily_pm25_avg)
                    if current_max > global_max_pm25:
                        global_max_pm25 = current_max

                    # 计算该日通量并记录最大通量
                    flux_lon, flux_lat, flux_magnitude = calculate_transport_flux(
                        lon_grid_lowres, lat_grid_lowres, daily_u_avg, daily_v_avg, daily_pm25_avg, pbl_grid
                    )
                    day_max_flux = np.nanmax(flux_magnitude)
                    all_days_max_flux.append(day_max_flux)

                    # 保存该日数据
                    day_data['daily_u_avg'] = daily_u_avg
                    day_data['daily_v_avg'] = daily_v_avg
                    day_data['daily_pm25_avg'] = daily_pm25_avg
                    day_data['lon_highres'] = lon_highres
                    day_data['lat_highres'] = lat_highres
                    day_data['pbl_grid'] = pbl_grid

                all_days_data.append(day_data)

            except Exception as e:
                print(f"\n处理 {time_target} 时出错: {e}")
                all_days_data.append(day_data)
                continue

        # 计算全局最大通量
        if all_days_max_flux:
            global_max_flux = max([f for f in all_days_max_flux if not np.isnan(f)])
            if global_max_flux <= 0:
                global_max_flux = 1.0
        else:
            global_max_flux = 1.0
        print(f"全局最大通量值: {global_max_flux}")

        # 第二阶段：使用统一比例绘制所有日期的图像
        print("\n第二阶段：绘制所有日期的图像（统一箭头比例）...")
        for day_data in all_days_data:
            time_target = day_data['time_target']
            if day_data['daily_u_avg'] is None:
                print(f"跳过 {time_target}：无有效数据")
                continue

            # 调用绘图函数并传入全局最大通量
            plot_daily_wind_flux(
                day_data['daily_pm25_avg'], day_data['lon_highres'], day_data['lat_highres'],
                lon_grid_lowres, lat_grid_lowres,
                day_data['daily_u_avg'], day_data['daily_v_avg'],
                time_target, plot_dir, city_file, day_data['pbl_grid'],
                static_status={'global_max_flux': global_max_flux}
            )
            print(f"完成 {time_target} 处理")

    except KeyboardInterrupt:
        print("\n用户中断处理，正在退出...")
    finally:
        total_time = (time.time() - start_time) / 60
        print(f"\n全部处理完成! 总耗时: {total_time:.2f} 分钟")
        print(f"PM2.5最大观测值: {global_max_pm25:.2f} μg/m³")