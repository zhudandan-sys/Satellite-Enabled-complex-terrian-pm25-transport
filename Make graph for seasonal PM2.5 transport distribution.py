import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
import glob
import geopandas as gpd
from shapely.geometry import box
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import CRS
from pyproj import Transformer
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from netCDF4 import Dataset, num2date
from scipy.interpolate import griddata
import datetime
import warnings
import gc
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
warnings.filterwarnings("ignore")

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# ---------- PM2.5颜色映射 ----------
PM25_MIN = 0
PM25_MAX = 78

# 定义季节对应的月份
SEASON_MONTHS = {
    '春季': [3, 4, 5],
    '夏季': [6, 7, 8],
    '秋季': [9, 10, 11],
    '冬季': [12, 1, 2]
}

# ---------- 箭头宽度映射（宽度表示通量大小，优化版）----------
WIDTH_MIN = 0.03          # 最小箭头宽度（度）- 小通量基础可视
WIDTH_MAX = 0.30          # 最大箭头宽度（度）- 扩大高值上限
FIXED_ARROW_LENGTH = 0.45  # 箭头固定长度（度）
HEAD_WIDTH_RATIO = 4.0    # 头部宽度/杆宽 比例（避免小通量头部过大）
HEAD_LENGTH_RATIO = 2.5   # 头部长度/杆宽 比例
HEAD_MIN_SIZE = 0.025     # 头部最小尺寸（保证小通量头部清晰）

def create_target_colormap():
    color_nodes = [
        (0.0, (0.2, 0.0, 0.4)),
        (13 / PM25_MAX, (0.0, 0.6, 0.6)),
        (26 / PM25_MAX, (0.0, 0.8, 0.4)),
        (39 / PM25_MAX, (1.0, 1.0, 0.2)),
        (52 / PM25_MAX, (1.0, 0.6, 0.0)),
        (65 / PM25_MAX, (1.0, 0.2, 0.0)),
        (78 / PM25_MAX, (0.8, 0.0, 0.0))
    ]
    cmap = LinearSegmentedColormap.from_list('target_pm25', color_nodes, N=256)
    norm = Normalize(vmin=PM25_MIN, vmax=PM25_MAX)
    return cmap, norm

TARGET_CMAP, TARGET_NORM = create_target_colormap()

# ---------- 辅助函数 ----------
def load_city_boundary(city_file, target_cities, target_crs='EPSG:4326'):
    try:
        if not os.path.exists(city_file):
            print(f"城市边界文件不存在: {city_file}")
            return None, None
        print(f"正在加载市级行政区文件: {city_file}")
        cities = gpd.read_file(city_file)
        name_col = None
        for col in cities.columns:
            if 'name' in col.lower() or 'NAME' in col or '市' in col:
                name_col = col
                break
        if name_col is None:
            name_col = cities.columns[0]
        if cities.crs is None:
            cities = cities.set_crs(target_crs, allow_override=True)
        else:
            cities = cities.to_crs(target_crs)
        matched_cities = []
        for city in target_cities:
            exact_match = cities[cities[name_col] == city]
            if not exact_match.empty:
                matched_cities.append(city)
            else:
                fuzzy_match = cities[cities[name_col].str.contains(city[:2], na=False)]
                if not fuzzy_match.empty:
                    matched_name = fuzzy_match[name_col].iloc[0]
                    matched_cities.append(matched_name)
        cities = cities[cities[name_col].isin(matched_cities)]
        return cities, name_col
    except Exception as e:
        print(f"市级数据加载失败: {str(e)}")
        return None, None

def get_shp_shape_records(path, fields):
    try:
        import shapefile
        encodings = "gbk"
        file = shapefile.Reader(path, encoding=encodings)
        shape_records = file.shapeRecords()
        shp_types = []
        shp_datas = []
        shp_fields = [[] for _ in fields]
        for idx, shape_record in enumerate(shape_records):
            shp_type = shape_record.shape.shapeType
            if shp_type == 5:
                shp_types.append('polygon')
                points = shape_record.shape.points
                parts = shape_record.shape.parts
                polygon_record = []
                for i in range(len(parts)):
                    start_idx = parts[i]
                    end_idx = parts[i + 1] if i < len(parts) - 1 else len(points)
                    polygon_record.append(points[start_idx:end_idx])
                shp_datas.append(polygon_record)
            else:
                shp_types.append('unknown')
                shp_datas.append(None)
            try:
                for f_idx, field in enumerate(fields):
                    shp_fields[f_idx].append(shape_record.record[field])
            except:
                for f_idx in range(len(fields)):
                    shp_fields[f_idx].append(None)
        return shp_types, shp_datas, shp_fields
    except:
        return [], [], []

def extract_coordinates_from_nc(nc_file):
    try:
        with Dataset(nc_file, 'r') as nc:
            lon = nc.variables['lon'][:]
            lat = nc.variables['lat'][:]
            if lon.ndim == 1 and lat.ndim == 1:
                lon_grid, lat_grid = np.meshgrid(lon, lat)
            else:
                lon_grid, lat_grid = lon, lat
            return lon_grid, lat_grid
    except:
        return None, None

# ---------- 优化：先原始网格平均，再插值 ----------
def compute_period_mean_pm25(season_name, lon_begin, lon_end, lat_begin, lat_end, pm25_root, target_res=0.01):
    """
    计算指定季节的PM2.5平均值：先在原始网格上平均，再插值到目标网格（一张图只插值一次）
    """
    # 生成目标网格
    lon_target = np.arange(lon_begin, lon_end + target_res/2, target_res)
    lat_target = np.arange(lat_begin, lat_end + target_res/2, target_res)
    nlon_t, nlat_t = len(lon_target), len(lat_target)

    # 获取一个示例文件以读取原始坐标和网格形状
    try:
        example_file = glob.glob(os.path.join(pm25_root, "*.nc"))[0]
        with Dataset(example_file, 'r') as nc:
            lon = nc.variables['lon'][:]
            lat = nc.variables['lat'][:]
            if lon.ndim == 1 and lat.ndim == 1:
                src_lon_2d, src_lat_2d = np.meshgrid(lon, lat)
                nlat_src, nlon_src = src_lat_2d.shape
            else:
                src_lon_2d, src_lat_2d = lon, lat
                nlat_src, nlon_src = src_lat_2d.shape
    except Exception as e:
        print(f"⚠️ {season_name}：无法获取NC文件坐标，返回空数组")
        return np.zeros((nlat_t, nlon_t)), lon_target, lat_target

    # 构建原始数据点坐标（二维网格拉平）
    points_src = np.array([src_lon_2d.ravel(), src_lat_2d.ravel()]).T

    # 创建目标网格点坐标
    lon_grid, lat_grid = np.meshgrid(lon_target, lat_target)
    points_target = np.array([lon_grid.ravel(), lat_grid.ravel()]).T

    # 获取所有NC文件
    all_files = glob.glob(os.path.join(pm25_root, "*.nc"))
    target_months = SEASON_MONTHS[season_name]
    season_files = []
    print(f"🔍 {season_name}：正在筛选{target_months}月的NC文件...")
    for f in tqdm(all_files, desc=f"筛选{season_name}文件"):
        try:
            with Dataset(f, 'r') as nc:
                # 优先从time变量读取月份
                if 'time' in nc.variables:
                    time_var = nc.variables['time']
                    dates = num2date(time_var[:], time_var.units)
                    month = dates[0].month
                    if month in target_months:
                        season_files.append(f)
                else:
                    # 备用：从文件名提取
                    fname = os.path.basename(f)
                    date_str = ''.join([c for c in fname if c.isdigit()])
                    if len(date_str) >= 6:
                        month = int(date_str[4:6])
                        if month in target_months:
                            season_files.append(f)
        except:
            continue

    if not season_files:
        print(f"⚠️ {season_name}：未筛选到对应季节文件，使用前30个文件")
        season_files = all_files[:30]
    print(f"✅ {season_name}：共筛选到{len(season_files)}个NC文件")

    # 初始化原始网格上的累加数组
    total_sum_src = np.zeros((nlat_src, nlon_src), dtype=np.float32)
    count_src = np.zeros((nlat_src, nlon_src), dtype=np.int16)
    for f in tqdm(season_files, desc=f"累加原始网格{season_name}PM2.5"):
        try:
            with Dataset(f, 'r') as nc:
                pm = nc.variables['pm25'][:]
                if pm.ndim == 3:
                    pm = pm[0]
                if pm.shape != (nlat_src, nlon_src):
                    if pm.shape == (nlon_src, nlat_src):
                        pm = pm.T
                    else:
                        continue
                valid = ~np.isnan(pm)
                total_sum_src[valid] += pm[valid]
                count_src[valid] += 1
        except Exception as e:
            continue

    # 计算原始网格上的平均场（除零处为NaN）
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_src = total_sum_src / count_src
    mean_src[count_src == 0] = np.nan

    # 将原始网格平均场插值到目标网格
    valid_src = ~np.isnan(mean_src.ravel())
    if np.any(valid_src):
        interp_values = griddata(
            points_src[valid_src],
            mean_src.ravel()[valid_src],
            points_target,
            method='linear',
            fill_value=np.nan
        )
    else:
        interp_values = np.full(points_target.shape[0], np.nan)
    mean_2d = interp_values.reshape(nlat_t, nlon_t)

    # 对仍缺失的网格进行最近邻填充
    if np.any(np.isnan(mean_2d)):
        y_idx, x_idx = np.where(~np.isnan(mean_2d))
        if len(y_idx) > 0:
            points_filled = np.array([lon_target[x_idx], lat_target[y_idx]]).T
            values_filled = mean_2d[y_idx, x_idx]
            nan_mask = np.isnan(mean_2d)
            mean_2d[nan_mask] = griddata(
                points_filled,
                values_filled,
                np.array([lon_grid[nan_mask], lat_grid[nan_mask]]).T,
                method='nearest'
            )
        else:
            mean_2d = np.zeros_like(mean_2d)
    return mean_2d, lon_target, lat_target

# ---------- 箭头绘制（优化版：立方映射放大高通量差异，头部尺寸比例化）----------
def draw_transport_channels_season(ax, season_df, city_boundary_files, special_channels,
                                   lon_begin, lon_end, lat_begin, lat_end,
                                   global_flux_min, global_flux_max):
    """
    绘制传输通道箭头
    宽度根据通量大小，采用立方映射（小通量基础可视，高通量显著加粗）
    长度固定（FIXED_ARROW_LENGTH）
    头部尺寸基于杆宽比例+最小尺寸兜底，保证小通量头部清晰
    """
    arrows = []
    arrow_params = []
    arrow_lengths = []
    fluxes = []

    special_channel_names = set()
    for g in special_channels.values():
        special_channel_names.update(g["channels"])

    flux_dict = dict(zip(season_df["trans_city"], season_df["trans(t/d)"]))
    flux = []
    for path in city_boundary_files:
        cname = os.path.splitext(os.path.basename(path))[0]
        flux.append(flux_dict.get(cname, 0))

    valid_flux = [abs(v) for v in flux if abs(v) > 1e-3]
    if not valid_flux:
        return arrows, arrow_params, arrow_lengths, fluxes, 0, 0, 0

    # 若全局范围有效，则使用全局 min/max；否则回退到本季节内归一化
    if global_flux_max > global_flux_min:
        flux_min, flux_max = global_flux_min, global_flux_max
    else:
        flux_min = np.min(valid_flux)
        flux_max = np.max(valid_flux)
    flux_mean = 250      # 或150，根据你的通量范围

    for i, boundary_path in enumerate(city_boundary_files):
        channel_name = os.path.splitext(os.path.basename(boundary_path))[0]
        if channel_name not in special_channel_names:
            continue

        try:
            bd = pd.read_csv(boundary_path, index_col=0)
            lat_bd = bd.iloc[:, 0]
            lon_bd = bd.iloc[:, 1]
        except:
            continue

        u = -(lat_bd.iloc[-1] - lat_bd.iloc[0])
        v = lon_bd.iloc[-1] - lon_bd.iloc[0]
        uv = np.hypot(u, v)
        if uv < 1e-3:
            continue

        f = flux[i]
        if abs(f) < 1e-3:
            continue
        dir_s = np.sign(f)
        u_norm = dir_s * u / uv
        v_norm = dir_s * v / uv

        lon_mid = np.mean(lon_bd)
        lat_mid = np.mean(lat_bd)

        # ---------- 宽度非线性映射（立方映射，放大高通量差异）----------
        if flux_max > flux_min:
            ratio = (abs(f) - flux_min) / (flux_max - flux_min + 1e-6)
            power = 1.0          # 立方映射（可根据效果调整为4.0）
            width = WIDTH_MIN + (WIDTH_MAX - WIDTH_MIN) * (ratio ** power)
        else:
            width = (WIDTH_MIN + WIDTH_MAX) / 2

        # 头部尺寸：比例映射 + 最小尺寸兜底，保证小通量头部清晰
        head_width = max(width * HEAD_WIDTH_RATIO, HEAD_MIN_SIZE)
        head_length = min(max(width * HEAD_LENGTH_RATIO, HEAD_MIN_SIZE), 0.30)   # 最大头部长度放宽到0.3

        arrow_length = FIXED_ARROW_LENGTH  # 长度固定

        arrow = ax.arrow(lon_mid, lat_mid,
                         u_norm * arrow_length,
                         v_norm * arrow_length,
                         head_width=head_width,
                         width=width,
                         head_length=head_length,
                         length_includes_head=True,
                         edgecolor='black',
                         facecolor='black',
                         transform=ccrs.PlateCarree(),
                         zorder=8)

        arrows.append(arrow)
        arrow_params.append({
            'start': (lon_mid, lat_mid),
            'direction': (u_norm, v_norm),
            'length': arrow_length,
            'width': width,
            'head_width': head_width,
            'head_length': head_length,
            'color': 'black',
            'flux': f
        })
        arrow_lengths.append(arrow_length)
        fluxes.append(f)

    return arrows, arrow_params, arrow_lengths, fluxes, flux_mean, flux_min, flux_max

def add_arrow_scale(ax, fluxes, flux_mean, lon_begin, lon_end, lat_begin, lat_end,
                    global_flux_min, global_flux_max):
    """
    绘制宽度比例尺箭头（示例：平均通量对应的箭头宽度）
    宽度计算采用与箭头相同的立方映射，确保比例尺与实际箭头一致
    """
    if len(fluxes) == 0:
        return

    # 计算平均通量对应的宽度（同步立方映射）
    if global_flux_max > global_flux_min:
        ratio_mean = (flux_mean - global_flux_min) / (global_flux_max - global_flux_min + 1e-6)
        power = 1.0  # 与箭头宽度幂次保持一致
        mean_width = WIDTH_MIN + (WIDTH_MAX - WIDTH_MIN) * (ratio_mean ** power)
    else:
        mean_width = (WIDTH_MIN + WIDTH_MAX) / 2

    # 比例尺箭头长度固定为 0.5 度
    scale_length = 0.5
    head_width = max(mean_width * HEAD_WIDTH_RATIO, HEAD_MIN_SIZE)
    head_length = min(max(mean_width * HEAD_LENGTH_RATIO, HEAD_MIN_SIZE), 0.30)

    x0 = lon_begin + 0.02 * (lon_end - lon_begin) + 1.8
    y0 = lat_begin + 0.08 * (lat_end - lat_begin)

    # 绘制水平箭头
    ax.arrow(x0, y0, scale_length, 0,
             head_width=head_width,
             width=mean_width,
             head_length=head_length,
             length_includes_head=True,
             color='black',
             zorder=9,
             transform=ccrs.PlateCarree())

    # 在箭头左侧标注通量值：无背景框，并预留边距避免文字超出地图边界
    text_gap = 0.12
    text_x = max(lon_begin + 0.15, x0 - text_gap)

    ax.text(
        text_x,
        y0,
        f'{flux_mean:.0f} t/d: ',
        fontsize=20,
        ha='right',
        va='center',
        clip_on=True,
        transform=ccrs.PlateCarree(),
        zorder=9
    )

# ---------- 箭头交互管理器（只移动位置，不改变大小方向）----------
class ArrowInteractiveManager:
    def __init__(self, fig, ax, arrows, params, out_path):
        self.fig = fig
        self.ax = ax
        self.arrows = arrows
        self.params = params
        self.out_path = out_path
        self.selected_idx = None
        self.selected_color = None
        self.is_dragging = False
        self.press_offset = (0, 0)
        for arr in self.arrows:
            arr.set_picker(True)
        self.cid_pick = fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.cid_motion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_key = fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_pick(self, event):
        if event.artist in self.arrows:
            if self.selected_idx is not None:
                self.arrows[self.selected_idx].set_color(self.params[self.selected_idx]['color'])
            self.selected_idx = self.arrows.index(event.artist)
            self.selected_color = 'red'
            selected_arrow = self.arrows[self.selected_idx]
            selected_arrow.set_color('red')
            mouse_x, mouse_y = event.mouseevent.xdata, event.mouseevent.ydata
            start_x, start_y = self.params[self.selected_idx]['start']
            self.press_offset = (mouse_x - start_x, mouse_y - start_y)
            self.is_dragging = True
            self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if not self.is_dragging or self.selected_idx is None or event.xdata is None or event.ydata is None:
            return
        new_start_x = event.xdata - self.press_offset[0]
        new_start_y = event.ydata - self.press_offset[1]
        self.params[self.selected_idx]['start'] = (new_start_x, new_start_y)
        self.redraw_arrow(self.selected_idx)
        self.fig.canvas.draw_idle()

    def redraw_arrow(self, idx):
        old_arrow = self.arrows[idx]
        old_arrow.remove()
        p = self.params[idx]
        start_x, start_y = p['start']
        dir_x, dir_y = p['direction']
        length = p['length']
        width = p['width']
        head_width = p['head_width']
        head_length = p['head_length']
        if idx == self.selected_idx and self.selected_color:
            color = self.selected_color
        else:
            color = p['color']
        end_x = start_x + dir_x * length
        end_y = start_y + dir_y * length
        new_arrow = self.ax.arrow(
            start_x, start_y,
            end_x - start_x, end_y - start_y,
            head_width=head_width,
            width=width,
            head_length=head_length,
            length_includes_head=True,
            edgecolor=color,
            facecolor=color,
            transform=ccrs.PlateCarree(),
            zorder=8,
            picker=True
        )
        self.arrows[idx] = new_arrow

    def on_release(self, event):
        if self.is_dragging and self.selected_idx is not None:
            self.selected_color = None
            self.redraw_arrow(self.selected_idx)
            self.is_dragging = False
            self.selected_idx = None
            self.press_offset = (0, 0)
            self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 's':
            self.fig.savefig(self.out_path, dpi=300, bbox_inches='tight')
            print(f"已保存：{self.out_path}")
            plt.close(self.fig)

# ---------- 主程序：四季绘图 ----------
if __name__ == "__main__":
    # ========== 路径设置 ==========
    excel_path = r"C:\Users\zyd\PycharmProjects\merged_trans_seasonly.xlsx"
    output_dir = r"C:\Users\zyd\Desktop"
    os.makedirs(output_dir, exist_ok=True)

    lon_begin, lon_end = 105.0, 115.0
    lat_begin, lat_end = 31.0, 41.0
    map_bounds = box(lon_begin, lat_begin, lon_end, lat_end)

    target_cities_cn = ['西安市', '运城市', '榆次市', '宝鸡市', '咸阳市', '铜川市', '临汾市', '洛阳市', '三门峡市',
                        '渭南市', '离石县']
    target_cities_en = ['Xian', 'Yuncheng', 'Jinzhong', 'Baoji', 'Xianyang',
                        'Tongchuan', 'Linfen', 'Luoyang', 'Sanmenxia', 'Weinan', 'Lvliang']
    city_cn2en = dict(zip(target_cities_cn, target_cities_en))

    city_file = r"D:\科研\中国ArcGIS数据(到县界、Lambert投影) 备份\Lambert\中国地州界.shp"
    boundary_dir = r"C:\Users\zyd\PycharmProjects\province_to_province_boundary_grid"
    city_boundary_files = glob.glob(os.path.join(boundary_dir, "*.csv"))
    pm25_root = r"E:\pm25_1km_24h_new"

    special_channels = {
        "Channel 1": {"channels": ('ankang_to_xian', 'baoji_to_xian', 'hanzhong_to_xian', 'shangzhou_to_xian',
                                   'weinan_to_xian', 'xianyang_to_xian', 'pingliang_to_baoji', 'xifeng_to_xianyang',
                                   'yanan_to_tongchuan', 'yanan_to_weinan'), "color": 'red'},
        "Channel 2": {"channels": ('yanan_to_lingfen', 'yanan_to_lishi', 'yulin_to_lishi', 'lishi_to_yuci',
                                   'shijiazhuang_to_yuci', 'xingtai_to_yuci', 'lingfen_to_changzhi',
                                   'lingfen_to_jincheng'), "color": 'orange'},
        "Channel 3": {"channels": ('tongchuan_to_xianyang', 'xianyang_to_baoji', 'weinan_to_tongchuan'),
                      "color": 'yellow'},
        "Channel 4": {"channels": ('lingfen_to_yuncheng', 'lishi_to_lingfen', 'yuci_to_lingfen'), "color": 'purple'},
        "Channel 5": {"channels": ('weinan_to_yuncheng','yuncheng_to_sanmenxia', 'luoyang_to_sanmenxia', 'pingdingshan_to_luoyang',
                                   'nanyang_to_luoyang', 'zhengzhou_to_luoyang', 'nanyang_to_sanmenxia'),
                      "color": 'pink'}
    }

    season_list = ['春季', '夏季', '秋季', '冬季']
    season_title = {'春季': 'Spring', '夏季': 'Summer', '秋季': 'Autumn', '冬季': 'Winter'}

    df_all = pd.read_excel(excel_path, sheet_name='Sheet1')

    # ---- 预计算全局通量范围，用于统一箭头尺度 ----
    all_fluxes = []
    for season in season_list:
        season_df = df_all[df_all['季节'] == season].copy()
        flux_dict = dict(zip(season_df["trans_city"], season_df["trans(t/d)"]))
        for path in city_boundary_files:
            cname = os.path.splitext(os.path.basename(path))[0]
            f = flux_dict.get(cname, 0)
            if abs(f) > 1e-3:
                all_fluxes.append(abs(f))
    if all_fluxes:
        global_flux_min = np.min(all_fluxes)
        global_flux_max = np.max(all_fluxes)
    else:
        global_flux_min = 0.0
        global_flux_max = 1.0

    # 加载城市边界（用于绘图底图，只需一次）
    cities_gdf, name_col = load_city_boundary(city_file, target_cities_cn)

    # 获取 Lambert 投影转换器（用于绘制原始边界线，可选）
    transformer = Transformer.from_crs(
        CRS.from_wkt(
            'PROJCS["China_Lambert_Conformal_Conic",GEOGCS["GCS_Beijing_1954",DATUM["Beijing_1954",SPHEROID["Krassowsky_1940",6378245.0,298.3]],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["Central_Meridian",105.0],PARAMETER["Standard_Parallel_1",30.0],PARAMETER["Standard_Parallel_2",62.0],UNIT["Meter",1.0]]'),
        CRS.from_epsg(4326), always_xy=True)
    shp_types, shp_datas, shp_fields = get_shp_shape_records(city_file, ['NAME'])

    # 四季循环绘图
    for season in season_list:
        print(f"\n===== 正在绘制：{season} =====")
        season_df = df_all[df_all['季节'] == season].copy()

        # 计算当前季节的PM2.5平均值
        pm25_mean, lon_1d, lat_1d = compute_period_mean_pm25(
            season, lon_begin, lon_end, lat_begin, lat_end, pm25_root, target_res=0.01)

        fig = plt.figure(figsize=(10, 20))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.set_facecolor('white')
        extent = [lon_begin, lon_end, lat_begin, lat_end]

        pmesh = ax.imshow(
            pm25_mean,
            extent=extent,
            origin='lower',
            cmap=TARGET_CMAP,
            norm=TARGET_NORM,
            transform=ccrs.PlateCarree(),
            zorder=1, alpha=0.7,
            interpolation='bilinear'
        )

        ax.grid(False)

        # 颜色条
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad=0.1, axes_class=plt.Axes)
        cbar = plt.colorbar(pmesh, cax=cax, extend='max')
        cbar.set_label('PM₂.₅ Concentration (μg/m³)', fontsize=26)
        cbar.set_ticks([0, 13, 26, 39, 52, 65, 78])
        cbar.set_ticklabels(['0', '13', '26', '39', '52', '65', '78'])
        cbar.ax.tick_params(labelsize=16)

        # 绘制传输箭头（使用全局尺度，宽度立方映射，长度固定）
        arrows, params, lengths, fluxes, fmean, fmin, fmax = draw_transport_channels_season(
            ax, season_df, city_boundary_files, special_channels,
            lon_begin, lon_end, lat_begin, lat_end,
            global_flux_min, global_flux_max)

        # 绘制城市边界（使用已加载的 cities_gdf，避免重复加载）
        if cities_gdf is not None and not cities_gdf.empty:
            if name_col is None:
                for col in cities_gdf.columns:
                    if 'name' in col.lower() or 'NAME' in col or '市' in col:
                        name_col = col
                        break
            for idx, row in cities_gdf.iterrows():
                city_cn = row[name_col]
                clipped_city = gpd.clip(cities_gdf[cities_gdf[name_col] == city_cn], map_bounds)
                if clipped_city.empty:
                    continue
                clipped_city.plot(
                    ax=ax, edgecolor='white', facecolor='none', linewidth=1.2,
                    linestyle='-', alpha=1.0, transform=ccrs.PlateCarree()
                )
                centroid = clipped_city.geometry.centroid.iloc[0]
                city_en = city_cn2en.get(city_cn)
                if city_en is None:
                    for cn_name, en_name in city_cn2en.items():
                        if cn_name in city_cn or city_cn in cn_name:
                            city_en = en_name
                            break
                if city_en is None:
                    city_en = city_cn
                    print(f"警告: 未找到城市 '{city_cn}' 的英文映射，使用中文名")
                x_offset = 0.7 if city_en == 'Luoyang' else 0
                y_offset = -0.15 if city_en == 'Weinan' else 0
                ax.text(centroid.x + x_offset, centroid.y + y_offset, city_en,
                        fontsize=16, ha='center', va='center', zorder=7,
                        fontname='Times New Roman')

        # 添加箭头比例尺（使用本季节平均通量对应的宽度）
        if fluxes:
            add_arrow_scale(ax, fluxes, fmean, lon_begin, lon_end, lat_begin, lat_end,
                            global_flux_min, global_flux_max)

        # 设置坐标轴
        ax.set_xticks(np.arange(lon_begin, lon_end + 1, 2), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(lat_begin, lat_end + 1, 2), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_xlabel('Longitude', fontsize=30)
        ax.set_ylabel('Latitude', fontsize=30)
        ax.tick_params(labelsize=24)
        ax.set_xlim(lon_begin, lon_end)
        ax.set_ylim(lat_begin, lat_end)
        ax.grid(ls='--', alpha=0.5, zorder=3)

        # 设置标题（仅季节名称）
        ax.set_title(f'{season_title[season]}', fontsize=30, fontweight='bold')

        out_path = os.path.join(output_dir, f"{season} average PM2.5 transport flux and concentration graph.png")
        manager = ArrowInteractiveManager(fig, ax, arrows, params, out_path)
        plt.show()

    print("\n✅ 春/夏/秋/冬 4张季节传输图全部绘制完成！")