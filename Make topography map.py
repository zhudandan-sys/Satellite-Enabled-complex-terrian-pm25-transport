import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
import glob
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import rasterio
from rasterio.merge import merge
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import CRS
from pyproj import Transformer

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


# ====================== 辅助函数 ======================
def load_city_boundary(city_file, target_cities, target_crs='EPSG:4326'):
    """加载指定城市的边界并转换到目标坐标系"""
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
            print(f"警告: 未找到明确的名称列，使用 {name_col} 作为城市名称列")
        if cities.crs is None:
            print("警告: 市级数据未定义坐标系，强制设置为EPSG:4326")
            cities = cities.set_crs(target_crs, allow_override=True)
        else:
            cities = cities.to_crs(target_crs)
        print(f"转换后坐标系: {cities.crs}")
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
                    print(f"模糊匹配: {city} -> {matched_name}")
        cities = cities[cities[name_col].isin(matched_cities)]
        print(f"找到城市: {cities[name_col].tolist()}")
        if cities.empty:
            print(f"警告: 未找到目标城市 {target_cities}")
            return None, None
        return cities, name_col
    except Exception as e:
        print(f"市级数据加载失败: {str(e)}")
        return None, None


def get_shp_shape_records(path, fields):
    """读取shapefile数据，提取几何和属性信息"""
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
            except Exception as e:
                print(f"提取记录 {idx} 的属性字段时出错: {str(e)}")
                for f_idx in range(len(fields)):
                    shp_fields[f_idx].append(None)
        return shp_types, shp_datas, shp_fields
    except Exception as e:
        print(f"读取shapefile错误: {str(e)}")
        return [], [], []


def create_terrain_background(ax, dem_folder_path, map_bounds,
                              shade_alpha=0.5,  # 山体阴影透明度
                              elev_alpha=0.7,  # 高程色带透明度
                              colorbar_max=3000,  # 色带最大值
                              shade_gamma=0.9,  # 阴影gamma校正
                              elev_stretch_power=0.7,  # 高程拉伸指数
                              azimuth=315, altitude=45):  # 光源角度
    """
    创建优化的地形背景，包括高程着色和山体阴影
    参数说明：
    - ax: 绘图轴对象
    - dem_folder_path: DEM数据文件夹路径
    - map_bounds: 地图边界（shapely box）
    - shade_alpha: 山体阴影透明度（0-1）
    - elev_alpha: 高程色带透明度（0-1）
    - colorbar_max: 高程色带最大值（避免过高值压缩色阶）
    - shade_gamma: 阴影gamma校正（>1增强暗部，<1增强亮部）
    - elev_stretch_power: 高程拉伸指数（<1增强低高程对比度，>1增强高高程）
    - azimuth/altitude: 光源方位角/高度角（度）
    """
    try:
        if not os.path.exists(dem_folder_path):
            print(f"DEM文件夹不存在: {dem_folder_path}")
            return False
        # 仅匹配HDR/ADF的栅格文件（避免非高程ADF）
        adf_files = glob.glob(os.path.join(dem_folder_path, "**/*.adf"), recursive=True)
        # 过滤掉非高程的ADF（如属性文件）
        adf_files = [f for f in adf_files if os.path.basename(f) in ['w001001.adf', 'hdr.adf'] or 'dem' in f.lower()]
        if not adf_files:
            print(f"在 {dem_folder_path} 中未找到有效的DEM ADF文件")
            return False
        print(f"找到 {len(adf_files)} 个有效ADF文件")

        # 批量打开栅格并合并
        src_files = []
        for f in adf_files:
            try:
                src = rasterio.open(f)
                src_files.append(src)
            except:
                print(f"跳过无效栅格文件: {f}")
        if not src_files:
            print("无可用的DEM栅格文件")
            return False

        merged_data, out_transform = merge(src_files)
        for src in src_files:
            src.close()

        # 预处理高程数据
        elevation = merged_data[0].astype(np.float32)
        # 过滤异常值（更严格的范围）
        elevation[(elevation > 9000) | (elevation < -500) | np.isinf(elevation)] = np.nan
        valid_elev = elevation[np.isfinite(elevation)]
        if valid_elev.size == 0:
            print("没有有效的高程数据")
            return False

        # 计算高程统计
        elevation_min = np.min(valid_elev)
        elevation_max = np.max(valid_elev)
        elevation_mean = np.mean(valid_elev)
        elevation_range = elevation_max - elevation_min
        print(f"高程统计 - 最小值: {elevation_min:.1f}m, 最大值: {elevation_max:.1f}m, "
              f"均值: {elevation_mean:.1f}m, 高差: {elevation_range:.1f}m")

        # 自适应高程拉伸（根据高差动态调整）
        if elevation_range < 1000:
            # 低高差区域增强对比度
            elev_stretch_power = 0.5
        elif elevation_range > 5000:
            # 高高差区域降低拉伸
            elev_stretch_power = 0.8
        elevation_normalized = (elevation - elevation_min) / elevation_range
        elevation_normalized = np.clip(elevation_normalized, 0, 1)  # 防止越界
        elevation_enhanced = np.power(elevation_normalized, elev_stretch_power) * elevation_range + elevation_min

        # 计算山体阴影（优化版）
        dx = out_transform.a
        dy = out_transform.e  # 修复原代码dy取值错误（原代码用了out_transform.a，应为e）
        # 梯度计算（添加小值避免除以0）
        grad_x, grad_y = np.gradient(elevation_enhanced, dx, dy)
        slope = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8))
        aspect = np.arctan2(grad_y, -grad_x)

        # 光源角度转换为弧度
        azimuth_rad = np.radians(360.0 - azimuth)
        altitude_rad = np.radians(altitude)

        # 计算阴影强度（物理模型优化）
        shaded = (np.sin(altitude_rad) * np.sin(slope) +
                  np.cos(altitude_rad) * np.cos(slope) *
                  np.cos(azimuth_rad - aspect))
        # 归一化阴影到0-1
        shaded = (shaded - np.min(shaded[np.isfinite(shaded)])) / \
                 (np.max(shaded[np.isfinite(shaded)]) - np.min(shaded[np.isfinite(shaded)]))
        # Gamma校正增强阴影质感
        shaded = np.power(shaded, shade_gamma)
        shaded = np.nan_to_num(shaded, nan=0.0)
        shaded = np.clip(255 * shaded, 0, 255).astype(np.uint8)

        # 获取栅格边界
        bounds = rasterio.transform.array_bounds(elevation.shape[0], elevation.shape[1], out_transform)
        x_min, y_min, x_max, y_max = bounds

        # 优化的地形配色（更自然的渐变，适配东亚地形）
        terrain_colors = [
            (0.0, '#0066CC'),  # 深蓝（低海拔/水域）
            (0.1, '#3399CC'),  # 浅蓝（近海/低地）
            (0.2, '#66CC99'),  # 浅绿（平原）
            (0.3, '#99CC66'),  # 黄绿（丘陵）
            (0.4, '#CCCC66'),  # 浅黄（台地）
            (0.5, '#E6CC66'),  # 深黄（低山）
            (0.6, '#E69966'),  # 橙黄（中山）
            (0.7, '#E66633'),  # 橙红（高山）
            (0.8, '#CC6633'),  # 棕红（极高山区）
            (0.9, '#996633'),  # 深棕（裸岩）
            (1.0, '#FFFFFF')  # 白色（雪线以上）
        ]
        # 创建色带（增加色阶数）
        terrain_cmap = LinearSegmentedColormap.from_list('natural_terrain', terrain_colors, N=1024)

        # 绘制山体阴影（底层）
        ax.imshow(shaded, cmap='gray', extent=[x_min, x_max, y_min, y_max],
                  alpha=shade_alpha, transform=ccrs.PlateCarree(), zorder=1)

        # 绘制高程色带（上层）
        vmax_value = min(elevation_max, colorbar_max)
        im = ax.imshow(elevation_enhanced, cmap=terrain_cmap, extent=[x_min, x_max, y_min, y_max],
                       alpha=elev_alpha, vmin=elevation_min, vmax=vmax_value,
                       transform=ccrs.PlateCarree(), zorder=2)

        # 优化色带样式
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
        norm = mcolors.Normalize(vmin=elevation_min, vmax=vmax_value)
        sm = plt.cm.ScalarMappable(cmap=terrain_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        # 色带标签优化
        cbar.set_label('Elevation (m)', fontsize=40, fontname='Times New Roman', labelpad=20)
        cbar.ax.tick_params(labelsize=30, length=8, width=2)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        # 色带边框美化
        cax.spines['top'].set_linewidth(2)
        cax.spines['right'].set_linewidth(2)
        cax.spines['bottom'].set_linewidth(2)
        cax.spines['left'].set_linewidth(2)

        return True
    except Exception as e:
        print(f"创建地形背景时出错: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细错误栈
        return False


def draw_region_boundaries(ax, cities_gdf, name_col, regions, transformer, map_bounds, linewidth=3, alpha=0.9):
    """绘制区域的外部边界（合并区域内所有城市的边界），不同区域使用不同颜色"""
    region_boundaries = {}

    # 定义不同区域的颜色
    region_colors = {
        "Fenhe": "black",
        "Weihe": "white",
        "Yuncheng-Sanmenxia-Luoyang": "gray"
    }

    for region_name, cities_in_region in regions.items():
        print(f"正在处理区域: {region_name}, 包含城市: {cities_in_region}")

        # 获取该区域对应的颜色
        region_color = region_colors.get(region_name, "black")

        # 获取该区域内所有城市的几何对象
        region_geometries = []
        for city_name in cities_in_region:
            city_data = cities_gdf[cities_gdf[name_col].str.contains(city_name, na=False)]
            if not city_data.empty:
                region_geometries.extend(city_data.geometry.values)
            else:
                print(f"  警告: 未找到城市 {city_name}")

        if not region_geometries:
            print(f"  错误: 区域 {region_name} 没有有效的几何数据")
            continue

        # 合并区域内所有城市的几何对象
        merged_region = unary_union(region_geometries)

        # 如果合并后的几何是 MultiPolygon，取合并后的外边界
        if merged_region.geom_type == 'MultiPolygon':
            # 获取外部边界（凸包或合并后的外边界）
            external_boundary = merged_region.convex_hull
        else:
            external_boundary = merged_region

        # 转换坐标系并绘制
        try:
            # 转换为WGS84坐标
            if cities_gdf.crs != 'EPSG:4326':
                external_boundary_wgs84 = \
                    gpd.GeoSeries([external_boundary], crs=cities_gdf.crs).to_crs('EPSG:4326').iloc[0]
            else:
                external_boundary_wgs84 = external_boundary

            # 绘制边界（使用区域特定颜色）
            if external_boundary_wgs84.geom_type == 'Polygon':
                x, y = external_boundary_wgs84.exterior.xy
                ax.plot(x, y, color=region_color, linewidth=linewidth, alpha=alpha,
                        linestyle='-', zorder=10, label=region_name)
            elif external_boundary_wgs84.geom_type == 'MultiPolygon':
                for poly in external_boundary_wgs84.geoms:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color=region_color, linewidth=linewidth, alpha=alpha,
                            linestyle='-', zorder=10)

            region_boundaries[region_name] = external_boundary_wgs84
            print(f"  成功绘制区域 {region_name} 的边界（颜色: {region_color}）")

        except Exception as e:
            print(f"  绘制区域 {region_name} 时出错: {str(e)}")

    return region_boundaries


# ====================== 主程序 ======================
if __name__ == "__main__":
    # ========== 请根据实际情况修改以下路径 ==========
    output_dir = r"C:\Users\zyd\Desktop\科研\time_period\plt"
    os.makedirs(output_dir, exist_ok=True)

    lon_begin, lon_end = 106.0, 114.2
    lat_begin, lat_end = 33.0, 39.0
    map_bounds = box(lon_begin, lat_begin, lon_end, lat_end)

    # 定义三个区域及其包含的城市
    REGIONS = {
        "Fenhe": ["榆次", "离石", "临汾"],
        "Weihe": ["西安", "咸阳", "宝鸡", "渭南", "铜川"],
        "Yuncheng-Sanmenxia-Luoyang": ["洛阳", "三门峡", "运城"]
    }

    # 收集所有需要加载的城市
    all_target_cities = set()
    for cities in REGIONS.values():
        all_target_cities.update(cities)
    target_cities_cn = list(all_target_cities)
    print(f"需要加载的城市: {target_cities_cn}")

    city_file = r"D:\科研\中国ArcGIS数据(到县界、Lambert投影) 备份\Lambert\中国地州界.shp"
    dem_folder_path = r"D:\科研\dem_1km\dem_1km_"
    # =============================================

    # 加载城市边界
    cities_gdf, name_col = load_city_boundary(city_file, target_cities_cn)

    if cities_gdf is None or cities_gdf.empty:
        print("错误: 无法加载城市边界数据")
        exit(1)

    # 坐标转换（Lambert -> WGS84）
    from_crs = CRS.from_wkt(
        'PROJCS["China_Lambert_Conformal_Conic",GEOGCS["GCS_Beijing_1954",DATUM["Beijing_1954",SPHEROID["Krassowsky_1940",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",105.0],PARAMETER["Standard_Parallel_1",30.0],PARAMETER["Standard_Parallel_2",62.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
    to_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)

    shp_types, shp_datas, shp_fields = get_shp_shape_records(city_file, ['NAME'])

    # 创建图形
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # 添加基础地理要素
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=2, zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3, zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.3, zorder=0)
    try:
        provinces = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                                 scale='50m', edgecolor='gray', facecolor='none')
        ax.add_feature(provinces, linewidth=1, linestyle='--', alpha=0.7, zorder=1)
    except:
        print("无法加载NaturalEarth省界数据")

    # 地形背景（传入自定义参数）
    create_terrain_background(
        ax, dem_folder_path, map_bounds,
        shade_alpha=0.45,  # 山体阴影透明度
        elev_alpha=0.85,  # 高程色带透明度
        colorbar_max=3500,  # 色带最大值适配黄土高原/关中地形
        shade_gamma=0.8,  # 增强阴影亮部
        elev_stretch_power=0.6,  # 增强低高程（平原/丘陵）对比度
        azimuth=300, altitude=40  # 调整光源角度更贴合实际光照
    )

    # 绘制区域外部边界（不同颜色）
    region_boundaries = draw_region_boundaries(ax, cities_gdf, name_col, REGIONS, transformer, map_bounds,
                                               linewidth=3, alpha=0.9)

    # 坐标轴设置
    ax.set_xticks(np.arange(lon_begin, lon_end + 1, 2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat_begin, lat_end + 1, 2), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xlabel('Longitude', fontsize=30, fontname='Times New Roman')
    ax.set_ylabel('Latitude', fontsize=30, fontname='Times New Roman')
    ax.tick_params(labelsize=24)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    ax.set_xlim(lon_begin, lon_end)
    ax.set_ylim(lat_begin, lat_end)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=3)

    # 保存图片（建议开启，高清输出）
    plt.savefig(os.path.join(output_dir, 'optimized_terrain_map.png'),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

    plt.show()