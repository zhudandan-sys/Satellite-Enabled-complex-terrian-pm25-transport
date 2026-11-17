import shapefile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import pandas as pd
import math
from pyproj import CRS
from pyproj import Transformer
import os
from tqdm import tqdm
import geopandas as gpd


# 地级市边界方向确定并转为逆时针
def change_anticlockwise(lat, lon):
    angle = 0
    for i in range(len(lat) - 2):
        vector0 = [1, 0]
        vector1 = [lon[i + 1] - lon[i], lat[i + 1] - lat[i]]
        vector2 = [lon[i + 2] - lon[i + 1], lat[i + 2] - lat[i + 1]]
        angle1 = (vector1[1] / abs(vector1[1])) * np.arccos(np.dot(vector1, vector0) / np.linalg.norm(vector1))
        angle2 = (vector2[1] / abs(vector2[1])) * np.arccos(np.dot(vector2, vector0) / np.linalg.norm(vector2))
        angle += angle2 - angle1
    if angle < 0:
        lat = lat[::-1]
        lon = lon[::-1]
    return lat, lon


# 删除海洋边界陆地部分
def delete_ocean(lat_boundary_ocean, lon_boundary_ocean, city_boun_lat_list, city_boun_lon_list):
    i_begin = 100
    i_end = 0
    for i in range(len(lon_boundary_ocean)):
        i_judge = True
        for j in range(len(city_boun_lat_list)):
            for k in range(len(city_boun_lat_list[j])):
                if abs(lat_boundary_ocean[i] - city_boun_lat_list[j][k]) < 0.01 and abs(
                        lon_boundary_ocean[i] - city_boun_lon_list[j][k]) < 0.01:
                    i_judge = False
        if i_judge:
            i_begin = min(i_begin, i)
            i_end = max(i_end, i + 1)
            print(i, i_begin, i_end)
    i_begin = max(0, i_begin - 1)
    i_end = min(i_end + 1, len(lon_boundary_ocean))
    lat_boundary_ocean = lat_boundary_ocean[i_begin:i_end]
    lon_boundary_ocean = lon_boundary_ocean[i_begin:i_end]
    lat_boundary_ocean, lon_boundary_ocean = delete(lat_boundary_ocean, lon_boundary_ocean)
    print('lat_boundary_ocean,lon_boundary_ocean:', lat_boundary_ocean, lon_boundary_ocean)
    return lat_boundary_ocean, lon_boundary_ocean


def delete(lat_boundary_ocean, lon_boundary_ocean):
    length = len(lat_boundary_ocean)
    for i in range(len(lon_boundary_ocean) - 1):
        if abs(lon_boundary_ocean[i] - lon_boundary_ocean[i + 1]) > 0.11 or abs(
                lat_boundary_ocean[i] - lat_boundary_ocean[i + 1]) > 0.11:
            if i == 0:
                del lon_boundary_ocean[i]
                del lat_boundary_ocean[i]
                break
            else:
                print('delete_i:', i)
                del lon_boundary_ocean[i + 1]
                del lat_boundary_ocean[i + 1]
                break
    if length == len(lat_boundary_ocean):
        return lat_boundary_ocean, lon_boundary_ocean
    else:
        return delete(lat_boundary_ocean, lon_boundary_ocean)


# 两市边界点排序
def reset_boundary(lat_boundary, lon_boundary):
    lat = []
    lon = []
    print('lat_boundary:', lat_boundary)
    print('lon_boundary:', lon_boundary)
    for i in range(len(lat_boundary) - 1):
        if max(abs(lat_boundary[i] - lat_boundary[i + 1]), abs(lon_boundary[i] - lon_boundary[i + 1])) > 0.11:
            print(lat_boundary[i], lon_boundary[i])
            # print(lat_boundary[0:i+1], lon_boundary[0:i+1], 0, len(lat_boundary)-i-1, len(lat_boundary))
            for j in range(i + 1, len(lat_boundary)):
                lat.append(lat_boundary[j])
                lon.append(lon_boundary[j])
            for j in range(i + 1):
                lat.append(lat_boundary[j])
                lon.append(lon_boundary[j])
            break
    if len(lat) > 0:
        print('lat:', lat)
        print('lon:', lon)
        for i in range(len(lat) - 1):
            if max(abs(lat[i] - lat[i + 1]), abs(lon[i] - lon[i + 1]) < 0.01):
                del lat[i]
                del lon[i]
                break
        return lat, lon
    else:
        return lat_boundary, lon_boundary


# 提取两市边界
def get_boundary(city1, city2):
    lat_bou = []
    lon_bou = []
    for i in range(len(city1.index)):
        for j in range(len(city2.index)):
            if abs(city1.loc[i]['lat'] - city2.loc[j]['lat']) < 0.01 and abs(
                    city1.loc[i]['lon'] - city2.loc[j]['lon']) < 0.01:
                if len(lat_bou) > 0:
                    if not lon_lat_equal((city1.loc[i]['lon'], city1.loc[i]['lat']),
                                         (lon_bou[len(lon_bou) - 1], lat_bou[len(lat_bou) - 1])):
                        lat_bou.append(city1.loc[i]['lat'])
                        lon_bou.append(city1.loc[i]['lon'])
                else:
                    lat_bou.append(city1.loc[i]['lat'])
                    lon_bou.append(city1.loc[i]['lon'])
    return lat_bou, lon_bou


def lon_lat_equal(lon_lat_1, lon_lat_2):
    if abs(lon_lat_1[0] - lon_lat_2[0]) < 0.001 and abs(lon_lat_1[1] - lon_lat_2[1]) < 0.001:
        return True
    else:
        return False


def delete_repeat_one(lon_lat):
    # lon_lat = lon_lat_in
    # print(lon_lat)
    break_judge = False
    for i in range(len(lon_lat)):
        for j in range(i + 1, len(lon_lat)):
            # 删除i到j的重复部分
            if lon_lat_equal(lon_lat[i], lon_lat[j]) and (j - i) % 2 == 0:
                # print('i:', i, 'j:', j)
                delete_judge = True
                for delta in range(int((j - i) / 2)):
                    if not lon_lat_equal(lon_lat[i + delta + 1], lon_lat[j - delta - 1]):
                        delete_judge = False
                if delete_judge:
                    print('i:', i, 'j:', j, len(lon_lat))
                    del lon_lat[i:j]
                    break_judge = True
            elif lon_lat_equal(lon_lat[i], lon_lat[j]) and j - i == 3:
                print('i:', i, 'j:', len(lon_lat))
                del lon_lat[i:j]
                break_judge = True
            # 删除j到i的重复部分
            if not break_judge:
                j_negative = j - len(lon_lat)
                length = len(lon_lat)
                if lon_lat_equal(lon_lat[j_negative], lon_lat[i]) and (i - j_negative) % 2 == 0:
                    delete_judge = True
                    for delta in range(int((i - j_negative) / 2)):
                        if not lon_lat_equal(lon_lat[j_negative + delta + 1], lon_lat[i - delta - 1]):
                            delete_judge = False
                    if delete_judge:
                        print('j_negative:', j_negative, 'i:', i, len(lon_lat))
                        del lon_lat[j:length]
                        del lon_lat[0:i]
                        print('j_negative:', j_negative, 'i:', i, len(lon_lat))
                        break_judge = True
                elif lon_lat_equal(lon_lat[j_negative], lon_lat[i]) and i - j_negative == 3:
                    print('j_negative:', j_negative, 'i:', i, len(lon_lat))
                    del lon_lat[j:length]
                    del lon_lat[0:i]
                    break_judge = True
            # 跳出循环
            if break_judge:
                break
        if break_judge:
            break
    return lon_lat


# 删除两市边界重复部分
def delete_repeat(lon_lat):
    length_old = len(lon_lat)
    lon_lat_out = delete_repeat_one(lon_lat)
    if len(lon_lat_out) == length_old:
        return lon_lat_out
    else:
        return delete_repeat(lon_lat_out)


from_crs = CRS.from_wkt(
    'PROJCS["China_Lambert_Conformal_Conic",GEOGCS["GCS_Beijing_1954",DATUM["Beijing_1954",SPHEROID["Krassowsky_1940",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",105.0],PARAMETER["Standard_Parallel_1",30.0],PARAMETER["Standard_Parallel_2",62.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
to_crs = CRS.from_epsg(4326)
transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)


def get_shp_field_list(path):
    try:
        try:
            file = shapefile.Reader(path)
        except UnicodeDecodeError:
            file = shapefile.Reader(path, encoding="gbk")
        fields = file.fields
        res = []
        for field in fields:
            res.append(field[0])

        return res

    except shapefile.ShapefileException as e:
        return repr(e)


def get_shp_shape_records(path, fields):
    try:
        try:
            file = shapefile.Reader(path)
            shape_records = file.shapeRecords()
        except UnicodeDecodeError:
            file = shapefile.Reader(path, encoding="gbk")
            shape_records = file.shapeRecords()

        shp_types = []
        shp_datas = []
        shp_fields = []
        for i in range(len(fields)):
            field_values = []
            shp_fields.append(field_values)

        for shape_record in shape_records:
            # type
            shp_type = shape_record.shape.shapeType
            # print(shape_record.shape, shp_type)
            if shp_type == 1:
                shp_types.append('point')
                points = shape_record.shape.points
                points_order = []
                points_order.append(len(points))
                for point in points:
                    points_order.append(point[0])
                    points_order.append(point[1])
                shp_datas.append(points_order)
            elif shp_type == 3:
                shp_types.append('polyline')
                points = shape_record.shape.points
                parts = shape_record.shape.parts
                polyline_record = []
                for idx in range(0, len(parts) - 1):
                    polyline_record_part = points[parts[idx]:parts[idx + 1]]
                    polyline_record.append(polyline_record_part)
                polyline_record_part = points[parts[-1]:]
                polyline_record.append(polyline_record_part)
                shp_datas.append(polyline_record)
            elif shp_type == 5:
                shp_types.append('polygon')
                points = shape_record.shape.points
                parts = shape_record.shape.parts
                # print(len(parts))
                polygon_record = []
                for idx in range(0, len(parts) - 1):
                    polygon_record_part = points[parts[idx]:parts[idx + 1]]
                    polygon_record.append(polygon_record_part)
                polygon_record_part = points[parts[-1]:]
                polygon_record.append(polygon_record_part)
                shp_datas.append(polygon_record)
            elif shp_type == 11:
                shp_types.append('pointz')
                points = shape_record.shape.points
                zs = shape_record.shape.z
                points_order = []
                points_order.append(len(points))
                for idx in range(len(points)):
                    points_order.append(points[idx][0])
                    points_order.append(points[idx][1])
                    points_order.append(zs[idx])
                shp_datas.append(points_order)
            elif shp_type == 13:
                shp_types.append('polylinez')
                points = shape_record.shape.points
                zs = shape_record.shape.z
                parts = shape_record.shape.parts
                polylinez_record = []
                for idx in range(0, len(parts) - 1):
                    polylinez_record_part = []
                    for idx_pt in range(parts[idx], parts[idx + 1]):
                        pt = list(points[idx_pt])
                        pt.append(zs[idx_pt])
                        polylinez_record_part.append(pt)
                    polylinez_record.append(polylinez_record_part)

                polylinez_record_part = []
                for idx_pt in range(parts[-1], len(points)):
                    pt = list(points[idx_pt])
                    pt.append(zs[idx_pt])
                    polylinez_record_part.append(pt)
                polylinez_record.append(polylinez_record_part)
                shp_datas.append(polylinez_record)
            elif shp_type == 15:
                shp_types.append('polygonz')
                points = shape_record.shape.points
                zs = shape_record.shape.z
                parts = shape_record.shape.parts
                polygonz_record = []
                for idx in range(0, len(parts) - 1):
                    polygonz_record_part = []
                    for idx_pt in range(parts[idx], parts[idx + 1]):
                        pt = list(points[idx_pt])
                        pt.append(zs[idx_pt])
                        polygonz_record_part.append(pt)
                    polygonz_record.append(polygonz_record_part)

                polygonz_record_part = []
                for idx_pt in range(parts[-1], len(points)):
                    pt = list(points[idx_pt])
                    pt.append(zs[idx_pt])
                    polygonz_record_part.append(pt)
                polygonz_record.append(polygonz_record_part)
                shp_datas.append(polygonz_record)
            else:
                return "undefined type"

            # field
            for idx, field in enumerate(fields):
                shp_fields[idx].append(shape_record.record[field])

        return shp_types, shp_datas, shp_fields

    except shapefile.ShapefileException as e:
        return repr(e)


city_name_chs =  ['庆阳', '平凉', '天水', '陇南','南阳','平顶山','郑州','焦作','济源','石家庄','邢台','邯郸','宝鸡','咸阳','榆次','三门峡','洛阳']
city_name_eng = ['qingyang', 'pingliang', 'tianshui', 'longnan','nanyang','pingdingshan','zhengzhou','jiaozuo','jiyuan','shijiazhuang','xingtai','handan','baoji','xianyang','yuci','sanmenxia','luoyang']######### 注意陕西和山西拼音相同！！！！！！！！！！！！！！！

############## 修改为自己对应的地图文件路径
path = r"D:\科研\中国ArcGIS数据(到县界、Lambert投影) 备份\Lambert\中国县界.shp"

# path_country = r"F:\work\广东省传输通道\code\矢量地图\矢量地图\(到县界、Lambert投影) 备份\中国ArcGIS数据(到县界、Lambert投影) 备份\Lambert\国界线.shp"    # 国界线在涉及到沿海城市的时候需要

# file_country = shapefile.Reader(path_country, encoding="gbk")
# fields_country = file_country.fields
# shape_records_country = file_country.shapeRecords()
# detail_country = get_shp_shape_records(path_country, ['BOU1_4M_'])

file = shapefile.Reader(path, encoding="gbk")
fields = file.fields
shape_records = file.shapeRecords()
# res = []
# for field in fields:
    # res.append(field[0])
detail = get_shp_shape_records(path, ['NAME'])

path = os.path.dirname(os.path.abspath(__file__))
boundary_save_path = path + os.sep + 'boundary_grid_province' + os.sep
if not os.path.exists(boundary_save_path):
    os.mkdir(boundary_save_path)
boundary_fig_save_path = path + os.sep + 'boundary_grid_province' + os.sep + 'fig' + os.sep
if not os.path.exists(boundary_fig_save_path):
    os.mkdir(boundary_fig_save_path)
#提取广州地级市边界，并转换成网格
boundary_to_boundary_save_path = path + os.sep + 'province_to_province_boundary_grid' + os.sep
if not os.path.exists(boundary_to_boundary_save_path):
    os.mkdir(boundary_to_boundary_save_path)
boundary_to_boundary_save_path_fig = path + os.sep + 'province_to_province_boundary_grid' + os.sep + 'fig' + os.sep
if not os.path.exists(boundary_to_boundary_save_path_fig):
    os.mkdir(boundary_to_boundary_save_path_fig)

#提取两市边界
cityi = 0
city_current = pd.read_csv(boundary_save_path + city_name_eng[cityi] + '_boundary.csv', index_col=0)
for cityj in range(cityi+1, len(city_name_eng)):
    city_next = pd.read_csv(boundary_save_path + city_name_eng[cityj] + '_boundary.csv', index_col=0)
    lat_boundary, lon_boundary = get_boundary(city_current, city_next)
    if len(lat_boundary)>0 and len(lon_boundary)>0:
        # lat_boundary, lon_boundary = reset_boundary(lat_boundary, lon_boundary)
        # plt.plot(lon_boundary, lat_boundary)
        boundary = pd.DataFrame({})
        boundary['lat_boundary'] = lat_boundary
        boundary['lon_boundary'] = lon_boundary
        boundary.to_csv(boundary_to_boundary_save_path + city_name_eng[cityi] + '_to_' + city_name_eng[cityj] + '.csv')
        print(boundary_to_boundary_save_path + city_name_eng[cityi] + '_to_' + city_name_eng[cityj] + '.csv')
        plt.figure(figsize=(24, 18))
        plt.plot(lon_boundary, lat_boundary)
        plt.savefig(boundary_to_boundary_save_path_fig + city_name_eng[cityi] + '_to_' + city_name_eng[cityj] + '.png')
        plt.close()