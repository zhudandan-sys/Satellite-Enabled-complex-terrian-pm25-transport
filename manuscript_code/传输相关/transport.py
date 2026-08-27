import numpy as np
import os
import datetime
import time
import glob
import pandas as pd
from math import *

if __name__ == '__main__':
    start_time = time.time()

    r_earth = 6371004
    ### 选择区域
    area_name = 'prd'
    lon_begin = 105.0
    lon_end = 115.0
    lat_begin = 31.0
    lat_end = 41.0

    res = 0.1
    print(str(res))

    lon_out_1 = np.arange(lon_begin, lon_end + 0.001, res)  # dtype=np.float32
    lat_out_1 = np.arange(lat_begin, lat_end + 0.001, res)
    lon_out, lat_out = np.meshgrid(lon_out_1, lat_out_1)
    # print(lon_out, lat_out)

    begin = datetime.date(2023, 4, 1)
    end = datetime.date(2024, 3, 31)
    d = begin
    delta = datetime.timedelta(days=1)

    gas_list = ['pm25']  # , 'O3'
    # path = 'F:\\work\\传输通道\\202503_汾渭平原\\'#运行的路径
    path = os.path.dirname(os.path.abspath(__file__))

    for gasi in range(len(gas_list)):
        d = begin
        gas = gas_list[gasi]
        ######################################################### 提取训练数据 ####################################################################
        while d <= end:
            time_target = d.strftime("%Y%m%d")
            time_target2 = time_target[0:4] + '-' + time_target[4:6] + '-' + time_target[6:8]
            ymonth = time_target[0:4] + 'M' + time_target[4:6]

            d = d + delta
            try:
                pbl_height_csv = pd.read_csv(
                    path + os.sep + 'pbl_height' + os.sep + 'pbl_height_' + time_target + '.csv', index_col=False)
                ustc_vcd_csv = pd.read_csv(
                    path + os.sep + 'reconstructed_ustc_' + gas + '_area' + os.sep + 'vcd_csv_' + str(
                        res) + os.sep + 'TROPOMI_USTC_VCD_' + time_target + '.csv', index_col=False)
                u_equal = np.load(path + os.sep + 'wind_surface' + os.sep + 'u_' + time_target + '.npy')
                v_equal = np.load(path + os.sep + 'wind_surface' + os.sep + 'v_' + time_target + '.npy')
            except Exception as e:
                print(f"异常信息: {e}")
                continue
            trans_all = []
            trans_city = []

            city_boundary_file = glob.glob(
                path + os.sep + "province_to_province_boundary_grid" + os.sep + '*.csv')  ## 边界数据

            for city_boundary_path in city_boundary_file:
                boundary = pd.read_csv(city_boundary_path, index_col=0)
                lat_boundary = boundary.iloc[:, 0]
                lon_boundary = boundary.iloc[:, 1]

                trans = 0
                for i in range(len(lat_boundary) - 1):
                    lon_current = lon_boundary[i]
                    lat_current = lat_boundary[i]
                    lon_next = lon_boundary[(i + 1) % (len(lat_boundary))]
                    lat_next = lat_boundary[(i + 1) % (len(lat_boundary))]
                    # try:
                    pbl_height_now = pbl_height_csv.loc[
                        round((10 * (lon_end - lon_begin) + 1) * (10 * (lat_current - lat_begin))) + round(
                            10 * (lon_current - lon_begin))]['pbl_heightchem']
                    pbl_height_next = pbl_height_csv.loc[
                        round((10 * (lon_end - lon_begin) + 1) * (10 * (lat_next - lat_begin))) + round(
                            10 * (lon_next - lon_begin))]['pbl_heightchem']
                    ustc_vcd_now = ustc_vcd_csv.loc[
                        round((10 * (lon_end - lon_begin) + 1) * (10 * (lat_current - lat_begin))) + round(
                            10 * (lon_current - lon_begin))]['ustc_vcd']
                    ustc_vcd_next = ustc_vcd_csv.loc[
                        round((10 * (lon_end - lon_begin) + 1) * (10 * (lat_next - lat_begin))) + round(
                            10 * (lon_next - lon_begin))]['ustc_vcd']
                    ustc_vcd = (ustc_vcd_now + ustc_vcd_next) / 2

                    n_ = [2 * r_earth * pi * (lon_next - lon_current) * cos(
                        pi * (lat_next + lat_current) / (2 * 180)) / 360,
                          2 * r_earth * pi * (lat_next - lat_current) / 360]
                    n_now = [-n_[1], n_[0]]
                    v_now = [u_equal[round((lat_current - lat_begin) * 10), round(10 * (lon_current - lon_begin))],
                             v_equal[round((lat_current - lat_begin) * 10), round(10 * (lon_current - lon_begin))]]
                    v_next = [u_equal[round((lat_next - lat_begin) * 10), round(10 * (lon_next - lon_begin))],
                              v_equal[round((lat_next - lat_begin) * 10), round(10 * (lon_next - lon_begin))]]
                    trans += pbl_height_now * ustc_vcd_now * np.dot(n_now,
                                                                    v_now) / 2 + pbl_height_next * ustc_vcd_next * np.dot(
                        n_now, v_next) / 2

                trans_all.append(trans / 1e6)
                trans_city.append(os.path.splitext(os.path.split(city_boundary_path)[-1])[0])

            trans_result = pd.DataFrame({})
            # trans_result['lat'] = lat_cen
            # trans_result['lon'] = lon_cen
            trans_result['trans_city'] = trans_city
            trans_result['trans (g/s)'] = trans_all
            if not os.path.exists(path + os.sep + 'transport_flux3' + os.sep):
                os.mkdir(path + os.sep + 'transport_flux3' + os.sep)

            trans_result.to_csv(path + os.sep + 'transport_flux3' + os.sep + time_target + '.csv')
            print(time_target)

    end_time = time.time()
    print('Minute: ', (end_time - start_time) / 60)
    print('ok')