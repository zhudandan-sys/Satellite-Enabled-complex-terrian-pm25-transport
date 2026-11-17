"""Closed city boundary net flux, positive outward, at native CAMS times."""
from pathlib import Path
from cams_transport_common import run

CAMS_FILE = r'E:\PM2.5_cams_data\pm25_group_1_ug_m3.nc'
CAMS_VARIABLE = 'pm25_ug_m3'
CAMS_UTC_OFFSET = 0
WIND_DIR = r'E:\wind_24h'
PBL_DIR = r'C:\Users\zyd\PycharmProjects\pbl_height'
BOUNDARY_DIR = r'C:\Users\zyd\PycharmProjects\boundary_grid_province'
OUTPUT_ROOT = str(Path(__file__).resolve().parent / 'transport_city_cams_results')
START_DATE = '2023-03-01'  # use '2023-12-27' to reproduce original city time window
END_DATE = '2024-02-29'
HEIGHT_KIND = 'thickness'  # now consistent with transport_24h_cams.py
TARGET_CITIES = ['xian','yuncheng','lishi','baoji','xianyang','tongchuan',
                 'linfen','luoyang','sanmenxia','weinan','yuci']

if __name__ == '__main__':
    run(dict(CAMS_FILE=CAMS_FILE, CAMS_VARIABLE=CAMS_VARIABLE,
             CAMS_UTC_OFFSET=CAMS_UTC_OFFSET, WIND_DIR=WIND_DIR, PBL_DIR=PBL_DIR,
             BOUNDARY_DIR=BOUNDARY_DIR, OUTPUT_ROOT=OUTPUT_ROOT,
             START_DATE=START_DATE, END_DATE=END_DATE, HEIGHT_KIND=HEIGHT_KIND,
             CLOSED_BOUNDARY=True, TARGET_CITIES=TARGET_CITIES))
