"""Open intercity boundaries, using native CAMS concentration times (normally 3 h)."""
from pathlib import Path
from cams_transport_common import run

# Edit these paths if necessary. Keep cams_transport_common.py in this folder.
CAMS_FILE = r'E:\PM2.5_cams_data\pm25_group_1_ug_m3.nc'
CAMS_VARIABLE = 'pm25_ug_m3'
# For the unconverted cams_group_1.nc, use CAMS_VARIABLE = 'pm2p5'.
# The variable's units attribute controls conversion, not its inherited GRIB_units.
CAMS_UTC_OFFSET = 0  # CAMS validity times are interpreted as UTC; do not set to 8.
WIND_DIR = r'E:\wind_24h'  # UTC filenames: YYYYMMDDHH.nc
PBL_DIR = r'C:\Users\zyd\PycharmProjects\pbl_height'  # daily CST CSVs
BOUNDARY_DIR = r'C:\Users\zyd\PycharmProjects\province_to_province_boundary_grid'
OUTPUT_ROOT = str(Path(__file__).resolve().parent / 'transport_open_cams_results')
START_DATE = '2023-03-01'  # inclusive, CST
END_DATE = '2024-02-29'
HEIGHT_KIND = 'thickness'  # same interpretation as original transport_24h.py
# Only change to 'layer_top_agl' if height contains layer TOP heights above ground.
# Layer-center heights cannot be substituted without defining layer interfaces.

if __name__ == '__main__':
    run(dict(CAMS_FILE=CAMS_FILE, CAMS_VARIABLE=CAMS_VARIABLE,
             CAMS_UTC_OFFSET=CAMS_UTC_OFFSET, WIND_DIR=WIND_DIR, PBL_DIR=PBL_DIR,
             BOUNDARY_DIR=BOUNDARY_DIR, OUTPUT_ROOT=OUTPUT_ROOT,
             START_DATE=START_DATE, END_DATE=END_DATE, HEIGHT_KIND=HEIGHT_KIND,
             CLOSED_BOUNDARY=False, TARGET_CITIES=None))
