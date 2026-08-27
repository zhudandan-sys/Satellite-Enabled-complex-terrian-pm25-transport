# -*- coding: utf-8 -*-
"""按照式(7)和式(8)计算各城市净传输通量与ASI的一阶合成不确定度。"""

import math
from pathlib import Path

import pandas as pd


# ============================== 输入均值 ==============================
rho = 33.9698855       # PM2.5浓度，μg/m³
H_pbl = 1216.027832    # 边界层高度，m
u_abs = 6.044755936    # PBL内平均风速模长，m/s

# ============================== 输入不确定度 ==============================
delta_rho = 6.53       # PM2.5浓度平均绝对偏差，μg/m³
delta_H = 544.13        # ERA5 PBLH平均绝对偏差，m
delta_R_day = 1.48     # 降水平均绝对偏差，mm/day
delta_R = delta_R_day / 24.0  # 式(8)中的ΔR，mm/h

# ============================== ASI常数 ==============================
C_s = 0.075            # PM2.5浓度限值，mg/m³
W_r = 6.0e5            # 雨洗常数（沿用原公式）
k = math.sqrt(math.pi) / 2.0
R_EARTH = 6371004.0     # 与transport_city.py一致，m
BOUNDARY_DIR = Path(r"C:\Users\zyd\PycharmProjects\boundary_grid_province")

# 城市面积，km²
city_area_km2 = {
    "baoji": 17697.49,
    "lingfen": 19395.20,
    "lvliang": 19999.44,
    "luoyang": 14440.44,
    "sanmenxia": 9454.23,
    "tongchuan": 3710.02,
    "weinan": 12304.15,
    "xian": 9848.00,
    "xianyang": 10025.70,
    "jinzhong": 15500.00,
    "yuncheng": 13777.23,
}

# 面积字典中的城市名与transport_city.py边界文件名之间的对应关系。
city_boundary_key = {
    "baoji": "baoji",
    "lingfen": "linfen",
    "lvliang": "lishi",
    "luoyang": "luoyang",
    "sanmenxia": "sanmenxia",
    "tongchuan": "tongchuan",
    "weinan": "weinan",
    "xian": "xian",
    "xianyang": "xianyang",
    "jinzhong": "yuci",
    "yuncheng": "yuncheng",
}

# 文献给出的有符号风速平均偏差（模拟值－观测值），m/s。
# 式(7)/(8)计算的是合成不确定度，所以计算时使用其绝对值。
wind_signed_bias = {
    "jinzhong": 0.16,
    "lvliang": 0.12,
    "lingfen": -0.55,
    "yuncheng": 0.90,
    "xian": 0.56,
    "baoji": -0.12,
    "xianyang": -0.05,
    "weinan": 0.89,
    "tongchuan": 0.35,
    "luoyang": 0.57,
    "sanmenxia": 1.07,
}

UG_TO_MG = 1.0e-3
SECONDS_PER_HOUR = 3600.0
MM_TO_M = 1.0e-3


def calculate_boundary_length(boundary_file):
    """按transport_city.py的经纬度线段算法计算闭合城市边界长度（m）。"""
    df = pd.read_csv(boundary_file, index_col=0)
    if df.shape[1] < 2:
        raise ValueError(f"边界文件至少应包含lat和lon两列：{boundary_file}")

    lat = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    lon = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    valid = lat.notna() & lon.notna()
    points = list(zip(lon[valid].astype(float), lat[valid].astype(float)))
    if len(points) < 3:
        raise ValueError(f"边界有效点少于3个：{boundary_file}")

    boundary_length = 0.0
    for i, (lon_cur, lat_cur) in enumerate(points):
        lon_next, lat_next = points[(i + 1) % len(points)]
        dlon = lon_next - lon_cur
        dlat = lat_next - lat_cur
        avg_lat = (lat_next + lat_cur) / 2.0
        dx = (
            2.0 * R_EARTH * math.pi * dlon
            * math.cos(math.radians(avg_lat)) / 360.0
        )
        dy = 2.0 * R_EARTH * math.pi * dlat / 360.0
        boundary_length += math.hypot(dx, dy)
    return boundary_length


def calculate_t_line_uncertainty(delta_u):
    """
    单位边界长度上的独立输入误差一阶合成不确定度（RSS）：
    error_T = sqrt([(H|u|)Δρ]² + [(ρ|u|)ΔH]² + [(ρH)Δ|u|]²)

    各分项单位为 μg/(m·s)。随后按照式(10)沿城市闭合边界积分，
    再除以城市面积，转换为 mg/(m²·h)。
    """
    term_rho = H_pbl * u_abs * delta_rho
    term_H = rho * u_abs * delta_H
    term_u = rho * H_pbl * delta_u
    total = math.sqrt(term_rho ** 2 + term_H ** 2 + term_u ** 2)
    return term_rho, term_H, term_u, total


def calculate_asi_uncertainty(area_km2, delta_u):
    """
    按图片公式进行一阶线性加和：
    ΔASI = Cs*k*|u|/sqrt(S)*ΔH
         + Cs*k*H/sqrt(S)*Δ|u|
         + Cs*Wr*ΔR

    Wr按无量纲处理。前两项的原始单位为mg/(m²·s)，乘3600后
    转为mg/(m²·h)。降水项将mm换算为m后，单位为mg/(m²·h)。
    """
    area_m2 = area_km2 * 1.0e6
    sqrt_area = math.sqrt(area_m2)
    term_H = (
        C_s * k * u_abs / sqrt_area * delta_H
        * SECONDS_PER_HOUR
    )
    term_u = (
        C_s * k * H_pbl / sqrt_area * delta_u
        * SECONDS_PER_HOUR
    )
    term_R = C_s * W_r * delta_R * MM_TO_M
    total = term_H + term_u + term_R
    return term_H, term_u, term_R, total


def main():
    rows = []
    t_contribution_rows = []
    asi_contribution_rows = []

    for city, area in city_area_km2.items():
        signed_du = wind_signed_bias[city]
        delta_u = abs(signed_du)
        area_m2 = area * 1.0e6

        boundary_file = (
            BOUNDARY_DIR / f"{city_boundary_key[city]}_boundary.csv"
        )
        if not boundary_file.exists():
            raise FileNotFoundError(f"缺少城市边界文件：{boundary_file}")
        boundary_length_m = calculate_boundary_length(boundary_file)

        t_rho_line, t_H_line, t_u_line, delta_T_line = (
            calculate_t_line_uncertainty(delta_u)
        )

        # error_Fnet = ∫city_boundary error_T ds。
        # 当前均值参数使error_T沿边界为常数，因此积分等于error_T×边界长度。
        # μg/s ÷ m² × 3600 s/h × 1e-3 mg/μg = mg/(m²·h)。
        t_city_scale = (
            boundary_length_m / area_m2
            * SECONDS_PER_HOUR * UG_TO_MG
        )
        t_rho = t_rho_line * t_city_scale
        t_H = t_H_line * t_city_scale
        t_u = t_u_line * t_city_scale
        delta_T_net = delta_T_line * t_city_scale

        asi_H, asi_u, asi_R, delta_ASI = calculate_asi_uncertainty(area, delta_u)

        rows.append({
            "城市": city,
            "城市面积(km²)": area,
            "城市边界长度(km)": boundary_length_m / 1000.0,
            "有符号风速偏差(m/s)": signed_du,
            "采用的Δ|u|(m/s)": delta_u,
            "净传输通量偏差ΔT_net(mg·m⁻²·h⁻¹)": delta_T_net,
            "ΔASI(mg·m⁻²·h⁻¹)": delta_ASI,
        })

        t_contribution_rows.append({
            "城市": city,
            "城市面积(km²)": area,
            "城市边界长度(km)": boundary_length_m / 1000.0,
            "浓度不确定度贡献(mg·m⁻²·h⁻¹)": t_rho,
            "PBLH不确定度贡献(mg·m⁻²·h⁻¹)": t_H,
            "风速不确定度贡献(mg·m⁻²·h⁻¹)": t_u,
            "净传输通量偏差RSS(mg·m⁻²·h⁻¹)": delta_T_net,
        })

        asi_contribution_rows.append({
            "城市": city,
            "PBLH贡献(mg·m⁻²·h⁻¹)": asi_H,
            "风速贡献(mg·m⁻²·h⁻¹)": asi_u,
            "降水贡献(mg·m⁻²·h⁻¹)": asi_R,
            "ΔASI线性加和(mg·m⁻²·h⁻¹)": delta_ASI,
        })

    result = pd.DataFrame(rows)
    t_contributions = pd.DataFrame(t_contribution_rows)
    asi_contributions = pd.DataFrame(asi_contribution_rows)

    parameters = pd.DataFrame([
        {"参数": "rho", "数值": rho, "单位": "μg/m³"},
        {"参数": "H_PBL", "数值": H_pbl, "单位": "m"},
        {"参数": "|u|", "数值": u_abs, "单位": "m/s"},
        {"参数": "Δrho", "数值": delta_rho, "单位": "μg/m³"},
        {"参数": "ΔH_PBL", "数值": delta_H, "单位": "m"},
        {"参数": "ΔR", "数值": delta_R, "单位": "mm/h"},
        {"参数": "Cs", "数值": C_s, "单位": "mg/m³"},
        {"参数": "Wr", "数值": W_r, "单位": "无量纲"},
    ])

    notes = pd.DataFrame({"说明": [
        "T按相互独立输入误差处理，采用各分项平方和开根号（RSS）合成。",
        "ASI严格按图片公式处理，将PBLH、风速和降水三个非负误差贡献直接相加。",
        "风速文献数据是有符号平均偏差；作为不确定度输入时取绝对值。",
        "净传输偏差按error_Fnet=∫city boundary error_T ds计算。",
        "城市边界长度采用与transport_city.py相同的经纬度线段算法，并按闭合边界积分。",
        "城市净传输偏差除以城市面积后，统一输出为mg/(m²·h)。",
        "式(8)的ASI偏差及其分项统一输出为mg/(m²·h)。",
        "Wr按无量纲处理；降水ΔR由mm/h乘0.001换算为m/h。",
        "净传输偏差是RSS不确定度幅度，不带净输入或净输出的正负方向。",
    ]})

    output_file = r"C:\Users\zyd\PycharmProjects\按公式7_8计算ASI_T偏差.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="汇总", index=False)
        t_contributions.to_excel(writer, sheet_name="T分项贡献", index=False)
        asi_contributions.to_excel(writer, sheet_name="ASI分项贡献", index=False)
        parameters.to_excel(writer, sheet_name="输入参数", index=False)
        notes.to_excel(writer, sheet_name="单位与说明", index=False)

    print(f"计算完成，共处理 {len(result)} 个城市。")
    print(f"结果文件：{output_file}")


if __name__ == "__main__":
    main()
