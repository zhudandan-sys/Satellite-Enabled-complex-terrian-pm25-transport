"""
pm25_validation_v6_optimized.py
PM2.5监测数据与国控站点数据验证（高性能版）
适配：逐小时站点数据（type=PM2.5 为小时均值，PM2.5_24h 为24小时滑动均值）
优化点：批量IO、多进程并行、规则网格插值、数据预过滤
时区说明：站点数据=北京时间(UTC+8)，监测数据=北京时间（已统一）
修改说明：
  1. 移除所有调试代码，简化输出。
  2. 监测数据直接使用北京时间匹配，无需时区转换。
  3. 绘图模块升级：全局新罗马字体、左上角添加平均偏差/RMSE/MFB/MFE、优化图表样式。
  4. 插值加速：利用规则网格特性，使用 interpn 替代 griddata，大幅提升性能。
  5. 散点着色：基于站点真实浓度（低白高红），更直观展示浓度分布。
  6. 绘图优化：大数据量自动采样、支持hexbin图、异常回退机制。
  7. 修改绘图部分为数据验证2的散点密度图+残差分析图。
  8. 散点图颜色映射改为黄-橙-红，右上角增加 MB/RMSE/MFB/MFE 指标。
  9. 新增：绘制散点图时剔除站点数据（观测值）大于500的数据点。
"""
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import xarray as xr
import os
from scipy import stats
from scipy.interpolate import griddata, interpn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from tqdm import tqdm
import warnings
import traceback
from multiprocessing import Pool, cpu_count
from functools import partial
import time
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# 全局设置新罗马字体（兼容Windows/Linux/Mac）
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'   # 公式字体匹配新罗马


class PM25MonitorValidator:
    """高性能版PM2.5监测数据验证类（站点数据=北京时间，监测数据=北京时间）"""
    def __init__(self, monitor_data_dir, site_data_dir, station_info_path,
                 start_date='20230301', end_date='20240229', n_workers=None,
                 interp_method='cubic', site_data_type='PM2.5'):
        """
        参数:
            monitor_data_dir: 监测数据目录（北京时间，nc文件）
            site_data_dir: 站点数据目录（北京时间，csv文件）
            station_info_path: 站点信息文件路径（excel或csv）
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            n_workers: 并行进程数，默认CPU核数-1
            interp_method: 插值方法，可选 'linear', 'nearest', 'cubic'
            site_data_type: 站点数据类型，可选 'PM2.5'（小时均值）或 'PM2.5_24h'（24小时滑动均值）
        """
        self.monitor_data_dir = monitor_data_dir
        self.site_data_dir = site_data_dir
        self.station_info_path = station_info_path
        self.start_date = start_date
        self.end_date = end_date
        self.interp_method = interp_method
        self.site_data_type = site_data_type
        # 并行配置
        self.n_workers = n_workers if n_workers else max(1, cpu_count() - 1)
        # 缓存
        self.station_info = None
        self.site_data_batch_cache = {}
        self.grid_cache = {}               # 保留作为回退（非规则网格）
        self.grid_lon = None               # 规则网格经度坐标（1D）
        self.grid_lat = None               # 规则网格纬度坐标（1D）
        self.validation_results = {}
        self.comparison_data = {}
        self._preload_station_info()

    def _preload_station_info(self):
        """预加载并过滤站点信息"""
        try:
            if self.station_info_path.endswith('.csv'):
                station_df = pd.read_csv(self.station_info_path, encoding='utf-8')
            else:
                station_df = pd.read_excel(self.station_info_path)
            # 列名匹配
            column_mapping = {}
            id_cols = ['监测点编码']
            lon_cols = ['经度']
            lat_cols = ['纬度']
            for col in id_cols:
                if col in station_df.columns:
                    column_mapping['station_id'] = col
                    break
            for col in lon_cols:
                if col in station_df.columns:
                    column_mapping['lon'] = col
                    break
            for col in lat_cols:
                if col in station_df.columns:
                    column_mapping['lat'] = col
                    break
            if 'station_id' not in column_mapping or 'lon' not in column_mapping or 'lat' not in column_mapping:
                raise ValueError("缺失必要列：station_id/lon/lat")
            processed_df = station_df[[column_mapping['station_id'], column_mapping['lon'], column_mapping['lat']]].copy()
            processed_df.columns = ['station_id', 'lon', 'lat']
            processed_df['station_id'] = processed_df['station_id'].astype(str).str.strip()
            processed_df['lon'] = pd.to_numeric(processed_df['lon'], errors='coerce')
            processed_df['lat'] = pd.to_numeric(processed_df['lat'], errors='coerce')
            # 地理范围过滤
            lon_mask = (processed_df['lon'] >= 105) & (processed_df['lon'] <= 115)
            lat_mask = (processed_df['lat'] >= 31) & (processed_df['lat'] <= 41)
            valid_mask = processed_df['lon'].notna() & processed_df['lat'].notna() & lon_mask & lat_mask
            processed_df = processed_df[valid_mask].reset_index(drop=True)
            # 数字部分映射
            def extract_num(s):
                return ''.join(filter(str.isdigit, str(s))) if pd.notna(s) else ''
            processed_df['station_num'] = processed_df['station_id'].apply(extract_num)
            self.station_info = {
                'df': processed_df,
                'points': processed_df[['lon', 'lat']].values,
                'ids': processed_df['station_id'].tolist(),
                'nums': processed_df['station_num'].tolist(),
                'count': len(processed_df)
            }
            print(f"预加载站点完成：{self.station_info['count']} 个有效站点")
        except Exception as e:
            print(f"加载站点信息错误: {e}")
            traceback.print_exc()
            self.station_info = None

    def _batch_load_site_data(self, date_list):
        """批量加载站点数据（北京时间），根据 self.site_data_type 筛选"""
        start_time = time.time()
        missing_dates = []
        for date_str in date_list:
            if date_str in self.site_data_batch_cache:
                continue
            site_file = os.path.join(self.site_data_dir, f"china_sites_{date_str}.csv")
            if not os.path.exists(site_file):
                missing_dates.append(date_str)
                self.site_data_batch_cache[date_str] = None
                continue
            try:
                site_df = pd.read_csv(site_file, encoding='utf-8')
            except:
                try:
                    site_df = pd.read_csv(site_file, encoding='gbk')
                except:
                    site_df = None
            if site_df is None:
                self.site_data_batch_cache[date_str] = None
                continue
            if 'hour' in site_df.columns:
                site_df['hour'] = pd.to_numeric(site_df['hour'], errors='coerce')
            if 'type' in site_df.columns:
                site_df['type'] = site_df['type'].astype(str).str.strip()
            # 根据用户指定的数据类型筛选
            type_mask = site_df['type'].isin([self.site_data_type])
            self.site_data_batch_cache[date_str] = site_df[type_mask].copy() if type_mask.any() else None
        if missing_dates:
            print(f"缺失站点数据文件：{len(missing_dates)} 个日期")
        print(f"批量加载站点数据耗时：{time.time() - start_time:.2f} 秒")
        return self.site_data_batch_cache

    def _get_site_pm25_fast(self, date_str, target_hour):
        """获取指定北京时间的站点数据"""
        site_df = self.site_data_batch_cache.get(date_str)
        if site_df is None or site_df.empty:
            return {}
        hour_mask = site_df['hour'] == target_hour
        if not hour_mask.any():
            return {}
        hour_data = site_df[hour_mask].iloc[0] if len(site_df[hour_mask]) > 0 else None
        if hour_data is None:
            return {}
        non_station_cols = ['date', 'hour', 'type', 'time']
        station_cols = [col for col in site_df.columns if col not in non_station_cols]
        site_dict = {}
        for col in station_cols:
            val = hour_data[col]
            if pd.notna(val) and 0 <= val <= 1000:
                site_dict[str(col).strip()] = float(val)
        return site_dict

    def _init_common_grid(self, file_path):
        """从任意一个nc文件初始化规则网格坐标"""
        try:
            ds = xr.open_dataset(file_path, engine='netcdf4')
            # 提取经纬度变量（支持常见命名）
            lon = None
            lat = None
            if 'lon' in ds.variables:
                lon = ds['lon'].values
            elif 'XLONG' in ds.variables:
                lon = ds['XLONG'].values
            if 'lat' in ds.variables:
                lat = ds['lat'].values
            elif 'XLAT' in ds.variables:
                lat = ds['XLAT'].values
            if lon is None or lat is None:
                ds.close()
                raise ValueError("未找到经纬度变量")
            # 如果是2D网格（如WRF），取第一行/列
            if lon.ndim == 2:
                lon = lon[0, :]
            if lat.ndim == 2:
                lat = lat[:, 0]
            # 确保单调递增（griddata要求）
            if lon[0] > lon[-1]:
                lon = lon[::-1]
            if lat[0] > lat[-1]:
                lat = lat[::-1]
            self.grid_lon = lon
            self.grid_lat = lat
            ds.close()
            print(f"初始化规则网格成功：经度范围 [{lon.min():.2f}, {lon.max():.2f}]，纬度范围 [{lat.min():.2f}, {lat.max():.2f}]")
        except Exception as e:
            print(f"初始化规则网格失败，将回退至原方法：{e}")
            self.grid_lon = None
            self.grid_lat = None

    def _extract_monitor_pm25_fast(self, date_str, cst_hour_str):
        """
        提取监测数据并插值到站点位置
        优先使用规则网格插值（interpn），失败则回退到griddata
        """
        # 构建文件名
        monitor_file = None
        for pattern in [f"{date_str}{cst_hour_str}.nc", f"pm25_{date_str}{cst_hour_str}.nc"]:
            f_path = os.path.join(self.monitor_data_dir, pattern)
            if os.path.exists(f_path):
                monitor_file = f_path
                break
        if monitor_file is None:
            return None

        # 若尚未初始化规则网格，尝试从当前文件初始化
        if self.grid_lon is None and self.grid_lat is None:
            self._init_common_grid(monitor_file)

        try:
            ds = xr.open_dataset(monitor_file, engine='netcdf4')
            # 提取PM2.5变量
            pm25_var = None
            for var in ['pm25']:
                if var in ds.variables:
                    pm25_var = ds[var]
                    break
            if pm25_var is None:
                ds.close()
                return None
            pm25_data = pm25_var.values.squeeze()
            if pm25_data.ndim != 2:
                ds.close()
                return None

            # 使用规则网格插值（若可用）
            if self.grid_lon is not None and self.grid_lat is not None:
                # 构建站点坐标 (lat, lon)，顺序需与网格一致 (纬度, 经度)
                xi = np.column_stack([self.station_info['points'][:, 1], self.station_info['points'][:, 0]])
                try:
                    station_pm25 = interpn((self.grid_lat, self.grid_lon), pm25_data, xi,
                                           method=self.interp_method,
                                           bounds_error=False,
                                           fill_value=np.nan)
                    # 值域过滤
                    station_pm25[(station_pm25 < 0) | (station_pm25 > 1000)] = np.nan
                    ds.close()
                    return station_pm25
                except Exception as e:
                    # 规则网格插值失败，尝试原方法
                    print(f"规则网格插值失败，回退至griddata：{e}")
                    # 继续执行下面的griddata逻辑

            # 如果规则网格不可用或插值失败，使用原griddata方法（并缓存网格）
            if monitor_file in self.grid_cache:
                grid_data = self.grid_cache[monitor_file]
            else:
                # 获取经纬度（用于不规则网格）
                lon_var = ds['lon'] if 'lon' in ds.variables else ds['XLONG'] if 'XLONG' in ds.variables else None
                lat_var = ds['lat'] if 'lat' in ds.variables else ds['XLAT'] if 'XLAT' in ds.variables else None
                if lon_var is None or lat_var is None:
                    ds.close()
                    return None
                lon_vals = lon_var.values.squeeze()
                lat_vals = lat_var.values.squeeze()
                if lon_vals.ndim == 1 and lat_vals.ndim == 1:
                    lon_mesh, lat_mesh = np.meshgrid(lon_vals, lat_vals)
                else:
                    lon_mesh, lat_mesh = lon_vals, lat_vals
                grid_points = np.column_stack([lon_mesh.flatten(), lat_mesh.flatten()])
                grid_values = pm25_data.flatten()
                valid_mask = ~np.isnan(grid_values)
                if np.sum(valid_mask) < 10:
                    ds.close()
                    return None
                grid_data = {
                    'points': grid_points[valid_mask],
                    'values': grid_values[valid_mask],
                    'pm25_data': pm25_data
                }
                self.grid_cache[monitor_file] = grid_data

            station_pm25 = griddata(
                grid_data['points'],
                grid_data['values'],
                self.station_info['points'],
                method=self.interp_method,
                fill_value=np.nan
            )
            station_pm25[(station_pm25 < 0) | (station_pm25 > 1000)] = np.nan
            ds.close()
            return station_pm25

        except Exception as e:
            print(f"提取监测数据错误({date_str} {cst_hour_str} CST): {e}")
            return None

    def _extract_num_fast(self, s):
        """快速提取数字"""
        return ''.join(filter(str.isdigit, str(s))) if s else ''

    def _validate_hour_worker(self, args):
        """单小时验证工作函数"""
        date_str, cst_hour_str = args
        try:
            monitor_pm25 = self._extract_monitor_pm25_fast(date_str, cst_hour_str)
            if monitor_pm25 is None:
                return None
            site_dict = self._get_site_pm25_fast(date_str, int(cst_hour_str))
            if not site_dict:
                return None
            # 匹配站点
            station_ids = np.array(self.station_info['ids'])
            station_nums = np.array(self.station_info['nums'])
            site_ids = np.array(list(site_dict.keys()))
            site_nums = np.array([self._extract_num_fast(oid) for oid in site_ids])
            match_mask = np.isin(station_ids, site_ids)
            if np.sum(match_mask) < 3:
                num_match_mask = np.isin(station_nums, site_nums[site_nums != ''])
                match_mask = match_mask | num_match_mask
            valid_mask = match_mask & ~np.isnan(monitor_pm25)
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) < 3:
                return None
            matched_monitor = monitor_pm25[valid_indices]
            matched_site = np.array([site_dict.get(station_ids[i], np.nan) for i in valid_indices])
            matched_site = matched_site[~np.isnan(matched_site)]
            matched_monitor = matched_monitor[:len(matched_site)]
            if len(matched_monitor) < 3:
                return None
            # 统计指标
            monitor_mean = np.mean(matched_monitor)
            site_mean = np.mean(matched_site)
            bias = monitor_mean - site_mean
            mae = np.mean(np.abs(matched_monitor - matched_site))
            rmse = np.sqrt(np.mean((matched_monitor - matched_site) ** 2))
            try:
                correlation = np.corrcoef(matched_monitor, matched_site)[0, 1]
                r2 = r2_score(matched_site, matched_monitor)
            except:
                correlation = np.nan
                r2 = np.nan
            return {
                'datetime': f"{date_str} {cst_hour_str}:00 (CST)",
                'n_points': len(matched_monitor),
                'monitor_mean': float(monitor_mean),
                'site_mean': float(site_mean),
                'bias': float(bias),
                'mae': float(mae),
                'rmse': float(rmse),
                'correlation': float(correlation),
                'r2': float(r2),
                'monitor_values': matched_monitor,
                'site_values': matched_site
            }
        except Exception as e:
            print(f"单小时验证错误({date_str} {cst_hour_str} CST): {e}")
            return None

    def validate_parallel(self):
        """并行验证主函数"""
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        date_range = pd.date_range(start=start, end=end, freq='D')
        date_list = [d.strftime("%Y%m%d") for d in date_range]
        print(f"验证范围：{len(date_list)} 天，使用 {self.n_workers} 进程并行处理")
        print(f"时区说明：站点数据=北京时间，监测数据=北京时间")
        print(f"当前插值方法：{self.interp_method}")
        print(f"站点数据类型：{self.site_data_type}")
        self._batch_load_site_data(date_list)
        tasks = []
        for date_str in date_list:
            for cst_hour in range(24):
                tasks.append((date_str, f"{cst_hour:02d}"))
        start_time = time.time()
        with Pool(processes=self.n_workers) as pool:
            worker_func = partial(self._validate_hour_worker)
            results = list(tqdm(pool.imap(worker_func, tasks), total=len(tasks), desc="并行验证中"))
        for i, (date_str, cst_hour_str) in enumerate(tasks):
            result = results[i]
            if result is None:
                continue
            period = f"h{cst_hour_str}"
            if date_str not in self.validation_results:
                self.validation_results[date_str] = {}
            if period not in self.validation_results[date_str]:
                self.validation_results[date_str][period] = []
            self.validation_results[date_str][period].append(result)
            key = f"{date_str}_{cst_hour_str}"
            self.comparison_data[key] = {
                'monitor': result['monitor_values'],
                'site': result['site_values']
            }
        for date_str, period_data in self.validation_results.items():
            for period, hour_results in period_data.items():
                if hour_results:
                    r = hour_results[0]
                    self.validation_results[date_str][period] = {
                        'mean_bias': r['bias'],
                        'mean_rmse': r['rmse'],
                        'mean_r2': r['r2'],
                        'mean_corr': r['correlation'],
                        'n_hours': 1,
                        'total_points': r['n_points']
                    }
                else:
                    self.validation_results[date_str][period] = None
        print(f"并行验证完成！总耗时：{time.time() - start_time:.2f} 秒")

    def generate_summary(self, output_path='pm25_validation_summary_optimized.csv'):
        """生成摘要统计"""
        summary_data = []
        for date_str, period_results in self.validation_results.items():
            for period_name, stats in period_results.items():
                if stats is None:
                    continue
                summary_data.append({
                    'date': date_str,
                    'period_cst': period_name,
                    'n_hours': stats['n_hours'],
                    'n_points': stats.get('total_points', 0),
                    'mean_bias': stats['mean_bias'],
                    'mean_rmse': stats['mean_rmse'],
                    'mean_r2': stats['mean_r2'],
                    'mean_corr': stats['mean_corr']
                })
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"摘要已保存：{output_path}")
            print("\n总体验证统计（监测数据VS站点数据）")
            print(f"平均R²: {summary_df['mean_r2'].mean():.3f}")
            print(f"平均RMSE: {summary_df['mean_rmse'].mean():.1f} μg/m³")
            print(f"总数据点: {summary_df['n_points'].sum()}")
            return summary_df
        else:
            print("无有效结果")
            return None

    # ==================== 绘图方法（来自数据验证2，并修改颜色映射和右上角指标） ====================
    def _collect_plot_data(self):
        """从 comparison_data 收集所有数据点，返回 DataFrame（列：pm25_obs, pm25_sim），并剔除观测值大于500的点"""
        obs_list = []
        sim_list = []
        for data in self.comparison_data.values():
            mon = data['monitor']
            sit = data['site']
            obs_list.extend(sit)
            sim_list.extend(mon)
        df = pd.DataFrame({'pm25_obs': obs_list, 'pm25_sim': sim_list})
        df = df.dropna()
        # 剔除站点数据（观测值）大于500的数据点
        df = df[df['pm25_obs'] <= 500]
        return df

    def plot_scatter(self, save_path='pm25_scatter.png'):
        """观测 vs 模拟 散点密度图 + 1:1 线 + 线性拟合（黄-橙-红配色，右上角显示MB/RMSE/MFB/MFE）"""
        df = self._collect_plot_data()
        if df.empty:
            print("没有有效数据点，无法绘图")
            return

        obs = df['pm25_obs'].to_numpy()
        sim = df['pm25_sim'].to_numpy()

        # 计算指标
        diff = sim - obs
        bias = float(np.mean(diff))
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        corr = float(np.corrcoef(obs, sim)[0, 1]) if len(obs) > 1 else np.nan
        r2 = r2_score(obs, sim) if len(obs) > 1 else np.nan

        # 计算 MFB 和 MFE
        epsilon = 1e-6
        mfb = 100 * np.mean(diff / (obs + sim + epsilon))
        mfe = 100 * np.mean(np.abs(diff) / (obs + sim + epsilon))

        coeffs = np.polyfit(obs, sim, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])

        plt.figure(figsize=(7.5, 7.5))
        # 自定义颜色列表：白 -> 黄 -> 橙 -> 红
        colors = ['white', (1.0, 1.0, 0.2), (1.0, 0.6, 0.0), (1.0, 0.2, 0.0)]
        cmap = LinearSegmentedColormap.from_list('white_yellow_orange_red', colors)

        hb = plt.hexbin(
            obs, sim,
            gridsize=80,
            bins="log",
            cmap=cmap,
            mincnt=1,
            linewidths=0.15,
            edgecolors="face",
        )
        cb = plt.colorbar(hb, fraction=0.046, pad=0.04)
        cb.set_label(r"$\log_{10}$(count)", fontsize=11)

        mx = max(400.0, np.nanmax(obs), np.nanmax(sim))
        plt.xlim(0, mx)
        plt.ylim(0, mx)
        plt.gca().set_aspect("equal")

        plt.plot([0, mx], [0, mx], "k--", linewidth=1.2, label="1:1 Line", zorder=5)
        xx = np.linspace(0, mx, 200)
        plt.plot(xx, slope * xx + intercept, "r-", linewidth=1.7,
                 label=f"Fit: y={slope:.2f}x+{intercept:.2f}", zorder=6)

        plt.xlabel(r"Site $PM_{2.5}$ ($\mu$g/m$^3$)", fontsize=12)
        plt.ylabel(r"Monitor $PM_{2.5}$ ($\mu$g/m$^3$)", fontsize=12)
        plt.title("PM2.5 Hourly Validation\n(Correlation Analysis)", fontsize=16,weight="bold", pad=12)
        plt.grid(True, alpha=0.35, linestyle="-", color="0.78")
        plt.legend(loc="lower right", frameon=True, fontsize=10)

        # 左上角标注：R, R², MAE, MFB, MFE, RMSE
        textstr_left = (
            rf"$R$ = {corr:.2f}" + "\n"
            rf"$R^2$ = {r2:.2f}" + "\n"
            rf"$MB$ = {bias:.2f}" + "\n"
            rf"$MAE$ = {mae:.2f}" + "\n"
            rf"$MFB$ = {mfb:.2f}%" + "\n"
            rf"$MFE$ = {mfe:.2f}%" + "\n"
            rf"$RMSE$ = {rmse:.2f}"
        )
        plt.text(
            0.03, 0.97,
            textstr_left,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", linewidth=0.8),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"散点图已保存：{save_path}")

    def plot_residual(self, save_path='pm25_residual.png'):
        """残差分析图（散点 + 直方图）"""
        df = self._collect_plot_data()
        if df.empty:
            print("没有有效数据点，无法绘图")
            return

        obs = df['pm25_obs'].to_numpy()
        residual = df['pm25_sim'].to_numpy() - obs

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(obs, residual, s=12, alpha=0.5, c="#d62728")
        ax1.axhline(0, color="k", linestyle="--", linewidth=1)
        ax1.set_xlabel("Site PM2.5 (obs)")
        ax1.set_ylabel("Residual (sim - obs)")
        ax1.set_title("Residual Scatter Plot")
        ax1.grid(alpha=0.3)

        ax2.hist(residual, bins=40, color="#2ca02c", alpha=0.8)
        ax2.axvline(0, color="k", linestyle="--", linewidth=1)
        ax2.set_xlabel("Residual (sim - obs)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Residual Distribution")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"残差图已保存：{save_path}")

    # ==================== 绘图方法结束 ====================


def main():
    """主函数（请根据实际情况修改路径）"""
    MONITOR_DATA_DIR = r"E:\pm25_1km_24h_new"                     # 监测数据目录（北京时间）
    SITE_DATA_DIR = r"E:\中国空气质量数据\站点空气质量\绔欑偣_20230101-20241231"  # 站点数据目录（北京时间）
    STATION_INFO_PATH = r"E:\中国空气质量数据\站点空气质量\站点列表-2022.02.13起.xlsx"
    # 初始化验证器，使用小时均值站点数据
    validator = PM25MonitorValidator(
        monitor_data_dir=MONITOR_DATA_DIR,
        site_data_dir=SITE_DATA_DIR,
        station_info_path=STATION_INFO_PATH,
        start_date="20230301",
        end_date="20240229",      # 可改为更长时间段
        n_workers=4,
        interp_method='nearest',    # 可根据需要修改
        site_data_type='PM2.5'     # 使用小时均值，而非24小时滑动平均
    )
    if validator.station_info is None:
        print("站点信息加载失败")
        return
    validator.validate_parallel()
    validator.generate_summary()
    # 使用新的绘图方法
    validator.plot_scatter(save_path=r"C:\Users\zyd\Desktop\scatter.png")
    validator.plot_residual(save_path=r"C:\Users\zyd\Desktop\residual.png")


if __name__ == "__main__":
    main()