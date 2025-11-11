import os
from typing import List
import numpy as np
import pandas as pd
import requests, requests_cache, openmeteo_requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import StringIO
from datetime import datetime, timedelta
from calendar import monthrange
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
# =================================================
# SECTION 0: GLOBAL CONFIGURATION AND UTILITIES
# =================================================

# --------------------------------------
# [0-1] 전역 상수 설정
# --------------------------------------
LATITUDE = 37.5714
LONGITUDE = 126.9658
HOURLY_VARS = ["temperature_2m", "wind_speed_10m"] 
MODELS = ["kma_seamless", "ecmwf_ifs", "icon_global", "gfs_global", "ukmo_global_deterministic_10km"]
TIMEZONE = "Asia/Seoul"

# KMA API 설정
API_KEY = 'Wz5SdaosTtS-UnWqLH7USA'
STN = '108' # 0=전체지점(제품별 지원 여부 확인), 108 : 서울

# KMA API 응답 컬럼 이름 
KMA_COLUMN_NAMES = [
    "YYMMDDHHMI_KST", "STN_ID", "WD_16", "WS_m/s", "GST_WD", "GST_WS", "GST_TM",
    "PA", "PS", "PT", "PR", "TA", "TD", "HM", "PV", "RN_mm", "RN_DAY", "RN_JUN", "RN_INT",
    "SD_HR3", "SD_DAY", "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN",
    "CT_TOP", "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_5", "TE_10",
    "TE_20", "TE_30", "ST_SEA", "WH", "BF", "IR", "IX"
]
# KMA와 비교할 변수 (기온, 풍속)
COMPARE_VARS = ['temperature_2m', 'wind_speed_10m'] 

# --------------------------------------
# [0-2] HTTP 재시도 유틸리티 설정
# --------------------------------------
def retry(
    session, 
    retries=3, 
    backoff_factor=0.2
):
    """요청 세션에 재시도 정책을 적용합니다."""

    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 504)
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


# ==========================================
# SECTION 1: WeatherForecastProcessor CLASS
# ==========================================

class WeatherForecastProcessor:
    """
    Open-Meteo API를 사용하여 날씨 데이터를 수집, KMA 관측 데이터와 비교, 
    정제, 요약, 시각화, 저장하는 통합 클래스.
    """

    def __init__(
        self,
        latitude,
        longitude,
        hourly_vars,
        models,
        timezone,
        kma_api_key,
        kma_stn,
        past_days=31,
        forecast_days=1
    ):
        # 속성 초기화
        self.latitude = latitude
        self.longitude = longitude
        self.hourly_vars = hourly_vars
        self.models = models
        self.timezone = timezone
        self.past_days = past_days
        self.forecast_days = forecast_days
        self.compare_vars = [var for var in hourly_vars if var in ['temperature_2m', 'wind_speed_10m']]

        # KMA 설정
        self.kma_api_key = kma_api_key
        self.kma_stn = kma_stn
        self.kma_column_names = KMA_COLUMN_NAMES # 전역 상수를 그대로 사용

        # 세션 및 클라이언트 초기화
        cache_session = requests_cache.CachedSession(
            '.cache',
            expire_after=3600
        )
        retry_session = retry(
            cache_session,
            retries=5,
            backoff_factor=0.2
        )
        self.client = openmeteo_requests.Client(session=retry_session)

        # 데이터프레임 속성 초기화
        self.df_raw = None           # OpenMeteo 원본 데이터
        self.df_clean = None         # OpenMeteo 정제 데이터
        self.df_summary = None       # OpenMeteo 모델별 통계 요약
        self.df_kma_processed = None # KMA 관측 데이터
        self.df_accuracy = None      # 모델 정확도 지표
        self.report_text = ""
        self.df_report = None        
        self.outlier_count = 0
        
    # --------------------------------------
    # [1-1] KMA 데이터 처리 유틸리티
    # --------------------------------------

    def _process_kma_data(
        self, 
        df_kma: pd.DataFrame
    ) -> pd.DataFrame:
        """KMA ASOS 데이터를 OpenMeteo 형식에 맞게 정리합니다."""

        if df_kma.empty:
            return df_kma

        # 시간 변환
        df_kma['time'] = pd.to_datetime(
            df_kma['YYMMDDHHMI_KST'],
            format='%Y%m%d%H%M',
            errors='coerce'
        )
        # OpenMeteo 형식에 맞춰 컬럼명 변경
        df_kma = df_kma.rename(columns={
            'TA': 'temperature_2m',
            'WS_m/s': 'wind_speed_10m'
        })
        # 필요한 컬럼만 선택
        df_processed = df_kma[['time', 'temperature_2m', 'wind_speed_10m']].copy()

        # 시간(H) 단위로 내림 (KMA 분 단위 -> OpenMeteo 시간 단위)
        df_processed['time'] = df_processed['time'].dt.floor('H')

        # 결측치 제거
        df_processed = df_processed.dropna(subset=['time', 'temperature_2m', 'wind_speed_10m'])

        print("⭕ KMA 데이터 시간 변환 및 컬럼 정리 완료.")
        print(f"✅ 변환된 데이터 기간: {df_processed['time'].min()} ~ {df_processed['time'].max()}")
        return df_processed

    def download_kma_data(
        self, 
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """KMA API에서 데이터를 다운로드하고 전처리합니다."""

        url = (
            "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
            f"?tm1={start_date}0000&tm2={end_date}2300&stn={self.kma_stn}&authKey={self.kma_api_key}"
        )
        print(f"✅ KMA API 호출: {url}")

        r = requests.get(url, timeout=30)
        if r.encoding is None:
            r.encoding = r.apparent_encoding or "euc-kr"
        text = r.text

        # 주석(#) 제거
        cleaned = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
        sio = StringIO(cleaned)

        # 칼럼 개수 탐지 및 이름 지정 로직
        sample_line = next((ln for ln in cleaned.splitlines() if ln.strip()), "")
        n_cols_detected = len(sample_line.split())

        if n_cols_detected != len(self.kma_column_names):
            if n_cols_detected < len(self.kma_column_names):
                names = self.kma_column_names[:n_cols_detected]
            else:
                extra = [f"col_{i}" for i in range(len(self.kma_column_names), n_cols_detected)]
                names = self.kma_column_names + extra
        else:
            names = self.kma_column_names

        # 데이터 로드
        try:
            df_full = pd.read_csv(
                sio,
                sep=r"\s+",
                names=names,
                engine="python"
            )
        except Exception as e:
            print(f"❌ KMA API 응답 파싱 실패: {e}")
            self.df_kma_processed = pd.DataFrame()
            return self.df_kma_processed

        # 필요한 컬럼만 필터링 및 전처리
        required_cols = ["YYMMDDHHMI_KST", "TA", "WS_m/s"]
        filtered_cols = [col for col in required_cols if col in df_full.columns]
        if len(filtered_cols) < 3:
            print(f"❌ KMA 데이터: 필수 컬럼 부족. \n✅ 존재하는 칼럼: {filtered_cols}")
            self.df_kma_processed = pd.DataFrame()
            return self.df_kma_processed

        df = df_full[filtered_cols].copy()
        df_processed = self._process_kma_data(df)

        # csv로 저장
        out_path = f'./ASOS_hourly_{start_date}_{end_date}_temp_wind_processed.csv'
        df_processed.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"⭕ {start_date}_{end_date} (기온/풍속 가공) 데이터 저장 완료 → \n✅ 저장된 파일 경로: {out_path}")
        
        self.df_kma_processed = df_processed
        return self.df_kma_processed


    def load_kma_data(
        self, 
        file_path: str
    ):
        """저장된 KMA CSV 파일을 로드합니다."""

        try:
            df = pd.read_csv(file_path, parse_dates=['time'])
            if not all(col in df.columns for col in ['temperature_2m', 'wind_speed_10m']):
                 raise ValueError("KMA file column names are incorrect.")

            print(f"✅ KMA 데이터 로드 성공. 기간: {df['time'].min()} ~ {df['time'].max()}")
            self.df_kma_processed = df
            return df
        except FileNotFoundError:
            print(f"❌ KMA 파일 찾을 수 없음: {file_path}. KMA 데이터 없이 진행합니다.")
            self.df_kma_processed = None
            return None
        except Exception as e:
            print(f"❌ KMA 데이터 로드 실패: {e}. KMA 데이터 없이 진행합니다.")
            self.df_kma_processed = None
            return None

    # ------------------------
    # [1-2] 데이터 수집 메서드
    # ------------------------
    def fetch_data(self):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": self.hourly_vars,
            "models": self.models,
            "timezone": self.timezone,
            "past_days": self.past_days,
            "forecast_days": self.forecast_days,
            "wind_speed_unit": "ms"
        }

        try:
            responses = self.client.weather_api(url, params=params)
            
            all_data = []
            model_map = {}
            for i, model_response in enumerate(responses):
                model_map[model_response.Model()] = self.models[i]

            for response in responses:
                model_id = response.Model()
                model_name = model_map.get(model_id)

                hourly = response.Hourly()

                # 시간 인덱스 생성
                time_index_utc = pd.date_range(
                    start=pd.to_datetime(
                        hourly.Time(), 
                        unit="s", 
                        utc=True
                    ).tz_convert(self.timezone),
                    end=pd.to_datetime(
                        hourly.TimeEnd(), 
                        unit="s",
                        utc=True
                    ).tz_convert(self.timezone),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
                time_seoul = time_index_utc.tz_localize(None)

                hourly_data = {
                    "time": time_seoul,
                    "model": model_name
                }
                for i, var_name in enumerate(self.hourly_vars):
                    hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

                df = pd.DataFrame(data=hourly_data)
                all_data.append(df)

            self.df_raw = pd.concat(all_data, ignore_index=True)
            print(f"✅ 수집된 예측 데이터 기간: {self.df_raw['time'].min().date()} ~ {self.df_raw['time'].max().date()}")
            print("⭕ 데이터 수집 완료. df_raw에 모든 모델 데이터 통합됨.")
            return self.df_raw

        except Exception as e:
            print(f"❌ 데이터 수집 실패: {e}")
            return pd.DataFrame()

    # --------------------------------
    # [1-3] 데이터 정제 및 요약 메서드
    # --------------------------------
    def make_summary(self):
        """데이터를 정제하고 모델별 요약 통계를 생성합니다. """

        if self.df_raw is None:
            raise ValueError("Raw data is not fetched. Call fetch_data() first.")

        df_clean = self.df_raw.copy()
        initial_count = len(df_clean)

        # 결측치 처리 (평균값으로 대체)
        for col in self.compare_vars:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

        # 이상치 처리 (IQR 기반)
        #df_temp = df_clean.copy()
        #outliers_removed = 0
        
        #for col in self.hourly_vars: 
        #    q1 = df_temp[col].quantile(0.25)
        #    q3 = df_temp[col].quantile(0.75)
        #    iqr = q3 - q1
        #    lower_bound = q1 - 1.5 * iqr
        #    upper_bound = q3 + 1.5 * iqr

        #    is_outlier = (df_temp[col] < lower_bound) | (df_temp[col] > upper_bound)
        #    outliers_removed += is_outlier.sum()
        #    df_temp = df_temp[~is_outlier]

        #df_clean = df_temp
        #self.outlier_count = initial_count - len(df_clean)
        #print(f"✅ 전 모델에 대한 데이터 정제 전 관측치 개수: {initial_count} -> 정제 후 관측치 개수: {len(df_clean)} (이상치 {self.outlier_count}개 제거)")


        # 모델별 요약 통계 생성
        summary_list = []
        for model_name, group in df_clean.groupby("model"):
            stats = {
                "model": model_name,
                "count": len(group),

                # 기온 정보
                "mean_temp": round(group["temperature_2m"].mean(), 2),
                "max_temp": round(group["temperature_2m"].max(), 2),
                "min_temp": round(group["temperature_2m"].min(), 2),
                "median_temp": round(group["temperature_2m"].median(), 2),
                "std_temp": round(group["temperature_2m"].std(), 2),
                "range_temp": round(group["temperature_2m"].max() - group["temperature_2m"].min(), 2),

                # 풍속 정보
                "mean_wind": round(group["wind_speed_10m"].mean(), 2),
                "max_wind": round(group["wind_speed_10m"].max(), 2),
                "min_wind": round(group["wind_speed_10m"].min(), 2),
                "median_wind": round(group["wind_speed_10m"].median(), 2),
                "std_wind": round(group["wind_speed_10m"].std(), 2),
                "range_wind": round(group["wind_speed_10m"].max() - group["wind_speed_10m"].min(), 2),
            }
            summary_list.append(stats)

        self.df_clean = df_clean
        self.df_summary = pd.DataFrame(summary_list)

        self._generate_report_text()
        print("⭕ 데이터 정제 및 요약 완료.")
        return self.df_clean
    
    # --------------------------
    # [1-4] 정확도 계산 유틸리티 
    # --------------------------
    def _calculate_model_metrics(
        self,
        df_clean: pd.DataFrame,
        df_kma_processed: pd.DataFrame,
        var_name: str
    ) -> pd.DataFrame:
        """KMA 관측 데이터와 OpenMeteo 모델 예측 데이터를 병합하여 피벗합니다."""

        if var_name not in df_kma_processed.columns:
            raise ValueError(f"KMA processed data does not contain the variable: {var_name}")

        df_kma_temp = df_kma_processed.rename(columns={var_name: 'Observed'}).copy()

        # 두 DataFrame을 time 컬럼을 기준으로 Inner Join
        df_merged = pd.merge(
            df_kma_temp[['time', 'Observed']],
            df_clean[['time', 'model', var_name]],
            on='time',
            how='inner'
        )

        # 모델 이름을 컬럼으로 피벗하여 비교하기 쉬운 형태로 변환
        df_pivot = df_merged.pivot(
            index='time',
            columns='model',
            values=var_name
        ).reset_index()

        # Observed 컬럼을 왼쪽으로 다시 병합
        df_final = pd.merge(
            df_kma_temp[['time', 'Observed']].drop_duplicates(subset=['time']),
            df_pivot,
            on='time',
            how='inner'
        )

        # 관측값/예측값에 결측치(NaN)가 있으면 해당 행 제거
        model_columns = [col for col in df_pivot.columns if col != 'time']
        df_final = df_final.dropna(subset=['Observed'] + model_columns)

        return df_final


    def _calculate_metrics_for_var(
        self,
        df_final: pd.DataFrame,
        var_name: str,
        models: List[str]
    ) -> pd.DataFrame:
        """병합된 DataFrame을 사용하여 RMSE, MAE, R 지표를 계산합니다."""

        results = []

        observed = df_final['Observed']
        available_models = [m for m in models if m in df_final.columns]

        for model in available_models:
            forecast = df_final[model]

            # RMSE 계산
            rmse = np.sqrt(np.mean((forecast - observed) ** 2))

            # MAE 계산
            mae = np.mean(np.abs(forecast - observed))

            # 상관계수 (R) 계산
            correlation = observed.corr(forecast)

            results.append({
                'Variable': var_name,
                'Model': model,
                'RMSE (Error)': round(rmse, 3),
                'MAE (Error)': round(mae, 3),
                'Corr (R)': round(correlation, 4)
            })

        return pd.DataFrame(results)

    def calculate_accuracy_metrics(self):
        """KMA 관측 데이터와 모델 예측 데이터의 정확도 지표를 계산하고 저장합니다."""

        df_kma_processed = self.df_kma_processed
        if df_kma_processed is None or df_kma_processed.empty:
            print("❌ KMA 관측 데이터가 없어 모델 정확도 비교를 수행할 수 없습니다.")
            return

        all_metrics = []

        # 기온/풍속 변수별로 RMSE, MAE, Corr 계산
        for var in self.compare_vars:
            print(f"\n--- 📈 {var.upper()} 정확도 지표 계산 시작 ---")

            try:
                df_merged_pivot = self._calculate_model_metrics(
                    self.df_clean, 
                    df_kma_processed, 
                    var
                  )
                metrics_df = self._calculate_metrics_for_var(
                    df_merged_pivot, 
                    var, 
                    self.models
                )
                all_metrics.append(metrics_df)
            except Exception as e:
                print(f"❌ {var.upper()} 정확도 계산 중 오류 발생: {e}")
                continue

        if not all_metrics:
            print("\n⚠️ 계산 가능한 정확도 지표가 없습니다.")
            return

        # 최종 결과 통합 및 순위 매기기
        df_final_metrics = pd.concat(
            all_metrics, 
            ignore_index=True
        )

        # 모델의 정확도 순위(Rank) 계산: RMSE가 가장 낮은 모델이 1위
        df_final_metrics['Accuracy Rank (RMSE)'] = df_final_metrics.groupby('Variable')['RMSE (Error)'].rank(
            method='min',
            ascending=True
        ).astype(int)

        self.df_accuracy = df_final_metrics.sort_values(
            by=['Variable', 'Accuracy Rank (RMSE)']
        ).reset_index(drop=True)
        
        print("⭕ 모델 정확도 지표 계산 완료.")
        
        # 최종 출력: 변수별로 분리하여 출력
        self._display_accuracy_metrics()
    
    
    def _display_accuracy_metrics(self):
        """계산된 정확도 지표를 콘솔에 출력합니다."""

        df_final_metrics = self.df_accuracy
        if df_final_metrics is None or df_final_metrics.empty:
            return

        print("\n" + "="*80)
        print("             ✅ 최종 모델 정확도 (오차/유사도) vs KMA 관측 결과 비교")
        print("="*80)

        # 기온 결과 출력
        df_temp = df_final_metrics[df_final_metrics['Variable'] == 'temperature_2m'].drop(columns=['Variable'])
        if not df_temp.empty:
            print("\n🌡️ [ 기온 (Temperature_2m) 모델 정확도 비교 ]")
            print("--------------------------------------------------------------------------------")
            print(df_temp.to_string(index=False))

        print("\n" + "-"*80)

        # 풍속 결과 출력
        df_wind = df_final_metrics[df_final_metrics['Variable'] == 'wind_speed_10m'].drop(columns=['Variable'])
        if not df_wind.empty:
            print("💨 [ 풍속 (Wind_speed_10m) 모델 정확도 비교 ]")
            print("--------------------------------------------------------------------------------")
            print(df_wind.to_string(index=False))

        print("\n" + "="*80)
        print("* 해석:")
        print(" - RMSE/MAE (Error): 낮을수록 정확합니다. 예측값과 실제값의 차이가 작음을 의미합니다.")
        print(" - Corr (R): 1에 가까울수록 유사합니다. 경향성(패턴)이 일치함을 의미합니다.")
        print(" - Accuracy Rank (RMSE): RMSE가 가장 낮은 모델이 1위입니다.")
        
    # -------------------
    # [1-5] 시각화 메서드
    # -------------------
    def visualize_data(
        self,
        start_time: str = None, # 시작 시간 (예: '2025-10-03')
        end_time: str = None     # 종료 시간 (예: '2025-10-09')
    ):
        """모델 예측값과 KMA 관측값(제공될 경우)을 시계열 그래프로 시각화합니다."""
        MODEL_COLORS = {
            "KMA Observed": "black"  
            # 다른 모델들은 seaborn이 기본 팔레트 사용
        }
        if self.df_clean is None:
            raise ValueError("Cleaned data is missing. Call make_summary() first.")
        
        df_kma_processed = self.df_kma_processed
        
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (18, 10) 

        fig, axes = plt.subplots(2, 1, sharex=True) # 2개 서브플롯
        has_kma_data = df_kma_processed is not None and not df_kma_processed.empty

        # OpenMeteo 모델 데이터에 KMA 관측 데이터를 통합
        if has_kma_data:
            df_kma_labeled = df_kma_processed.copy()
            df_kma_labeled['model'] = 'KMA Observed'

            # KMA 데이터에 없는 컬럼(예: shortwave_radiation)을 NaN으로 채워서 통합
            df_kma_filled = df_kma_labeled.reindex(columns=self.df_clean.columns)
            df_plot = pd.concat([self.df_clean, df_kma_filled], ignore_index=True)
            print("✅ KMA 관측값을 모델 데이터에 통합하여 시각화합니다.")
        else:
            df_plot = self.df_clean
            print("⚠️ KMA 관측 데이터가 제공되지 않아 모델 예측값만 시각화합니다.")


        # 지정된 기간으로 데이터 필터링
        if start_time and end_time:
            try:
                # pandas datetime으로 변환 (시간 정보가 없으면 자정으로 간주)
                start_dt = pd.to_datetime(start_time)
                # end_time은 해당 날짜의 끝(23:59:59)까지 포함하도록 하루를 더합니다.
                end_dt = pd.to_datetime(end_time) + timedelta(days=1)
                
                # 필터링 수행
                df_plot = df_plot[
                    (df_plot['time'] >= start_dt) & 
                    (df_plot['time'] < end_dt)
                ].copy()
                
                print(f"✅ 시각화 기간 필터링: {start_time} ~ {end_time}")
            except Exception as e:
                print(f"❌ 기간 필터링 중 오류 발생: {e}. 전체 기간에 대해 시각화합니다.")
        
        if df_plot.empty:
             print("❌ 필터링된 기간에 해당하는 데이터가 없습니다. 시각화를 건너뜁니다.")
             return
        
        
        # 시각화 수행
        BASE_LINEWIDTH = 1.5
        KMA_LINEWIDTH = 3.0
        
        def _plot_one(ax, yvar, title_prefix):
            non_kma = df_plot[df_plot["model"] != "KMA Observed"]
            kma_only = df_plot[df_plot["model"] == "KMA Observed"]
        
            # 1) 모델들 먼저: legend ON (항목 생성)
            sns.lineplot(
                data=non_kma,
                x="time",
                y=yvar,
                hue="model",
                ax=ax,
                linewidth=BASE_LINEWIDTH,
                legend="brief"  # ✅ 모델별 항목 생성
            )
        
            # 2) 관측선: 검정/굵게/맨 위
            obs_line = sns.lineplot(
                data=kma_only,
                x="time",
                y=yvar,
                ax=ax,
                color="black",
                linewidth=KMA_LINEWIDTH,
                label="KMA Observed"  # ✅ 범례 라벨
            )
            obs_line.lines[-1].set_zorder(10)
        
            # 3) 범례 재구성: 중복 제거 + 'KMA Observed'를 마지막에
            handles, labels = ax.get_legend_handles_labels()
            pairs = OrderedDict(
                (lab, h) for h, lab in zip(handles, labels)
                if lab and lab != "_nolegend_"
            )
            # 관측을 맨 마지막에 배치
            if "KMA Observed" in pairs:
                kma_handle = pairs.pop("KMA Observed")
                pairs["KMA Observed"] = kma_handle
        
            ax.legend(pairs.values(), pairs.keys(), title="Model", loc="upper right")
        
            ax.set_title(
                f"{title_prefix} ({df_plot['time'].min().strftime('%Y-%m-%d')} ~ {df_plot['time'].max().strftime('%Y-%m-%d')})",
                fontsize=16
            )
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)
        
        # 기온
        temp_ax = axes[0]
        _plot_one(temp_ax, "temperature_2m", "Comparison of 2m Temperature Predictions vs. KMA Observed")
        temp_ax.set_ylabel("Temperature (°C)")
        
        # 풍속
        wind_ax = axes[1]
        _plot_one(wind_ax, "wind_speed_10m", "Comparison of 10m Wind Speed Predictions vs. KMA Observed")
        wind_ax.set_ylabel("Wind Speed (m/s)")
        
        plt.tight_layout()
        plt.show()
        print("⭕ 시각화 완료.")

    # ------------------------
    # [1-6] 보고서 텍스트 생성
    # ------------------------
    def _generate_report_text(self):
        """요약 보고서 텍스트를 생성하여 self.report_text와 self.df_report에 저장합니다."""
        if self.df_summary is None or self.df_clean is None:
            raise ValueError("Data processing is incomplete.")

        df_summary = self.df_summary
        df_clean = self.df_clean

        start_date = df_clean["time"].min().strftime("%Y-%m-%d %H:%M") if not df_clean.empty else "N/A"
        end_date = df_clean["time"].max().strftime("%Y-%m-%d %H:%M") if not df_clean.empty else "N/A"
        total_count = len(df_clean)

        report_text = f"""\
    {'='*65}
    [ 날씨 모델 요약 보고서 ]
    {'='*65}

    📍 분석 위치: 서울 ({self.latitude:.2f}, {self.longitude:.2f})
    📅 분석 기간: {start_date} ~ {end_date}
    📊 총 관측치: {total_count}개 ({len(self.models)}개 모델)
    🧹 이상치 제거 관측치: {getattr(self, 'outlier_count', 0)}개

    {'='*65}
    --- 🔍 모델별 상세 예측 및 통계 결과 (기온/풍속) ---
    {'='*65}
    """

        for _, row in df_summary.iterrows():
            report_text += f"""
    [ 모델: {row['model']} ] (총 관측치: {row['count']}개)

    🌡️ 기온 (Temperature_2m)
      - 평균 기온: {row['mean_temp']:.2f}°C (중앙값: {row['median_temp']:.2f}°C)
      - 최고 기온: {row['max_temp']:.2f}°C / 최저 기온: {row['min_temp']:.2f}°C
      - 기온 변동폭 (Max-Min): {row['range_temp']:.2f}°C
      - 기온 표준편차(변동성): {row['std_temp']:.2f}

    💨 풍속 (Wind_speed_10m)
      - 평균 풍속: **{row['mean_wind']:.2f} m/s (중앙값: {row['median_wind']:.2f} m/s)
      - 최고 풍속: {row['max_wind']:.2f} m/s / 최저 풍속: {row['min_wind']:.2f} m/s
      - 풍속 변동폭 (Max-Min): {row['range_wind']:.2f} m/s
      - 풍속 표준편차(변동성): {row['std_wind']:.2f}

    {'-'*65}
    """
        self.report_text = report_text
        self.df_report = pd.DataFrame({"Report Text": [self.report_text]})


    # -----------------
    # [1-7] 저장 메서드
    # -----------------
    def save_report(
        self,
        path="report_class_based.xlsx"
    ):
        """모든 데이터를 엑셀 파일로 저장하고 보고서를 출력합니다."""

        if self.df_raw is None or self.df_summary is None or self.df_report is None:
            raise ValueError("Data processing is incomplete. Call fetch_data() and make_summary() first.")

        with pd.ExcelWriter(path) as writer:
            self.df_raw.to_excel(
                writer,
                sheet_name="data_raw",
                index=False
            )
            self.df_summary.to_excel(
                writer,
                sheet_name="summary",
                index=False
            )
            self.df_report.to_excel(
                writer,
                sheet_name="report",
                index=False
            )
            
            if self.df_accuracy is not None:
                self.df_accuracy.to_excel(
                    writer,
                    sheet_name="accuracy_metrics",
                    index=False
                )

        print("\n" + self.report_text)
        print(f"⭕ 모든 과정이 완료. ✅ 보고서 파일: {path}")


# =========================
# SECTION 2: MAIN EXECUTION
# =========================

def main():
    print("🚀 날씨 모델 데이터 분석 프로세스를 시작합니다!")

    # KMA 데이터 다운로드 기간 설정 (OpenMeteo past_days와 동일하게)
    past_days = 31 
    END_DATE = datetime.now().strftime('%Y%m%d')
    START_DATE = (datetime.now() - timedelta(days=past_days)).strftime('%Y%m%d')
    
    # 프로세서 클래스 인스턴스 생성
    processor = WeatherForecastProcessor(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        hourly_vars=HOURLY_VARS,
        models=MODELS,
        timezone=TIMEZONE,
        kma_api_key=API_KEY,
        kma_stn=STN,
        past_days=past_days
    )

    # KMA 데이터 다운로드 및 로드
    KMA_FILE_NAME = f'./ASOS_hourly_{START_DATE}_{END_DATE}_temp_wind_processed.csv'
    
    if os.path.exists(KMA_FILE_NAME):
        processor.load_kma_data(KMA_FILE_NAME)
    else:
        try:
            processor.download_kma_data(START_DATE, END_DATE)
        except Exception as e:
            print(f"❌ KMA 데이터 다운로드 실패: {e}")
            processor.df_kma_processed = None
    
    # OpenMeteo 데이터 수집
    processor.fetch_data()
    
    # 정제 및 요약
    if processor.df_raw is not None:
        processor.make_summary()
    
    # 정확도 비교 및 출력
    if processor.df_clean is not None and processor.df_kma_processed is not None:
        processor.calculate_accuracy_metrics()
    
    # 시각화
    processor.visualize_data(
        start_time="2025-10-03",
        end_time="2025-10-09"
    )

    # 보고서 저장
    processor.save_report()

if __name__ == "__main__":
    main()