import os, copy
import numpy as np
import torch
import pandas as pd
import warnings
from .ar import generate_process
from darts.models.forecasting.prophet_model import Prophet
from darts.models.forecasting.arima import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX
#from darts.models.forecasting.sf_auto_arima import SFAutoARIMA 
from darts.models.forecasting.theta import Theta
from darts.models.forecasting.transformer_model import TransformerModel
from darts.utils.utils import ModelMode, SeasonalityMode
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.lgbm import LightGBMModel
from darts import TimeSeries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pdb
from tqdm import tqdm
"""
    Generates forecasts from an ARIMA model
"""
    
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def compute_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
    
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    

def compute_wis(y_true, y_forecast):
    in_interval = np.logical_and(y_true >= y_forecast[0], y_true <= y_forecast[1])
    interval_width = y_forecast[1] - y_forecast[0]
    return np.mean(interval_width * np.logical_not(in_interval))
    

    
    
def generate_forecasts(
    data,
    model_name,
    savename,
    overwrite,
    log,
    fit_every,
    ahead,
    *args,
    **kwargs
    ):
    if not overwrite:
        try:
            saved = np.load(savename)
            forecasts = saved["forecasts"]
            return forecasts
        except:
            pass
    T = data.shape[0]
    forecasts = np.zeros((T,))
    data2 = copy.deepcopy(data)
    if log:
        data2['y'] = np.log(data2['y'])
        #data2[['y','admissions_all_covid_confirmed','inpatient_adult_beds','staff_icu_patients_covid_confirmed','absolute_change_percent_staff_icu_beds_occupied']] = data2[['y','admissions_all_covid_confirmed','inpatient_adult_beds','staff_icu_patients_covid_confirmed','absolute_change_percent_staff_icu_beds_occupied']].apply(np.log)
    # Create a new timestamp that is daily
    data2.index = pd.date_range(start=data2.index.min(), periods=len(data2), freq='D')
    #y = TimeSeries.from_dataframe(data2[['y','admissions_all_covid_confirmed','inpatient_adult_beds','staff_icu_patients_covid_confirmed','absolute_change_percent_staff_icu_beds_occupied']].interpolate().reset_index(drop=False).sort_values(by='index'), time_col='index', value_cols=['y','admissions_all_covid_confirmed','inpatient_adult_beds','staff_icu_patients_covid_confirmed','absolute_change_percent_staff_icu_beds_occupied'])
    y = TimeSeries.from_dataframe(data2['y'].interpolate().reset_index(drop=False).sort_values(by='index'), time_col='index', value_cols='y')
    # Create a TimeSeries for the exogenous variables
    #exog = TimeSeries.from_dataframe(data2[['admissions_all_covid_confirmed','inpatient_adult_beds','staff_icu_patients_covid_confirmed','absolute_change_percent_staff_icu_beds_occupied','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend']].interpolate().reset_index(drop=False).sort_values(by='index'), time_col='index', value_cols=['admissions_all_covid_confirmed','inpatient_adult_beds','staff_icu_patients_covid_confirmed','absolute_change_percent_staff_icu_beds_occupied','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend'])
    #exog = TimeSeries.from_dataframe(data2[['percent_adult_inpatient_beds_covid','percent_inpatient_beds_covid','total_staffed_pediatric_icu_beds','staff_pediatric_icu_beds_occupied','inpatient_pediatric_beds_used']].interpolate().reset_index(drop=False).sort_values(by='index'), time_col='index', value_cols=['percent_adult_inpatient_beds_covid','percent_inpatient_beds_covid','total_staffed_pediatric_icu_beds','staff_pediatric_icu_beds_occupied','inpatient_pediatric_beds_used'])
    #exog = TimeSeries.from_dataframe(data2[['AZ', 'CA', 'CT', 'DC', 'DE', 'ID', 'IL', 'KY', 'LA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MT', 'NC', 'NE', 'NH', 'NJ', 'NV', 'OR', 'RI', 'Region 1', 'Region 4', 'Region 7', 'Region 9', 'SC', 'VA', 'VI', 'VT', 'WI', 'WY', 'AK', 'AL', 'AR', 'CO', 'FL', 'GA', 'IA', 'IN', 'KS', 'MA', 'MP', 'MS', 'ND', 'NM', 'NY', 'OH', 'OK', 'PA', 'PR', 'Region 10', 'Region 2', 'Region 3', 'Region 5', 'Region 6', 'Region 8', 'SD', 'TN', 'TX', 'US', 'WA', 'WV', 'HI', 'UT','admissions_18_29_covid_confirmed','admissions_40_49_covid_confirmed','admissions_60_69_covid_confirmed','admissions_30_49_covid_confirmed','admissions_50_69_covid_confirmed','admissions_20_29_covid_confirmed','admissions_0_17_covid_confirmed','admissions_70_covid_confirmed','admissions_50_59_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','inpatient_beds','total_icu_beds','total_admissions_all_influenza_confirmed_past_7days','percent_inpatient_beds_influenza']].interpolate().reset_index(drop=False).sort_values(by='index'), time_col='index', value_cols=['AZ', 'CA', 'CT', 'DC', 'DE', 'ID', 'IL', 'KY', 'LA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MT', 'NC', 'NE', 'NH', 'NJ', 'NV', 'OR', 'RI', 'Region 1', 'Region 4', 'Region 7', 'Region 9', 'SC', 'VA', 'VI', 'VT', 'WI', 'WY', 'AK', 'AL', 'AR', 'CO', 'FL', 'GA', 'IA', 'IN', 'KS', 'MA', 'MP', 'MS', 'ND', 'NM', 'NY', 'OH', 'OK', 'PA', 'PR', 'Region 10', 'Region 2', 'Region 3', 'Region 5', 'Region 6', 'Region 8', 'SD', 'TN', 'TX', 'US', 'WA', 'WV', 'HI', 'UT','admissions_18_29_covid_confirmed','admissions_40_49_covid_confirmed','admissions_60_69_covid_confirmed','admissions_30_49_covid_confirmed','admissions_50_69_covid_confirmed','admissions_20_29_covid_confirmed','admissions_0_17_covid_confirmed','admissions_70_covid_confirmed','admissions_50_59_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','inpatient_beds','total_icu_beds','total_admissions_all_influenza_confirmed_past_7days','percent_inpatient_beds_influenza'])
    #exog = TimeSeries.from_dataframe(data2[['number_hospitals_reporting_today', 'inpatient_beds', 'total_icu_beds', 'total_staffed_adult_icu_beds', 'total_staffed_pediatric_icu_beds','inpatient_adult_beds','average_percent_adult_inpatient_beds_covid','percent_adult_inpatient_beds_occupied','percent_inpatient_beds_covid','percent_staff_icu_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend']].interpolate().reset_index(drop=False).sort_values(by='index'),time_col='index',value_cols=['number_hospitals_reporting_today', 'inpatient_beds', 'total_icu_beds', 'total_staffed_adult_icu_beds', 'total_staffed_pediatric_icu_beds','inpatient_adult_beds','average_percent_adult_inpatient_beds_covid','percent_adult_inpatient_beds_occupied','percent_inpatient_beds_covid','percent_staff_icu_beds_covid','average_percent_adult_inpatient_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend'])
    #exog = TimeSeries.from_dataframe(data2[['average_percent_staff_adult_icu_beds_occupied','percent_pediatric_inpatient_beds_occupied','average_percent_adult_inpatient_beds_covid','absolute_change_average_percent_inpatient_beds_occupied','total_admissions_all_influenza_confirmed_past_7days_per_100k','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend']].interpolate().reset_index(drop=False).sort_values(by='index'),time_col='index',value_cols=['average_percent_staff_adult_icu_beds_occupied','percent_pediatric_inpatient_beds_occupied','average_percent_adult_inpatient_beds_covid','absolute_change_average_percent_inpatient_beds_occupied','total_admissions_all_influenza_confirmed_past_7days_per_100k','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend'])
    #exog = TimeSeries.from_dataframe(data2[['admissions_all_covid_confirmed','admissions_70_covid_confirmed','admissions_30_49_covid_confirmed','admissions_30_39_covid_confirmed','admissions_12_17_covid_confirmed','average_percent_adult_inpatient_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend']].interpolate().reset_index(drop=False).sort_values(by='index'),time_col='index',value_cols=['admissions_all_covid_confirmed','admissions_70_covid_confirmed','admissions_30_49_covid_confirmed','admissions_30_39_covid_confirmed','admissions_12_17_covid_confirmed','average_percent_adult_inpatient_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend'])
    #exog = TimeSeries.from_dataframe(data2[['total_adult_patients_hospitalized_covid_confirmed','percent_inpatient_beds_covid','percent_adult_inpatient_beds_covid','average_admissions_70_covid_confirmed','average_admissions_70_covid_confirmed_per_100k','average_percent_adult_inpatient_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend']].interpolate().reset_index(drop=False).sort_values(by='index'),time_col='index',value_cols=['total_adult_patients_hospitalized_covid_confirmed','percent_inpatient_beds_covid','percent_adult_inpatient_beds_covid','average_admissions_70_covid_confirmed','average_admissions_70_covid_confirmed_per_100k','average_percent_adult_inpatient_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend'])
    #exog = TimeSeries.from_dataframe(data2[['percent_inpatient_beds_influenza','total_admissions_all_influenza_confirmed_past_7days_per_100k','total_admissions_all_influenza_confirmed_past_7days','absolute_change_average_percent_staff_icu_beds_influenza','icu_patients_influenza_confirmed','average_admissions_18_29_covid_confirmed_per_100k','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend']].interpolate().reset_index(drop=False).sort_values(by='index'),time_col='index',value_cols=['percent_inpatient_beds_influenza','total_admissions_all_influenza_confirmed_past_7days_per_100k','total_admissions_all_influenza_confirmed_past_7days','absolute_change_average_percent_staff_icu_beds_influenza','icu_patients_influenza_confirmed','average_admissions_18_29_covid_confirmed_per_100k','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend'])
    #exog = TimeSeries.from_dataframe(data2[['total_admissions_all_covid_confirmed_past_7days_per_100k','average_admissions_all_covid_confirmed','average_admissions_all_covid_confirmed_per_100k','average_admissions_70_covid_confirmed_per_100k','average_admissions_70_covid_confirmed','average_admissions_30_49_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend']].interpolate().reset_index(drop=False).sort_values(by='index'),time_col='index',value_cols=['total_admissions_all_covid_confirmed_past_7days_per_100k','average_admissions_all_covid_confirmed','average_admissions_all_covid_confirmed_per_100k','average_admissions_70_covid_confirmed_per_100k','average_admissions_70_covid_confirmed','average_admissions_30_49_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend'])
    #exog = TimeSeries.from_dataframe(data2[['total_admissions_all_influenza_confirmed_past_7days_per_100k','absolute_change_average_percent_inpatient_beds_covid','percent_inpatient_beds_influenza','absolute_change_average_percent_staff_icu_beds_influenza','total_patients_hospitalized_influenza_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend']].interpolate().reset_index(drop=False).sort_values(by='index'),time_col='index',value_cols=['total_admissions_all_influenza_confirmed_past_7days_per_100k','absolute_change_average_percent_inpatient_beds_covid','percent_inpatient_beds_influenza','absolute_change_average_percent_staff_icu_beds_influenza','total_patients_hospitalized_influenza_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend'])
    #exog = TimeSeries.from_dataframe(data2[['total_adult_patients_hospitalized_covid_confirmed','total_patients_hospitalized_covid_confirmed','proportion_prevalent_inpatient_beds','percent_adult_inpatient_beds_covid','percent_inpatient_beds_covid','average_admissions_60_69_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','year','is_weekend']].interpolate().reset_index(drop=False).sort_values(by='index'),time_col='index',value_cols=['total_adult_patients_hospitalized_covid_confirmed','total_patients_hospitalized_covid_confirmed','proportion_prevalent_inpatient_beds','percent_adult_inpatient_beds_covid','percent_inpatient_beds_covid','average_admissions_60_69_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','year','is_weekend'])
    
    exog = TimeSeries.from_dataframe(data2[['admissions_18_29_covid_confirmed','admissions_30_39_covid_confirmed','admissions_40_49_covid_confirmed','admissions_60_69_covid_confirmed','admissions_30_49_covid_confirmed','admissions_50_69_covid_confirmed','admissions_20_29_covid_confirmed','admissions_0_17_covid_confirmed','admissions_70_covid_confirmed','admissions_50_59_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','year','is_weekend','jurisdiction_target_encoded']].interpolate().reset_index(drop=False).sort_values(by='index'),time_col='index',value_cols=['admissions_18_29_covid_confirmed','admissions_30_39_covid_confirmed','admissions_40_49_covid_confirmed','admissions_60_69_covid_confirmed','admissions_30_49_covid_confirmed','admissions_50_69_covid_confirmed','admissions_20_29_covid_confirmed','admissions_0_17_covid_confirmed','admissions_70_covid_confirmed','admissions_50_59_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','year','is_weekend','jurisdiction_target_encoded'])
         
    print("TimeSeries 'y':")
    print(y)
    
    # Generate the forecasts
    print("Generating forecasts...")
    model = None
    if model_name == "prophet":
    #yearly_seasonality=True,weekly_seasonality=True,seasonality_prior_scale=10.0,holidays_prior_scale=10.0,weekly_seasonality=True,growth='linear'
        model = Prophet(seasonality_mode='additive',daily_seasonality=True)
        model.fit(y,future_covariates=exog)
       
    elif model_name == "sarimax":
        model = SARIMAX(y,order=(1,0,0),seasonal_order=(0,1,1,7))
    elif model_name == "lgbm":
        model = LightGBMModel(lags=12,verbose=-1)
        #model.fit(y,exog=exog)
    elif model_name == "ar":
        model = ARIMA(p=1,d=0,q=1,seasonal_order=(0,1,1,7))
        #model = ARIMA(p=1,d=0,q=1,seasonal_order=(0,1,3,7),)
        model.fit(series=y,future_covariates=exog)
    elif model_name == "holtwinter":
         model = ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.ADDITIVE,seasonal_periods=12)
         #model.fit(y,future_covariates=exog)
    elif model_name == "theta":
        if fit_every > 1:
            raise ValueError("Theta does not support fit_every > 1")
        model = Theta()
    elif model_name == "transformer":
        model = TransformerModel(12,n_epochs=10)
        os.system("export PYTORCH_ENABLE_MPS_FALLBACK=1") # WARNING: This doesn't always work. If not, make sure to execute on your system to use the transformer architecture.
        y = y.astype(np.float32)
    else:
        raise ValueError("Invalid model name")
        
    
    # Ignore ConvergenceWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    retrain = fit_every if model_name != "transformer" else fit_every * 100  
    # Forecast future values
    #,future_covariates=exog
    model_forecasts = model.historical_forecasts(y,future_covariates=exog,forecast_horizon=fit_every, retrain=retrain, verbose=True).values()[:,0].squeeze()
    #results = model.fit()
    #forecasts = results.get_forecast(steps=fit_every)
    # Print the shape and values of the forecasts
    print("Shape of forecasts:", model_forecasts.shape)
    print("Model Forecasts:", model_forecasts)
    
    forecasts[-model_forecasts.shape[0]:] = model_forecasts
    #print(forecasts)
    #forecasts[:model_forecasts.shape[0]] = model_forecasts
    # Save and return
    np.savez(savename, forecasts=forecasts)
    return forecasts
