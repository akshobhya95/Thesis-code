import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import pandas as pd
import pdb

def load_dataset(name):
    if name == "daily-climate":
        df = pd.read_csv('./datasets/daily-climate.csv')
        df.rename({'date': 'timestamp', 'meantemp': 'y'}, axis='columns', inplace=True)
        df = df.drop("Unnamed: 0", axis='columns')
        data = df.melt(id_vars=['timestamp'], value_name='target')
        data.rename({'variable': 'item_id'}, axis='columns', inplace=True)
    if name == "MSFT":
        df = pd.read_csv('./datasets/djia.csv')
        df = df.drop(["High", "Low", "Close", "Volume"], axis='columns')
        df.rename({'Date': 'timestamp', 'Name': 'item_id'}, axis='columns', inplace=True)
        df.replace({'item_id': {'MSFT': 'y'}}, inplace=True)
        data = df.melt(id_vars=['timestamp', 'item_id'], value_name='target')
        data.drop("variable", axis='columns', inplace=True)
    if name == "RQ1":
        df = pd.read_csv('./datasets/Region.csv')
        #columns_to_keep = [col for col in df.columns if col not in ['jurisdiction', 'state']]
        #df = df[columns_to_keep]
        #df.rename({'collection_date': 'timestamp', 'total_patients_hospitalized_covid_confirmed': 'y'}, axis='columns', inplace=True)
        #value_vars = [col for col in df.columns if col not in ['timestamp']]
        #data = df.melt(id_vars=['timestamp'], value_vars=value_vars, var_name='item_id', value_name='target')
        #data = df.melt(id_vars=['timestamp'], value_name='target')
        #data.rename({'variable': 'item_id'}, axis='columns', inplace=True)
        #df = df[df['jurisdiction'] == 'CA']
        
        #df = df[['collection_date', 'total_patients_hospitalized_covid_confirmed']]
        df.rename({'collection_date': 'timestamp', 'percent_inpatient_beds_occupied': 'y'}, axis='columns', inplace=True)
        #df.rename({'collection_date': 'timestamp', 'percent_staff_icu_beds_occupied': 'y'}, axis='columns', inplace=True)
        #df.rename({'collection_date': 'timestamp', 'proportion_inpatient_beds_covid_vs_normal': 'y'}, axis='columns', inplace=True)
        #df.rename({'collection_date': 'timestamp', 'admissions_all_covid_confirmed': 'y'}, axis='columns', inplace=True)
        #df.rename({'collection_date': 'timestamp', 'total_patients_hospitalized_influenza_confirmed': 'y'}, axis='columns', inplace=True)
        #df.rename({'collection_date': 'timestamp', 'total_admissions_all_covid_confirmed_past_7days': 'y'}, axis='columns', inplace=True)
        #df.rename({'collection_date': 'timestamp', 'total_admissions_all_influenza_confirmed_past_7days': 'y'}, axis='columns', inplace=True)
        #exog_cols = ['timestamp','y','total_adult_patients_hospitalized_covid_confirmed','percent_adult_inpatient_beds_covid','percent_inpatient_beds_covid','total_staffed_pediatric_icu_beds', 'staff_pediatric_icu_beds_occupied', 'inpatient_pediatric_beds_used']
        #exog_cols = ['timestamp','y','AZ', 'CA', 'CT', 'DC', 'DE', 'ID', 'IL', 'KY', 'LA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MT', 'NC', 'NE', 'NH', 'NJ', 'NV', 'OR', 'RI', 'Region 1', 'Region 4', 'Region 7', 'Region 9', 'SC', 'VA', 'VI', 'VT', 'WI', 'WY', 'AK', 'AL', 'AR', 'CO', 'FL', 'GA', 'IA', 'IN', 'KS', 'MA', 'MP', 'MS', 'ND', 'NM', 'NY', 'OH', 'OK', 'PA', 'PR', 'Region 10', 'Region 2', 'Region 3', 'Region 5', 'Region 6', 'Region 8', 'SD', 'TN', 'TX', 'US', 'WA', 'WV', 'HI', 'UT','admissions_18_29_covid_confirmed','admissions_40_49_covid_confirmed','admissions_60_69_covid_confirmed','admissions_30_49_covid_confirmed','admissions_50_69_covid_confirmed','admissions_20_29_covid_confirmed','admissions_0_17_covid_confirmed','admissions_70_covid_confirmed','admissions_50_59_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','inpatient_beds','total_icu_beds','total_admissions_all_influenza_confirmed_past_7days','percent_inpatient_beds_influenza']
        exog_cols = ['timestamp','number_hospitals_reporting_today', 'inpatient_beds', 'total_icu_beds', 'total_staffed_adult_icu_beds', 'total_staffed_pediatric_icu_beds','inpatient_adult_beds','percent_adult_inpatient_beds_occupied','percent_inpatient_beds_covid','percent_staff_icu_beds_covid','average_percent_adult_inpatient_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y']
        #exog_cols = ['timestamp','average_percent_staff_adult_icu_beds_occupied','percent_pediatric_inpatient_beds_occupied','average_percent_adult_inpatient_beds_covid','absolute_change_average_percent_inpatient_beds_occupied','total_admissions_all_influenza_confirmed_past_7days_per_100k','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y']
        #exog_cols = ['timestamp','admissions_all_covid_confirmed','admissions_70_covid_confirmed','admissions_30_49_covid_confirmed','admissions_30_39_covid_confirmed','admissions_12_17_covid_confirmed','average_percent_adult_inpatient_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y']
        #Prevalent admissions
        #exog_cols = ['timestamp','total_adult_patients_hospitalized_covid_confirmed','percent_inpatient_beds_covid','percent_adult_inpatient_beds_covid','average_admissions_70_covid_confirmed','average_admissions_70_covid_confirmed_per_100k','average_percent_adult_inpatient_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y']
        #exog_cols = ['timestamp','percent_inpatient_beds_influenza','total_admissions_all_influenza_confirmed_past_7days_per_100k','total_admissions_all_influenza_confirmed_past_7days','absolute_change_average_percent_staff_icu_beds_influenza','icu_patients_influenza_confirmed','average_admissions_18_29_covid_confirmed_per_100k','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y']
        #Newer admissions past 7 days total---> SARIMA Failed ue to LU decomposition error, Prophet passed.
        #exog_cols = ['timestamp','total_admissions_all_covid_confirmed_past_7days_per_100k','average_admissions_all_covid_confirmed','average_admissions_all_covid_confirmed_per_100k','average_admissions_70_covid_confirmed_per_100k','average_admissions_70_covid_confirmed','average_admissions_30_49_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y']
        #exog_cols = ['timestamp','total_admissions_all_influenza_confirmed_past_7days_per_100k','absolute_change_average_percent_inpatient_beds_covid','percent_inpatient_beds_influenza','absolute_change_average_percent_staff_icu_beds_influenza','total_patients_hospitalized_influenza_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y']
        
        #exog_cols = ['timestamp','total_adult_patients_hospitalized_covid_confirmed','total_patients_hospitalized_covid_confirmed','proportion_prevalent_inpatient_beds','percent_adult_inpatient_beds_covid','percent_inpatient_beds_covid','average_admissions_60_69_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','year','is_weekend','y']
        
        #exog_cols = ['timestamp','admissions_18_29_covid_confirmed','admissions_30_39_covid_confirmed','admissions_40_49_covid_confirmed','admissions_60_69_covid_confirmed','admissions_30_49_covid_confirmed','admissions_50_69_covid_confirmed','admissions_20_29_covid_confirmed','admissions_0_17_covid_confirmed','admissions_70_covid_confirmed','admissions_50_59_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','year','is_weekend','jurisdiction','jurisdiction_target_encoded','y']

        df = df[exog_cols]
        #data = df.melt(id_vars=['timestamp'], value_name='target')
        #data = df.melt(id_vars=['timestamp','jurisdiction'], value_name='target')
        #ReserachQuestion1:-How do hospital admission rates for COVID-19 cases vary across different age groups and geographic jurisdictions, considering temporal patterns and seasonal variations?
        #data = df.melt(id_vars=['timestamp'], value_vars=['AZ', 'CA', 'CT', 'DC', 'DE', 'ID', 'IL', 'KY', 'LA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MT', 'NC', 'NE', 'NH', 'NJ', 'NV', 'OR', 'RI', 'Region 1', 'Region 4', 'Region 7', 'Region 9', 'SC', 'VA', 'VI', 'VT', 'WI', 'WY', 'AK', 'AL', 'AR', 'CO', 'FL', 'GA', 'IA', 'IN', 'KS', 'MA', 'MP', 'MS', 'ND', 'NM', 'NY', 'OH', 'OK', 'PA', 'PR', 'Region 10', 'Region 2', 'Region 3', 'Region 5', 'Region 6', 'Region 8', 'SD', 'TN', 'TX', 'US', 'WA', 'WV', 'HI', 'UT','admissions_18_29_covid_confirmed','admissions_40_49_covid_confirmed','admissions_60_69_covid_confirmed','admissions_30_49_covid_confirmed','admissions_50_69_covid_confirmed','admissions_20_29_covid_confirmed','admissions_0_17_covid_confirmed','admissions_70_covid_confirmed','admissions_50_59_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','inpatient_beds','total_icu_beds','total_admissions_all_influenza_confirmed_past_7days','percent_inpatient_beds_influenza','y'], value_name='target')
        #ReserachQuestion2 :- How do bed occupancy rates for inpatient and ICU beds change over time, especially during COVID-19 outbreaks, and what proportion of inpatient beds are taken by COVID-19 patients compared to all patients?
        data = df.melt(id_vars=['timestamp'], value_vars=['number_hospitals_reporting_today', 'inpatient_beds', 'total_icu_beds', 'total_staffed_adult_icu_beds', 'total_staffed_pediatric_icu_beds','inpatient_adult_beds','percent_adult_inpatient_beds_occupied','percent_inpatient_beds_covid','percent_staff_icu_beds_covid','average_percent_adult_inpatient_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y'], value_name='target')
        #data = df.melt(id_vars=['timestamp'], value_vars=['average_percent_staff_adult_icu_beds_occupied','percent_pediatric_inpatient_beds_occupied','average_percent_adult_inpatient_beds_covid','absolute_change_average_percent_inpatient_beds_occupied','total_admissions_all_influenza_confirmed_past_7days_per_100k','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y'], value_name='target')
        #data = df.melt(id_vars=['timestamp'], value_vars=['total_adult_patients_hospitalized_covid_confirmed','total_patients_hospitalized_covid_confirmed','proportion_prevalent_inpatient_beds','percent_adult_inpatient_beds_covid','percent_inpatient_beds_covid','average_admissions_60_69_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','year','is_weekend','y'],value_name='target')
   
        #data = df.melt(id_vars=['timestamp'], value_vars=['admissions_all_covid_confirmed','admissions_70_covid_confirmed','admissions_30_49_covid_confirmed','admissions_30_39_covid_confirmed','admissions_12_17_covid_confirmed','average_percent_adult_inpatient_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y'],value_name='target')
        #ReserachQuestion3:- How does the rate of hospitalization for COVID-19 patients compare to that of patients with confirmed influenza, and what does this reveal about the strain on healthcare systems?
        #Prevalent admissions
        #data = df.melt(id_vars=['timestamp'], value_vars=['total_adult_patients_hospitalized_covid_confirmed','percent_inpatient_beds_covid','percent_adult_inpatient_beds_covid','average_admissions_70_covid_confirmed','average_admissions_70_covid_confirmed_per_100k','average_percent_adult_inpatient_beds_covid','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y'],value_name='target')
        #data = df.melt(id_vars=['timestamp'], value_vars=['percent_inpatient_beds_influenza','total_admissions_all_influenza_confirmed_past_7days_per_100k','total_admissions_all_influenza_confirmed_past_7days','absolute_change_average_percent_staff_icu_beds_influenza','icu_patients_influenza_confirmed','average_admissions_18_29_covid_confirmed_per_100k','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y'],value_name='target')
        #Newer admissions past 7 days total
        #data = df.melt(id_vars=['timestamp'], value_vars=['total_admissions_all_covid_confirmed_past_7days_per_100k','average_admissions_all_covid_confirmed','average_admissions_all_covid_confirmed_per_100k','average_admissions_70_covid_confirmed_per_100k','average_admissions_70_covid_confirmed','average_admissions_30_49_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y'],value_name='target')
        #data = df.melt(id_vars=['timestamp'], value_vars=['total_admissions_all_influenza_confirmed_past_7days_per_100k','absolute_change_average_percent_inpatient_beds_covid','percent_inpatient_beds_influenza','absolute_change_average_percent_staff_icu_beds_influenza','total_patients_hospitalized_influenza_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','is_weekend','y'],value_name='target')
        
        #ReserachQuestion1
        data = df.melt(id_vars=['timestamp','jurisdiction'], value_vars=['admissions_18_29_covid_confirmed','admissions_30_39_covid_confirmed','admissions_40_49_covid_confirmed','admissions_60_69_covid_confirmed','admissions_30_49_covid_confirmed','admissions_50_69_covid_confirmed','admissions_20_29_covid_confirmed','admissions_0_17_covid_confirmed','admissions_70_covid_confirmed','admissions_50_59_covid_confirmed','peak_flu_season','is_holiday?','quarter','day_of_year_sin','day_of_year_cos','month_sin','month_cos','day_of_month_sin','day_of_month_cos','day_of_week_sin','day_of_week_cos','season','year','is_weekend','jurisdiction_target_encoded','y'], value_name='target')
        data.rename({'variable': 'item_id'}, axis='columns', inplace=True)
    if name == "AMZN":
        df = pd.read_csv('./datasets/djia.csv')
        df = df.drop(["High", "Low", "Close", "Volume"], axis='columns')
        df.rename({'Date': 'timestamp', 'Name': 'item_id'}, axis='columns', inplace=True)
        df.replace({'item_id': {'AMZN': 'y'}}, inplace=True)
        data = df.melt(id_vars=['timestamp', 'item_id'], value_name='target')
        data.drop("variable", axis='columns', inplace=True)
    if name == "GOOGL":
        df = pd.read_csv('./datasets/djia.csv')
        df = df.drop(["High", "Low", "Close", "Volume"], axis='columns')
        df.rename({'Date': 'timestamp', 'Name': 'item_id'}, axis='columns', inplace=True)
        df.replace({'item_id': {'GOOGL': 'y'}}, inplace=True)
        data = df.melt(id_vars=['timestamp', 'item_id'], value_name='target')
        data.drop("variable", axis='columns', inplace=True)
    if name == "COVID-deaths4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
    if name == "tx-COVID-deaths-4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/statewide/tx_proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
        data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    if name == "ca-COVID-deaths-4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/statewide/ca_proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
        data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    if name == "ga-COVID-deaths-4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/statewide/ga_proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
        data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    if name == "fl-COVID-deaths-4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/statewide/fl_proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
        data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    if name == "ny-COVID-deaths-4wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/statewide/ny_proc_4wkdeaths.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
        data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    if name == "COVID-cases3wk":
        df = pd.read_pickle('./datasets/covid-ts-proc/proc_3wkcases.pkl')
        df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
        data = df
    if name == "elec2":
        df = pd.read_csv('./datasets/elec2.csv')
        df['timestamp'] = pd.date_range(start='1996-5-7', end='1998-12-6 23:30:00', freq='30T', inclusive='both')
        df['class'] = (df['class'] == 'UP').astype(float)
        df.rename({'nswdemand': 'y'}, axis='columns', inplace=True)
        df = df[:2000]
        data = df.melt(id_vars=['timestamp'], value_name='target')
        data.rename({'variable': 'item_id'}, axis='columns', inplace=True)
        data.astype({'target': 'float64'})
    if name == "M4":
        data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d-%m-%Y')
    #data = data.pivot(columns="item_id", index="timestamp", values="target")
    data = data.pivot_table(columns="item_id", index=["timestamp", "jurisdiction"], values="target", aggfunc='sum')
    data['y'] = data['y'].astype(float)
    data = data.interpolate()
    #data.index = pd.to_datetime(data.index)
    data.index = pd.to_datetime([index_tuple[0] for index_tuple in data.index])
    return data

if __name__ == "__main__":
    # Iterate through all the datasets and attempt loading them
    datasets = ['RQ1']#,'tx-COVID-deaths-4wk', 'ca-COVID-deaths-4wk']
    #datasets = ['PFINEW']#,'tx-COVID-deaths-4wk', 'ca-COVID-deaths-4wk']
    for dataset in datasets:
        print(f"Loading {dataset} dataset")
        data = load_dataset(dataset)
        print(f"Loaded {dataset} dataset")
        print(data.columns)
