from math import inf
from datetime import datetime

import pandas as pd

# code to consumption tier. consumption tier in kWh
tax_tier_map_pre_2007_hh = {'4161050': 600, '4161100': 1200, '4161150': 3500, '4161200': 7500, '4161250': inf}

tax_tier_map_post_2007_hh = {'KWH_LT1000': 1000, 'KWH1000-2499': 2500, 'KWH2500-4999': 5000, 'KWH5000-14999': 15000,
                             'KWH_GE15000': inf}

tax_tier_map_post_2007_nh = {'MWH_LT20': 20000, 'MWH20-499': 500000, 'MWH500-1999': 2000000, 'MWH2000-19999': 20000000,
                             'MWH20000-69999': 70000000, 'MWH70000-149999': 150000000, 'MWH_GE150000': inf}

tax_tier_map_pre_2007_nh = {'4162050': 30000, '4162100': 50000, '4162150': 160000, '4162200': 1250000,
                            '4162250': 2000000, '4162300': 10000000, '4162350': 24000000, '4162400': 50000000,
                            '4162450': inf}
tax_tier_map = {**tax_tier_map_pre_2007_hh, **tax_tier_map_post_2007_hh, **tax_tier_map_post_2007_nh,
                **tax_tier_map_pre_2007_nh}


def process_electricity_df(country_code: str, elec_df: pd.DataFrame, with_tax=True) -> pd.DataFrame:
    tax_filter = 'I_TAX' if with_tax else 'X_TAX'
    filtered_df: pd.DataFrame = elec_df[(elec_df['geo'] == country_code) & (elec_df['tax'] == tax_filter)]
    filtered_df.loc[:, 'year'] = filtered_df['TIME_PERIOD'].apply(lambda x: x[:x.find('-')])
    if 'consom' in filtered_df:
        filtered_df = filtered_df.rename(columns={'consom': 'nrg_cons'})
    filtered_df = filtered_df.assign(tax_tier=filtered_df.loc[:, 'nrg_cons'].apply(str))
    filtered_df = filtered_df.drop(columns=['nrg_cons', 'OBS_FLAG', 'DATAFLOW', 'LAST UPDATE', 'freq', 'product'])
    filtered_df = filtered_df.rename(columns={'OBS_VALUE': 'price'})
    filtered_df = filtered_df.sort_values(by=['currency', 'tax_tier', 'year'])
    # need to finish fillna
    filtered_df['price'] = filtered_df.loc[:, 'price'].fillna(method='ffill')
    annual_data = filtered_df.groupby(['currency', 'tax_tier', 'year'], as_index=False)['price'].mean()
    # annual_data = annual_data.rename(columns={'OBS_VALUE': 'price', 'TIME_PERIOD': 'year'})
    tot_kwh_indices = annual_data.index[annual_data['tax_tier'] == 'TOT_KWH'].tolist()
    annual_data = annual_data.drop(index=tot_kwh_indices)
    return annual_data


def get_year_and_semester_from_date(date: datetime, sep="-") -> str:
    month = date.strftime(format='%m')
    year = date.strftime('%Y')
    semester = '2'
    if int(month) >= 1 and int(month) <= 6:
        semester = '1'
    return f"{year}{sep}S{semester}"


def process_electricity_df_semi_annually(country_code: str, elec_df: pd.DataFrame, with_tax=True) -> pd.DataFrame:
    tax_filter = 'I_TAX' if with_tax else 'X_TAX'
    filtered_df: pd.DataFrame = elec_df[(elec_df['geo'] == country_code) & (elec_df['tax'] == tax_filter)]
    if 'consom' in filtered_df:
        filtered_df = filtered_df.rename(columns={'consom': 'nrg_cons'})
    filtered_df = filtered_df.assign(tax_tier=filtered_df.loc[:, 'nrg_cons'].apply(str))
    filtered_df = filtered_df.drop(columns=['nrg_cons', 'OBS_FLAG', 'DATAFLOW', 'LAST UPDATE', 'freq', 'product'])
    filtered_df = filtered_df.rename(columns={'OBS_VALUE': 'price'})
    annual_data = filtered_df
    annual_data = annual_data.rename(columns={'TIME_PERIOD': 'semester'})
    annual_data = annual_data.sort_values(by=['currency', 'tax_tier', 'semester'])
    # filtered_df['price'] = filtered_df.loc[:, 'price'].fillna(method='ffill')
    tot_kwh_indices = annual_data.index[annual_data['tax_tier'] == 'TOT_KWH'].tolist()
    annual_data = annual_data.drop(index=tot_kwh_indices)
    return annual_data


def transform_elec_price_to_tax_tier_time_series(elec_df: pd.DataFrame, country_code: str,
                                                 currency: str) -> pd.DataFrame:
    elec_df_semi_annualy = process_electricity_df_semi_annually(country_code=country_code, elec_df=elec_df)
    elec_df_semi_annualy = elec_df_semi_annualy.replace({'tax_tier': tax_tier_map})
    elec_df_semi_annualy.index = elec_df_semi_annualy['semester']
    # first I'll try predicting only using pre-2007 data because of the differences in category
    tax_tiers_time_series = []
    for tax_tier in list(elec_df_semi_annualy['tax_tier'].unique()):
        df_filtered_for_tax_tier = elec_df_semi_annualy[
            (elec_df_semi_annualy['tax_tier'] == tax_tier) & (elec_df_semi_annualy['currency'] == currency)]
        tax_tier_series = df_filtered_for_tax_tier['price']
        tax_tier_series.name = tax_tier
        tax_tiers_time_series.append(tax_tier_series)

    return pd.concat(tax_tiers_time_series, axis=1)


def process_electricity_consumption_df(country_code: str, df: pd.DataFrame) -> pd.DataFrame:
    filtered_df: pd.DataFrame = df[(df['geo'] == country_code)]
    filtered_df = filtered_df.drop(columns=['OBS_FLAG', 'DATAFLOW', 'LAST UPDATE', 'freq'])
    annual_data = {}
    for index, row in filtered_df.iterrows():
        current_year = row['TIME_PERIOD']
        if current_year not in annual_data:
            annual_data[current_year] = {'year': current_year, 'hh_consumption': 0, 'nh_consumption': 0}
        current_year_item = annual_data[current_year]
        if row['nrg_bal'] == 'FC_OTH_HH_E':
            current_year_item['year'] = row['TIME_PERIOD']
            current_year_item['hh_consumption'] = row['OBS_VALUE']
            current_year_item['nh_consumption'] -= row['OBS_VALUE']
        elif row['nrg_bal'] == 'FC':
            current_year_item['nh_consumption'] += row['OBS_VALUE']
    # list of dictionaries of years, each item contains year, hh and nh consumption
    annual_data = [annual_data[key] for key in annual_data]
    return pd.DataFrame(annual_data)
