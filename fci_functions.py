from typing import List, Optional
import functools

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components=1)


def read_and_process_data(path: str, date_column_name: str, price_column_name: str, rename_map: dict,
                          timestamp_format: str = None, freq='year') -> pd.DataFrame:
    df = pd.read_csv(path)
    processed_data: pd.DataFrame = process_data_for_fci(df, date_column_name, price_column_name,
                                                        timestamp_format=timestamp_format, freq=freq)
    return processed_data.rename(columns=rename_map)


def process_data_for_fci(df: pd.DataFrame, date_column_name: str, price_column_name: str,
                         freq: str, timestamp_format: Optional[str] = None) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df[date_column_name], format=timestamp_format)
    df[price_column_name] = pd.to_numeric(df[price_column_name])
    if freq == 'quarter':
        agg_df = df.groupby(df['date'].dt.to_period('Q'))[price_column_name].mean().to_frame()
    else:
        agg_df = df.groupby(df['date'].dt.to_period('Y'))[price_column_name].mean().to_frame()

    agg_df.index = agg_df.index.to_timestamp()
    agg_df = agg_df.rename(columns={price_column_name: f'price'})
    # agg_df['quarter'] = agg_df['date'].dt.quarter
    agg_df['growth'] = agg_df['price'] / agg_df['price'].shift(1) - 1
    # agg_df = agg_df.rename(columns={'growth': f'{prefix_name}_growth', 'price': f'{prefix_name}_price'})
    return agg_df.dropna(axis=0, how='any')


def join_and_normalize_data(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    concatenated_df = pd.concat(df_list, ignore_index=True)
    # pandas applies column wise functions automatically
    return (concatenated_df - concatenated_df.mean()) / concatenated_df.std()


def run_pca_and_choose(df: pd.DataFrame) -> pd.DataFrame:
    normalized_data = (df - df.mean()) / df.std()
    test = first_pc = pca.fit(normalized_data)
    print("1234")


def join_dfs(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    return df_1.join(df_2)


def serial_join(df_list: List[pd.DataFrame]):
    def join_dfs(df_1, df_2) -> pd.DataFrame:
        if isinstance(df_1, pd.Series):
            df_1 = df_1.to_frame()
        if isinstance(df_2, pd.Series):
            df_2 = df_2.to_frame()
        return df_1.join(df_2)

    df_1: pd.DataFrame
    df_2: pd.DataFrame
    # return functools.reduce(lambda df_1, df_2: df_1.join(df_2), df_list)
    return functools.reduce(lambda df_1, df_2: join_dfs(df_1, df_2), df_list)


def create_wa_fci_from_data(joint_df: pd.DataFrame, fci_weights: dict) -> Optional[pd.Series]:
    """

    :param joint_df:
    :param fci_weights:
    :return:
    """
    if not set(fci_weights.keys()).intersection(set(joint_df.columns.values)):
        return None
    total_weight = sum([value for _, value in fci_weights.items()])
    fci_weights_normalized = {key: value / total_weight for key, value in fci_weights.items()}
    fci_df = joint_df.copy()
    fci_df['fci'] = 0
    for factor in fci_weights:
        fci_df['fci'] += fci_weights_normalized[factor] * fci_df[factor]
    return fci_df['fci']


def transform_to_fci(df: pd.DataFrame, fci_name: str = 'fci') -> pd.DataFrame:
    df_normalized = (df - df.mean()) / df.std()
    if isinstance(df, pd.Series):
        return df_normalized.rename(fci_name)
    # calculating the fci using all the data
    fci_data = df_normalized @ pca.fit(df_normalized).components_.transpose()
    fci_data.columns = [fci_name]
    return fci_data


# def normalize_date_index(df: pd.DataFrame, date_format="%Y-%m-%d") -> pd.DataFrame:
#     current_index = df.index
#     if isinstance(df.index, pd.DatetimeIndex):
#         current_index = df.index.strftime(date_format)
#
#     for date in current_index:
#         if date


def Lag(x, n):
    if n == 0:
        return x
    return x.shift(n)


def calculate_ewma_volatility(returns_df: pd.Series, gamma: float):
    returns_df = returns_df - returns_df.mean()
    returns_var = returns_df.apply(lambda x: x ** 2)
    vol_list = [returns_var[0]]
    for i in range(len(returns_df)):
        curr_vol = (1 - gamma) * returns_var[i - 1] + gamma * vol_list[-1]
        vol_list.append(curr_vol)
    vol_series = pd.Series(vol_list[1:]).rename(f"ewma_vol_{returns_df.name}")
    vol_series.index = returns_df.index
    return np.sqrt(vol_series)
