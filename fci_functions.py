from typing import List, Optional
import functools

import pandas as pd
from sklearn.decomposition import PCA

pca = PCA(n_components=1)


def read_and_process_data(path: str, date_column_name: str, price_column_name: str, rename_map: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    processed_data: pd.DataFrame = process_data_for_fci(df, date_column_name, price_column_name)
    return processed_data.rename(columns=rename_map)


def process_data_for_fci(df: pd.DataFrame, date_column_name: str, price_column_name: str) -> pd.DataFrame:
    df['year'] = pd.to_datetime(df[date_column_name])
    df[price_column_name] = pd.to_numeric(df[price_column_name])
    agg_df = df.groupby(df['year'].dt.year)[price_column_name].mean().to_frame()
    agg_df = agg_df.rename(columns={price_column_name: f'price'})
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
    df_1: pd.DataFrame
    df_2: pd.DataFrame
    return functools.reduce(lambda df_1, df_2: df_1.join(df_2), df_list)


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
