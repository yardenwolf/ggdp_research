from typing import List

import pandas as pd

def get_summary_statistics(df: pd.DataFrame, columns:List[str]) -> pd.DataFrame:
    """
    get summary statistics for columns specified. Each column is a collection of samples
    :param df:
    :param columns:
    :return:
    """
    list_of_summary_df = []
    for column_name in columns:
        curr_summary_stats_df = df[column_name].describe()
        curr_summary_stats_df['variance'] = df[column_name].var()
        curr_summary_stats_df['skewness'] = df[column_name].skew()
        curr_summary_stats_df['kurtosis'] = df[column_name].kurtosis()
        list_of_summary_df.append(curr_summary_stats_df)

    return pd.concat(list_of_summary_df, axis=1)