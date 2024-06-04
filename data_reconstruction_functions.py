from dataclasses import dataclass

from electrcity_processing_functions import process_electricity_semi_annual_imputed_df, tax_tier_map

import pandas as pd
from statsmodels.regression.linear_model import OLSResults


@dataclass
class RegressionData():
    y_name: str
    x_name: str
    reg_result: OLSResults


def fill_missing_values(values_df: pd.DataFrame,
                        y_name: str,
                        x_name: str,
                        reg_result: OLSResults,
                        sig_level=0.05):
    """
    :param values_df:
    :param columns_map: mapping from columns with missing data to columns that have samples for regression
    :param reg_result:
    :param sig_level:
    :return: returns False for an insignificant model (intercept or slope insignificant)
    """
    # use only columns with same name for oos
    if reg_result.pvalues[0] > sig_level or reg_result.pvalues[1] > sig_level:
        # we would rather not use this model for regression
        return False
    for index, row in values_df[values_df[y_name].isna()].iterrows():
        # Apply the regression function to the corresponding x_name value
        reg_function = lambda x: reg_result.params[0] + x * reg_result.params[1]
        values_df.loc[index, y_name] = reg_function(row[x_name])
    return True


def format_and_calculate_electricity_df(pre_2007_imputed_df: pd.DataFrame, post_2007_df: pd.DataFrame):
    """

    :param pre_2007_imputed_df:
    :param post_2007_df:
    :return:
    """

    pre_2007_imputed_cols = pre_2007_imputed_df.columns.values.tolist()

    pre_2007_imputed_ggdp_format = pd.melt(pre_2007_imputed_df, id_vars=['year'],
                                             value_vars=pre_2007_imputed_cols, var_name='tax_tier',
                                             value_name='price')
    # pre_2007_imputed_ggdp_format = pre_2007_imputed_ggdp_format.replace(tax_tier_map)
    pre_2007_imputed_ggdp_format['tax_tier'] = pre_2007_imputed_ggdp_format['tax_tier'].apply(lambda x: float(x))
    pre_2007_imputed_ggdp_format['currency'] = 'EUR'
    pre_2007_imputed_ggdp_format_processed = process_electricity_semi_annual_imputed_df(
        pre_2007_imputed_ggdp_format)

    # deleting overlap in pre-2007 data with post_2007 data
    # there's overlap with the two dataframe: both have 2007 data
    imputed_index_to_drop = pre_2007_imputed_ggdp_format_processed[
        pre_2007_imputed_ggdp_format_processed['year'] == '2007'].index
    pre_2007_imputed_ggdp_format_processed = pre_2007_imputed_ggdp_format_processed.drop(
        index=imputed_index_to_drop)

    # maybe add a way to check overlap of dataframes
    # overlapping_rows_after_drop = pd.merge(elec_hh_post_2007_replaced, fi_hh_pre_2007_imputed_ggdp_format_processed,
    #                                        on=['year', 'tax_tier'], how='inner')

    elec_imputed = pd.concat([pre_2007_imputed_ggdp_format_processed, post_2007_df])
    elec_imputed = elec_imputed.replace(tax_tier_map)
    elec_imputed['year'] = elec_imputed['year'].apply(lambda x: float(x))
    elec_imputed['tax_tier'] = elec_imputed['tax_tier'].apply(lambda x: float(x))
    # droping any rows that having missing values because they won't help calculate prices
    return elec_imputed.dropna(axis=0, how='any')
