import pandas as pd
from dataclasses import dataclass
from statsmodels.regression.linear_model import OLSResults


@dataclass
class RegressionData():
    y_name: str
    x_name: str
    reg_result: OLSResults


def fill_missing_values(values_df: pd.DataFrame,
                        y_name: str,
                        x_name: str,
                        reg_result:OLSResults,
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