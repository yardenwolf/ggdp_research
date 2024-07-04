from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np


@dataclass
class QuantileRegRes():
    q: float
    a: float
    b: float
    b_ci: List[float]

    def to_list(self) -> list:
        return [self.q, self.a, self.b, *self.b_ci]

    def predict(self, x: float) -> float:
        return self.a + self.b * x


def Lag(x, n):
    if n == 0:
        return x
    return x.shift(n)


def fit_qr(q: float, formula: str, data: pd.DataFrame) -> QuantileRegRes:
    """
    :param q:
    :param formula:
    :param data:
    :return:
    """
    mod = smf.quantreg(formula, data=data, missing='drop')
    res = mod.fit(q=q)
    print(f"quantile: {q}")
    print(res.summary())
    param_names = res.params.index.to_list()
    return QuantileRegRes(q=q,
                          a=res.params['Intercept'],
                          b=res.params[param_names[1]],
                          b_ci=res.conf_int().loc[param_names[1]].to_list())


def estimate_quantiles(quantiles: np.ndarray, formula: str, data: pd.DataFrame) -> Tuple[List[QuantileRegRes], dict]:
    qr_res_list = [fit_qr(q, formula, data) for q in quantiles]
    models_summary_df = pd.DataFrame([res.to_list() for res in qr_res_list], columns=['q', 'a', 'b', 'lb', 'ub'])

    # OLS prediction as benchmark
    ols_model = smf.ols(formula, data=data, missing='drop').fit()
    ols_param_names = ols_model.params.index.to_list()
    ols_ci = ols_model.conf_int().loc[ols_param_names[1]].to_list()
    ols_model_summary = dict(a=ols_model.params[0], b=ols_model.params[1], lb=ols_ci[0], ub=ols_ci[1])
    return qr_res_list, ols_model_summary


def predict_quantiles(qr_results_list: List[QuantileRegRes], lag: int, quantiles_to_predict: List[float],
                      data: pd.DataFrame, regressor_name: str, include_ols: bool = True,
                      ols_model_summary: dict = None) -> pd.DataFrame:
    #### predict using the data
    q_pred_list = []
    # each model is for a specific quantile
    # quantiles_to_predict = [0.05, 0.25, 0.5, 0.75, 0.95]
    qr_res_for_pred = [res for res in qr_results_list if res.q in quantiles_to_predict]
    # q_predict = data[['double_pca_fci', 'quarterly_ggdp_ppp_growth_annualized']]
    q_predict = data[[regressor_name]]
    q_predict = q_predict.rename(columns={'quarterly_ggdp_ppp_growth_annualized': 'realized_ggdp'})
    for qr_model in qr_res_for_pred:
        q_predict[qr_model.q] = q_predict[regressor_name].apply(lambda x: qr_model.predict(x))

    if include_ols and ols_model_summary:
        # adding an OLS prediction, OLS predicts the mean
        q_predict['ols'] = q_predict['double_pca_fci'].apply(
            lambda x: ols_model_summary['a'] + x * ols_model_summary['b'])

    # q_predict['realized_ggdp'] = q_predict['realized_ggdp'].shift(-lag)
    q_predict.index = q_predict.index.shift(lag, freq='QS')
    # q_predict = q_predict.dropna(axis=0, how='any')
    return q_predict