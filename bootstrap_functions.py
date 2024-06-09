from typing import Callable
from math import ceil

import pandas as pd
import numpy as np



class MovingBlockBootstrap:

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size

    def _get_bootstrap_sample(self, sample_data: pd.DataFrame) -> pd.DataFrame:
        """
        #assuming all the data we want to bootstrap is in the columns. each column is a time series.
        :param sample_data:
        :param block_size:
        :return:
        """
        m = ceil(len(sample_data) / self.block_size)
        start_index_list = [i for i in range(0, len(sample_data), self.block_size)]
        chosen_blocks_start_index = np.random.choice(start_index_list, size=m)
        chosen_blocks = [sample_data.iloc[i:i + self.block_size, ] for i in chosen_blocks_start_index]
        return pd.concat(chosen_blocks, ignore_index=True)

    def bootstrap(self, sample_data: pd.DataFrame, reps: int, statistic_func: Callable, **kwargs) -> list:
        """
        bootstrap loop to calculate statistic with resampling
        :param sample_data: data to bootstrap
        :param statistic_func: function that calculates the bootstrapped statistic
        :param reps: number of resampling repetitions
        :param kwargs: arguments to pass to the statistic function
        :return: a list of statistic over all the bootstrap samples
        """
        bootstrap_res_list = []
        for _ in range(reps):
            curr_bs_sample = self._get_bootstrap_sample(sample_data)
            bootstrap_res_list.append(statistic_func(data=curr_bs_sample, **kwargs))

        return bootstrap_res_list

# def bootstrap_quantile_regression(mbb: MovingBlockBootstrap, quantiles: List[float], formula: str,
#                                   reps: int) -> Dict[float, list]:
#     """
#
#     :param mbb:
#     :param quantiles:
#     :param formula:
#     :param reps:
#     :return:
#     """
#     bs_res_for_qr = {q: [] for q in quantiles}
#     for curr_itr_data in mbb.bootstrap(reps=reps):
#         # iterator returns back a tuple where second element is a dictionary of bs samples
#         curr_bs = pd.DataFrame(curr_itr_data[1])
#         for curr_q in quantiles:
#             quantile_regression_model = smf.quantreg(formula=formula, data=curr_bs)
#             bs_res_for_qr[curr_q].append(quantile_regression_model.fit(q=curr_q))
#
#     return bs_res_for_qr
