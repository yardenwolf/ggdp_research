from typing import List

import numpy as np
import pandas as pd
from evt.estimators.hill import Hill
from evt.methods.peaks_over_threshold import PeaksOverThreshold
from evt.dataset import Dataset
from evt.estimators.estimator_abc import Estimate
import evt.utils as utils


class HillEstimator():
    def __init__(self, data: pd.Series, tail: str):
        if tail == 'left':
            self.data = Dataset(-1 * data)
        else:
            self.data = Dataset(data)
    def estimate_hill_statistic(self, threshold: float):
        peaks_over_threshold = PeaksOverThreshold(self.data, threshold)
        max_n_obs = len(peaks_over_threshold.series_tail)
        estimator_res = []
        for curr_obs in range(1, max_n_obs):
            hill = Hill(peaks_over_threshold)
            # appending dummy variable to correct the library
            hills_estimator: Estimate = hill.estimate(curr_obs)[0]
            estimator_res.append(
                (curr_obs, hills_estimator.estimate, hills_estimator.ci_lower, hills_estimator.ci_upper))
        return pd.DataFrame(estimator_res, columns=['num_tail_observations', 'estimate', 'ci_lower', 'ci_upper'])
