import numpy as np
import scipy.stats as stats


def t_test_two_tailed(seris_1: np.array, seris_2: np.array):
    #need to fix this
    f_statistic = np.var(seris_1) / np.var(seris_2)
    df1 = len(seris_1) - 1
    df2 = len(seris_2) - 1
    p_value = (1 - stats.f.cdf(f_statistic, df1, df2)) + stats.f.cdf(f_statistic, df1, df2)
    return p_value


def ks_test(sample_1: np.array, sample_2: np.array):
    ecdf_1 = stats.ecdf(sample_1)
    q_ecdf_2 = stats.ecdf(sample_2)
    ks_statistic = stats.ks_2samp()