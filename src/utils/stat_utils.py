import scipy.stats


def get_welch_t_test_p(m1: float, s1: float, m2: float, s2: float, n1: int, n2: int) -> float:
    """Returns the p-value of a Welch's t-test. The null hypothesis is that the two samples have the same mean.
    Alternative hypothesis is that the first sample has a smaller mean than the second sample.
    The first distribution is (m1, s1) and the second distribution is (m2, s2). The number of samples in each distribution is n1 and n2 respectively.

    Returns:
        float: p-value
    """
    return scipy.stats.ttest_ind_from_stats(
        mean1=m1,
        mean2=m2,
        std1=s1,
        std2=s2,
        nobs1=n1,
        nobs2=n2,
        equal_var=False,  # Welch's t-test - unequal variances
        alternative="less",  # one-sided - generated is faster
    ).pvalue

