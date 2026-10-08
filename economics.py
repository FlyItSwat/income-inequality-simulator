"""Transparent, stylized income-tax-and-transfer simulation.
All incomes are annual INR; households are equally weighted.
"""
import numpy as np
import pandas as pd


def gini(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError('Income must be a nonempty array of finite nonnegative numbers.')
    if a.sum() == 0:
        return 0.0
    s = np.sort(a)
    n = len(s)
    return float((2 * np.arange(1, n + 1) - n - 1) @ s / (n * s.sum()))


def lorenz(values):
    a = np.asarray(values, dtype=float)
    gini(a)  # validate
    a = np.sort(a)
    shares = np.arange(len(a) + 1) / len(a)
    cumulative = np.r_[0, np.cumsum(a)]
    if cumulative[-1] == 0:
        return shares, shares.copy()  # no inequality when all incomes are zero
    return shares, cumulative / cumulative[-1]


def progressive_tax(incomes, thresholds, rates):
    """Marginal tax: thresholds are ascending lower bounds, starting at zero.

    Example: thresholds [0,300000,700000], rates [0,0.1,0.2].
    """
    x = np.asarray(incomes, dtype=float)
    gini(x)
    t = np.asarray(thresholds, dtype=float)
    r = np.asarray(rates, dtype=float)
    if (len(t) != len(r) or len(t) == 0 or t[0] != 0 or
        np.any(np.diff(t) <= 0) or np.any(t < 0) or
        np.any(~np.isfinite(t)) or np.any(~np.isfinite(r)) or
        np.any((r < 0) | (r > 1))):
        raise ValueError('Invalid thresholds or rates.')
    upper = np.r_[t[1:], np.inf]
    result = np.zeros_like(x)
    for lo, hi, rate in zip(t, upper, r):
        result += np.maximum(np.minimum(x, hi) - lo, 0) * rate
    return result


def simulate(incomes, thresholds, rates, transfer_share=1.0):
    """Return policy results. transfer_share of taxes redistributed equally.

    Remaining taxes represent government retention (not modeled as spending).
    """
    x = np.asarray(incomes, dtype=float)
    gini(x)
    if not 0 <= transfer_share <= 1:
        raise ValueError('transfer_share must be between 0 and 1')
    taxes = progressive_tax(x, thresholds, rates)
    revenue = float(taxes.sum())
    transfer = revenue * transfer_share / len(x)
    after = x - taxes + transfer
    return pd.DataFrame({'income_before': x, 'tax': taxes,
                         'transfer': np.full(len(x), transfer),
                         'income_after': after, 'net_change': after - x})


def decile_summary(results):
    """Assign deciles by pre-policy income rank; supports small datasets."""
    df = results.sort_values('income_before', kind='stable').copy()
    n = len(df)
    df['decile'] = np.floor(np.arange(n) * 10 / n).astype(int) + 1
    return df.groupby('decile', as_index=False).agg(
        households=('income_before', 'size'),
        mean_before=('income_before', 'mean'),
        mean_after=('income_after', 'mean'),
        mean_tax=('tax', 'mean'),
        mean_transfer=('transfer', 'mean'),
        mean_net_change=('net_change', 'mean'))
