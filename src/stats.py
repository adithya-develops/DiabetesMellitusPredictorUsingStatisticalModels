import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, f_oneway, f_oneway

def conditional_probability(data, condition, target="diabetes"):
    subset = data.query(condition)
    return float(subset[target].mean()) if len(subset) else np.nan

def proportion_ci(successes, n, confidence=0.95):
    p = successes / n
    z = norm.ppf(1 - (1-confidence)/2)
    se = np.sqrt(p*(1-p)/n)
    return p, max(0.0, p-z*se), min(1.0, p+z*se)

def chi_square(data, feature, target="diabetes"):
    table = pd.crosstab(data[feature], data[target])
    chi2, p, dof, expected = chi2_contingency(table)
    return table, chi2, p, dof


def one_way_anova(data, numeric_feature, group_feature, min_group_size=2):
    groups = []
    labels = []
    for label, subset in data[[numeric_feature, group_feature]].dropna().groupby(group_feature):
        values = subset[numeric_feature].astype(float).to_numpy()
        if len(values) >= min_group_size:
            labels.append(label)
            groups.append(values)
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least two groups with enough observations.")
    f_stat, p_value = f_oneway(*groups)
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
    eta_sq = ss_between / ss_total if ss_total else np.nan
    summary = pd.DataFrame({
        "Group": labels,
        "N": [len(g) for g in groups],
        "Mean": [g.mean() for g in groups],
        "SD": [g.std(ddof=1) if len(g) > 1 else np.nan for g in groups],
    })
    return f_stat, p_value, len(groups)-1, len(all_values)-len(groups), eta_sq, summary


def one_way_anova(data, numeric_feature, group_feature, min_group_size=2):
    groups, labels = [], []
    for label, subset in data[[numeric_feature, group_feature]].dropna().groupby(group_feature):
        values = subset[numeric_feature].astype(float).to_numpy()
        if len(values) >= min_group_size:
            labels.append(label); groups.append(values)
    if len(groups) < 2: raise ValueError("ANOVA requires at least two groups with enough observations.")
    f_stat, p_value = f_oneway(*groups)
    all_values = np.concatenate(groups); grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
    eta_sq = ss_between / ss_total if ss_total else np.nan
    summary = pd.DataFrame({"Group": labels, "N": [len(g) for g in groups], "Mean": [g.mean() for g in groups], "SD": [g.std(ddof=1) if len(g)>1 else np.nan for g in groups]})
    return f_stat, p_value, len(groups)-1, len(all_values)-len(groups), eta_sq, summary
