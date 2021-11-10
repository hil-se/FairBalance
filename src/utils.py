import numpy as np
import pandas as pd
import copy
from scipy.stats import mannwhitneyu
try:
   import cPickle as pickle
except:
   import pickle


def merge_dict(results, result):
    # Merge nested dictionaries
    for key in result:
        if type(result[key]) == dict:
            if key not in results:
                results[key] = {}
            results[key] = merge_dict(results[key], result[key])
        else:
            if key not in results:
                results[key] = []
            results[key].append(result[key])
    return results



def is_larger(x, y):
    # Check if results in x is significantly larger than those in y.
    # Return int values:
        # 0: not significantly larger
        # 1: larger with small effect size
        # 2: larger with medium effect size
        # 3: larger with large effect size

    # Mann Whitney U test
    if np.array_equal(x, y):
        return 0
    U, pvalue = mannwhitneyu(x, y, alternative="greater")
    if pvalue>0.05:
        # If x is not greater than y in 95% confidence
        return 0
    else:
        # Calculate Cliff's delta with U
        delta = 2*U/(len(x)*len(y))-1
        # Return different levels of effect size
        if delta<0.147:
            return 0
        elif delta<0.33:
            return 1
        elif delta<0.474:
            return 2
        else:
            return 3

def rank(samples, lower_better=True, thres = 2):
    # Rank samples based on is_larger function, thres can be {1, 2, 3}
    medians = [np.abs(np.median(sample)) for sample in samples]
    order = np.argsort(medians)
    order = order if lower_better else order[::-1]
    baseline = samples[order[0]]
    ranking = 0
    ranks = [0]
    for i in order[1:]:
        if lower_better:
            better = is_larger(np.abs(samples[i]), np.abs(baseline))
        else:
            better = is_larger(np.abs(baseline), np.abs(samples[i]))
        if better >= thres:
            ranking += 1
            baseline = samples[i]
        ranks.append(ranking)
    ordered_rank = ranks[:]
    for i, o in enumerate(order):
        ordered_rank[o] = ranks[i]
    return ordered_rank




def rank_dict(results):
    # Rank each treatment based on statistical significance for each metric.
    treatments = list(results.keys())
    if len(treatments)==0:
        return results
    keys = list(results[treatments[0]].keys())

    for key in keys:
        lower_better = key == "runtime" or key == "fpr" or type(results[treatments[0]][key]) == dict
        if type(results[treatments[0]][key]) == dict:
            for key2 in results[treatments[0]][key]:
                to_compare = []
                for treatment in treatments:
                    to_compare.append(results[treatment][key][key2])

                ranks = rank(to_compare, lower_better)
                for i, treatment in enumerate(treatments):
                    results[treatment][key][key2] = ranks[i]
        else:
            to_compare = []
            for treatment in treatments:
                to_compare.append(results[treatment][key])
            ranks = rank(to_compare, lower_better)
            for i, treatment in enumerate(treatments):
                results[treatment][key] = ranks[i]
    return results



def compare_dict(results, baseline="None"):
    # Check if results of non-baseline treatments are significantly better than the baseline.

    y = results[baseline]
    for treatment in results:
        if treatment==baseline:
            continue
        x = results[treatment]
        for key in x:
            if type(x[key]) == dict:
                # Bias Metrics: lower abs the better
                for key2 in x[key]:
                    xx = x[key][key2]
                    yy = y[key][key2]
                    better = is_larger(np.abs(yy), np.abs(xx))
                    if better == 0:
                        better = -is_larger(np.abs(xx), np.abs(yy))
                    x[key][key2] = better
            else:
                # General Metrics: higher the better except for runtime and fpr
                xx = x[key]
                yy = y[key]
                better = is_larger(xx, yy)
                if better == 0:
                    better = -is_larger(yy, xx)
                if key == "runtime" or key == "fpr":
                    x[key] = -better
                else:
                    x[key] = better
    for key in y:
        if type(y[key]) == dict:
            for key2 in y[key]:
                y[key][key2] = "n/a"
        else:
            y[key] = "n/a"
    return results

def median_dict(results, use_iqr = True, abs = False):
    # Compute median value of lists in the dictionary
    for key in results:
        if type(results[key]) == dict:
            results[key] = median_dict(results[key], use_iqr = use_iqr, abs = abs)
        else:
            if abs:
                med = np.median(np.abs(results[key]))
            else:
                med = np.median(results[key])
            if use_iqr:
                iqr = np.percentile(results[key],75)-np.percentile(results[key],25)
                results[key] = "%d (%d)" % (med*100, iqr*100)
            else:
                results[key] = "%d" % (med*100)
    return results

def mean_dict(results, std = True):
    # Compute mean value of lists in the dictionary
    for key in results:
        if type(results[key]) == dict:
            results[key] = median_dict(results[key])
        else:
            med = np.mean(results[key])
            if std:
                std = np.std(results[key])
                results[key] = "%.2f (%.2f)" % (med, std)
            else:
                results[key] = "%.2f" % (med)
    return results

def dict2dfRQ1(results):
    # Generate a pandas dataframe based on the dictionary
    columns = ["Algorithm", "Dataset", "Treatment", "F1", "Accuracy", "Runtime", "Sex: AOD", "Sex: EOD", "Sex: SPD", "Race/Age: AOD",
               "Race/Age: EOD", "Race/Age: SPD"]
    df = {key:[] for key in columns}
    for treatment in results:
        for dataset in results[treatment]:
            # for balance in results[treatment][dataset]:
            for balance in ["None", "FairBalance", "FairBalanceClass"]:
                x = results[treatment][dataset][balance]
                df["Algorithm"].append(treatment)
                df["Dataset"].append(dataset)
                df["Treatment"].append(balance)
                df["F1"].append(x["f1"])
                df["Accuracy"].append(x["acc"])
                df["Runtime"].append(x["runtime"])
                df["Sex: AOD"].append(x["sex"]["aod"])
                df["Sex: EOD"].append(x["sex"]["eod"])
                df["Sex: SPD"].append(x["sex"]["spd"])
                if "race" in x:
                    df["Race/Age: AOD"].append(x["race"]["aod"])
                    df["Race/Age: EOD"].append(x["race"]["eod"])
                    df["Race/Age: SPD"].append(x["race"]["spd"])
                elif "age" in x:
                    df["Race/Age: AOD"].append(x["age"]["aod"])
                    df["Race/Age: EOD"].append(x["age"]["eod"])
                    df["Race/Age: SPD"].append(x["age"]["spd"])
                else:
                    df["Race/Age: AOD"].append("")
                    df["Race/Age: EOD"].append("")
                    df["Race/Age: SPD"].append("")
    df = pd.DataFrame(df, columns = columns)
    return df

def dict2dfRQ3(results):
    # Generate a pandas dataframe based on the dictionary
    columns = ["Dataset", "Algorithm", "F1", "Accuracy", "Runtime", "Sex: AOD", "Sex: EOD", "Sex: SPD", "Race/Age: AOD",
               "Race/Age: EOD", "Race/Age: SPD"]
    df = {key: [] for key in columns}
    for dataset in results:
        # for balance in results[treatment][dataset]:
        for treatment in results[dataset]:
            x = results[dataset][treatment]
            df["Dataset"].append(dataset)
            df["Algorithm"].append(treatment)
            df["F1"].append(x["f1"])
            df["Accuracy"].append(x["acc"])
            df["Runtime"].append(x["runtime"])
            df["Sex: AOD"].append(x["sex"]["aod"])
            df["Sex: EOD"].append(x["sex"]["eod"])
            df["Sex: SPD"].append(x["sex"]["spd"])
            if "race" in x:
                df["Race/Age: AOD"].append(x["race"]["aod"])
                df["Race/Age: EOD"].append(x["race"]["eod"])
                df["Race/Age: SPD"].append(x["race"]["spd"])
            elif "age" in x:
                df["Race/Age: AOD"].append(x["age"]["aod"])
                df["Race/Age: EOD"].append(x["age"]["eod"])
                df["Race/Age: SPD"].append(x["age"]["spd"])
            else:
                df["Race/Age: AOD"].append("")
                df["Race/Age: EOD"].append("")
                df["Race/Age: SPD"].append("")
    df = pd.DataFrame(df, columns = columns)
    return df

def color(median, compare):
    mapping = {3: "\\cellcolor{green!70}", 2: "\\cellcolor{green!35}", 1: "\\cellcolor{green!15}",
               -3: "\\cellcolor{red!70}", -2: "\\cellcolor{red!35}", -1: "\\cellcolor{red!15}",
               0: "", "n/a": ""}
    for key in median:
        if type(median[key]) == dict:
            median[key] = color(median[key], compare[key])
        else:
            median[key] = mapping[compare[key]]+median[key]
    return median

def combine(median, compare):
    for key in median:
        if type(median[key]) == dict:
            median[key] = combine(median[key], compare[key])
        else:
            median[key] = "R"+str(compare[key])+": "+median[key]
    return median