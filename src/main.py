from demos import cmd
import copy
try:
   import cPickle as pickle
except:
   import pickle
from utils import *
from experiment import Experiment


def one_exp(treatment, data, fair_balance, target="", repeats=10):
    # Conduct one experiment:
    #     treatment in {"SVM", "RF", "LR", "DT"}
    #     data in {"compas", "adult", "german"}
    #     fair_balance in {"None", "FairBalance", "Reweighing", "AdversialDebiasing", "RejectOptionClassification"}
    #     target = target protected attribute, not used if fair_balance == "FairBlance" or "None"
    #     repeats = number of times repeating the experiments

    np.random.seed(2)
    exp = Experiment(treatment, data=data, fair_balance=fair_balance, target_attribute=target)
    results = {}
    for _ in range(repeats):
        result = exp.run()
        if result:
            results = merge_dict(results, result)
    print(results)
    return results

def RQ1():
    # Perform an overall experiment on different algorithms, datasets, and FairBalance settings.
    treatments = ["LR", "SVM", "DT", "RF"]
    datasets = ["compas", "adult", "german"]
    balances = ["None", "FairBalance"]
    results = {}
    for treatment in treatments:
        results[treatment] = {}
        for dataset in datasets:
            results[treatment][dataset] = {}
            for balance in balances:
                results[treatment][dataset][balance] = one_exp(treatment, dataset, balance, repeats=50)
                # Print progress
                print(treatment+", "+dataset+", "+balance)
    # dump results
    with open("../dump/RQ1.pickle", "wb") as p:
        pickle.dump(results, p)
    parse_results_RQ1()

def RQ2():
    # Compare FairBalance against other soa baseline bias mitigation algorithms.
    # Classifier is fixed to logistic regression.
    treatment = "LR"
    datasets = ["compas", "adult", "german"]
    balances = ["Reweighing", "AdversialDebiasing", "RejectOptionClassification", "FairBalance"]
    targets = {"compas": ["sex", "race"], "adult": ["sex", "race"], "german": ["sex", "age"]}
    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for balance in balances:
            if balance!="FairBalance":
            # Need target attribute
                for target in targets[dataset]:
                    results[dataset][balance+": "+target] = one_exp(treatment, dataset, balance, target=target)
            else:
                results[dataset][balance] = one_exp(treatment, dataset, balance)
            # Print progress
            print(dataset + ", " + balance)
    # dump results
    with open("../dump/RQ2.pickle", "wb") as p:
        pickle.dump(results, p)
    parse_results_RQ2()

def parse_results_RQ1(iqr="True"):
    # Parse results of RQ1 and save as csv files.
    with open("../dump/RQ1.pickle", "rb") as p:
        results = pickle.load(p)
    # Compare results between w/ and w/o FairBalance
    compares = copy.deepcopy(results)
    for treatment in compares:
        for dataset in compares[treatment]:
            compares[treatment][dataset] = compare_dict(compares[treatment][dataset])
    compare_df = dict2dfRQ1(compares)
    compare_df.to_csv("../results/RQ1_compare.csv", index=False)

    # Calculate medians and iqrs of 30 repeats
    medians = copy.deepcopy(results)
    medians = median_dict(medians, use_iqr = iqr=="True")
    median_df = dict2dfRQ1(medians)
    median_df.to_csv("../results/RQ1_median.csv", index=False)

def parse_results_RQ2(iqr="True"):
    # Parse results of RQ2 and save as csv files.
    with open("../dump/RQ2.pickle", "rb") as p:
        results = pickle.load(p)
    # Calculate medians and iqrs of 30 repeats
    medians = copy.deepcopy(results)
    medians = median_dict(medians, use_iqr = iqr=="True")
    median_df = dict2dfRQ2(medians)
    median_df.to_csv("../results/RQ2_median.csv", index=False)


if __name__ == "__main__":
    eval(cmd())