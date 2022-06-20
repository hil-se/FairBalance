from exp import Exp
import pandas as pd
import numpy as np
from stats import is_larger, statics, ranking
import time

def run(data, repeat = 50, seed = 0):
    np.random.seed(seed)
    treatments = ["None", "FERMI", "Reweighing", "FairBalance", "FairBalanceVariant"]
    metrics = ["Accuracy", "F1 Minority", "EOD", "AOD", "Runtime"]
    columns = ["Treatment"] + metrics
    test_result = {column: [] for column in columns}
    for treat in treatments:
        test_result["Treatment"].append(treat)
        test_accuracy = []
        test_f1 = []
        test_eod = []
        test_aod = []
        test_time = []
        for i in range(repeat):
            t1 = time.time()
            exp = Exp(data = data, treatment = treat)
            m_train, m_test = exp.one_exp()
            runtime = time.time() - t1
            test_accuracy.append(m_test.accuracy())
            test_f1.append(m_test.f1_minority())
            test_eod.append(m_test.eod())
            test_aod.append(m_test.aod())
            test_time.append(runtime)
        test_result["Accuracy"].append(test_accuracy)
        test_result["F1 Minority"].append(test_f1)
        test_result["EOD"].append(test_eod)
        test_result["AOD"].append(test_aod)
        test_result["Runtime"].append(test_time)
    for key in metrics:
        if key == "Accuracy" or key == "F1 Minority":
            rank = ranking(test_result[key], better="higher")
        else:
            rank = ranking(test_result[key], better="lower")
        test_result[key] = ["r%d: %.2f (%.2f)" %(rank[i], np.median(test_result[key][i]), np.quantile(test_result[key][i], 0.75)-np.quantile(test_result[key][i], 0.25)) for i in range(len(test_result[key]))]
    df_test = pd.DataFrame(test_result, columns=columns)
    df_test.to_csv("../results/" + data + ".csv", index=False)

if __name__ == "__main__":
    seed = 0
    repeat = 50
    datasets = ["synthetic1", "synthetic2", "synthetic3", "adult", "compas", "heart", "bank"]
    for data in datasets:
        run(data, repeat = repeat, seed=seed)