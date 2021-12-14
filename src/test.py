from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from fairbalance import FairBalance
from fairbalance import Reweighing_multiple
from fairbalance import FairBalance2
from experiment import Experiment


data = load_preproc_data_adult()

X = data.features
Y = data.labels.ravel()

data_train, data_test = data.split([0.7], shuffle=True)

# privileged_groups = [{target_attribute: 1}]
# unprivileged_groups = [{target_attribute: 0}]
print(type(data_train))

exp = Experiment("LR", data="adult", fair_balance="FairBalance", target_attribute='')
result = exp.run()

