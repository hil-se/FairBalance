import copy
from collections import Counter

# Balance training data
def FairBalance(data, class_balance = False):
    y = data.labels.ravel()
    grouping = {}
    for i, label in enumerate(y):
        key = tuple(list(data.protected_attributes[i])+[label])
        if key not in grouping:
            grouping[key]=[]
        grouping[key].append(i)
    class_weight = Counter(y)
    if class_balance:
        class_weight = {key: 1.0 for key in class_weight}
    weighted_data = copy.deepcopy(data)
    for key in grouping:
        weight = class_weight[key[-1]]/len(grouping[key])
        for i in grouping[key]:
            weighted_data.instance_weights[i] = weight
    # Rescale the total weights to len(y)
    weighted_data.instance_weights = weighted_data.instance_weights * len(y) / sum(weighted_data.instance_weights)
    return weighted_data

# Balance training data in a simpler way
def ClassBalance(data):
    y = data.labels.ravel()
    grouping = {}
    groups_only = {}
    for i, label in enumerate(y):
        key = tuple(list(data.protected_attributes[i]) + [label])
        key_group_only = tuple(list(data.protected_attributes[i]))
        if key not in grouping:
            grouping[key] = []
        grouping[key].append(i)
        if key_group_only not in groups_only:
            groups_only[key_group_only] = []
        groups_only[key_group_only].append(i)
    weighted_data = copy.deepcopy(data)
    for key in grouping:
        weight = len(groups_only[key[:-1]]) / len(grouping[key])
        for i in grouping[key]:
            weighted_data.instance_weights[i] = weight
    # Rescale the total weights to len(y)
    weighted_data.instance_weights = weighted_data.instance_weights * len(y) / sum(weighted_data.instance_weights)
    return weighted_data

# Extended version of reweighing for multiple sensitive attributes
def Reweighing_multiple(data):
    y = data.labels.ravel()
    grouping = {}
    dist = {}
    for i, label in enumerate(y):
        key = tuple(list(data.protected_attributes[i])+[label])
        if key not in grouping:
            grouping[key]=[]
        att = tuple(data.protected_attributes[i])
        if att not in dist:
            dist[att] = 0
        dist[att] += 1
        grouping[key].append(i)
    class_weight = Counter(y)
    weighted_data = copy.deepcopy(data)
    for key in grouping:
        weight = dist[key[:-1]]*class_weight[key[-1]]/len(grouping[key])/len(y)
        for i in grouping[key]:
            weighted_data.instance_weights[i] = weight
    return weighted_data
