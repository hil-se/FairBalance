import copy
from collections import Counter


# Balance training data
def FairBalance(data, class_balance=False):
    y = data.labels.ravel()
    grouping = {}
    for i, label in enumerate(y):
        key = tuple(list(data.protected_attributes[i]) + [label])
        if key not in grouping:
            grouping[key] = []
        grouping[key].append(i)
    class_weight = Counter(y)
    if class_balance:
        class_weight = {key: 1.0 for key in class_weight}
    weighted_data = copy.deepcopy(data)
    for key in grouping:
        weight = class_weight[key[-1]] / len(grouping[key])
        for i in grouping[key]:
            weighted_data.instance_weights[i] = weight
    # Rescale the total weights to len(y)
    weighted_data.instance_weights = weighted_data.instance_weights * len(y) / sum(weighted_data.instance_weights)
    return weighted_data


# Balance training data in a simpler way
def FairBalance2(data, class_balance=False):
    y = data.labels.ravel()
    m, n = data.protected_attributes.shape
    groups = []
    for i in range(n):
        groups.append(Counter(data.protected_attributes[:, i]))
    classes = Counter(y)
    weighted_data = copy.deepcopy(data)
    for i, label in enumerate(y):
        weight = 1
        if class_balance:
            weight /= classes[label]
        for j in range(n):
            weight /= groups[j][data.protected_attributes[i, j]]
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
        key = tuple(list(data.protected_attributes[i]) + [label])
        if key not in grouping:
            grouping[key] = []
        att = tuple(data.protected_attributes[i])
        if att not in dist:
            dist[att] = 0
        dist[att] += 1
        grouping[key].append(i)
    class_weight = Counter(y)
    weighted_data = copy.deepcopy(data)
    for key in grouping:
        weight = dist[key[:-1]] * class_weight[key[-1]] / len(grouping[key]) / len(y)
        for i in grouping[key]:
            weighted_data.instance_weights[i] = weight
    return weighted_data
