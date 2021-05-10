import copy

# Balance training data
def FairBalance(data):
    y = data.labels.ravel()
    grouping = {}
    for i, label in enumerate(y):
        key = tuple(list(data.protected_attributes[i])+[label])
        if key not in grouping:
            grouping[key]=[]
        grouping[key].append(i)
    weighted_data = copy.deepcopy(data)
    for key in grouping:
        weight = 1.0/len(grouping[key])
        for i in grouping[key]:
            weighted_data.instance_weights[i] = weight
    return weighted_data

