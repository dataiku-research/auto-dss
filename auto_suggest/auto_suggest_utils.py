# -*- coding: utf-8 -*-

def compute_jaccard_similarity(set_1, set_2):
    intersection = len(set_1.intersection(set_2))
    union = len(set_1.union(set_2))

    if union > 0:
        return float(intersection) / union
    else:
        return None


def compute_jaccard_containment(set_1, set_2):
    intersection = len(set_1.intersection(set_2))
    if len(set_1) > 0:
        return float(intersection) / len(set_1)
    else:
        return None

def totuple(a):
    try:
        if isinstance(a, (list or tuple)):
            return tuple(totuple(i) for i in a)
        else:
            return a
    except TypeError:
        return a