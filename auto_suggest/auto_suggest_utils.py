# -*- coding: utf-8 -*-

def compute_jaccard_similarity(set_1, set_2):
    intersection = len(set_1.intersection(set_2))
    union = len(set_1.union(set_2))
    return float(intersection) / union


def compute_jaccard_containment(set_1, set_2):
    intersection = len(set_1.intersection(set_2))
    return float(intersection) / len(set_1)