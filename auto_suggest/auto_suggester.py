# -*- coding: utf-8 -*-
import itertools
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from auto_suggest_utils import compute_jaccard_containment, compute_jaccard_similarity


class AutoSuggester(object):

    def __init__(self, df1, df2, max_num_components=1):
        self.df1 = df1
        self.df2 = df2
        self.column_set_1 = df1.columns.tolist()
        self.column_set_2 = df2.columns.tolist()
        self.max_num_components = max_num_components
        self.numerical_dtypes = ['number', 'datetimetz', 'datetime']

        self.candidate_pairs = self._generate_candidate_pairs()
        self._check()

    def _check(self):
        pass

    def _prune_candidate_pairs(self, raw_candidate_pairs):
        """
        Remove pairs of candidate that have different dtypes
        """

        pruned_candidate_pairs = []
        for candidate_pair in raw_candidate_pairs:

            keep = True
            col_list_1 = candidate_pair[0]
            col_list_2 = candidate_pair[1]

            for col_1, col_2 in zip(col_list_1, col_list_2):
                try:
                    if self.df1[col_1].dtypes.name != self.df2[col_2].dtypes.name:
                        keep = False
                        break
                except Exception as e:
                    print(col_1, col_2)
                    raise ValueError('stop')

            if keep:
                pruned_candidate_pairs.append(candidate_pair)

        print('Number of original candidate pairs: ', len(raw_candidate_pairs))
        print('Number of pruned candidate paris: ', len(pruned_candidate_pairs))

        return pruned_candidate_pairs

    def _generate_all_pairs(self):

        all_pairs = []
        for num_component in range(1, self.max_num_components + 1):
            c1 = list(itertools.combinations(self.column_set_1, num_component))
            c2 = list(itertools.combinations(self.column_set_2, num_component))
            tmp_pairs = list(itertools.product(c1, c2))
            all_pairs.extend(tmp_pairs)

        return all_pairs

    def _generate_candidate_pairs(self):

        raw_candidate_pairs = self._generate_all_pairs()
        candidate_pairs = self._prune_candidate_pairs(raw_candidate_pairs)
        return candidate_pairs

    def compute_distinct_value_ratio(self):
        """
        Ratio of distinct tuples in each (set of) column in the candidate pairs over total number of rows in the parent table.

        :param x:
        """

        result_dict = {
            'col_list_1': [],
            'col_list_2': [],
            'distinct_value_ratio_1': [],
            'distinct_value_ratio_2': []
        }

        cache_dict_1 = {}
        cache_dict_2 = {}
        for candidate_pair in tqdm(self.candidate_pairs):
            col_list_1 = candidate_pair[0]
            col_list_2 = candidate_pair[1]

            if col_list_1 in cache_dict_1:
                # print('Cached: ', col_list_1)
                num_distinct_tuple_1 = cache_dict_1[col_list_1]
            else:
                num_distinct_tuple_1 = len(self.df1.loc[:, col_list_1].drop_duplicates())
                cache_dict_1[col_list_1] = num_distinct_tuple_1

            if col_list_2 in cache_dict_2:
                # print('Cached: ', col_list_2)
                num_distinct_tuple_2 = cache_dict_2[col_list_2]
            else:
                num_distinct_tuple_2 = len(self.df2.loc[:, col_list_2].drop_duplicates())
                cache_dict_2[col_list_2] = num_distinct_tuple_2

            distinct_value_ratio_1 = num_distinct_tuple_1 / len(self.df1)
            distinct_value_ratio_2 = num_distinct_tuple_2 / len(self.df2)

            result_dict['col_list_1'].append(col_list_1)
            result_dict['col_list_2'].append(col_list_2)
            result_dict['distinct_value_ratio_1'].append(distinct_value_ratio_1)
            result_dict['distinct_value_ratio_2'].append(distinct_value_ratio_2)

        return pd.DataFrame(result_dict)

    def compute_value_overlap(self):
        """
        Compute the Jaccard-similarity, as well as Jaccard-containment in both directions

        """
        result_dict = {
            'col_list_1': [],
            'col_list_2': [],
            'jaccard_similarity': [],
            'jaccard_containment_1': [],
            'jaccard_containment_2': []
        }

        cache_dict_1 = {}
        cache_dict_2 = {}

        for candidate_pair in tqdm(self.candidate_pairs):
            col_list_1 = candidate_pair[0]
            col_list_2 = candidate_pair[1]

            if col_list_1 in cache_dict_1:
                set_1 = cache_dict_1[col_list_1]
            else:
                set_1 = set(tuple(i) for i in self.df1.loc[:, col_list_1].values.tolist())
                cache_dict_1[col_list_1] = set_1

            if col_list_2 in cache_dict_2:
                set_2 = cache_dict_2[col_list_2]
            else:
                set_2 = set(tuple(i) for i in self.df2.loc[:, col_list_2].values.tolist())
                cache_dict_2[col_list_2] = set_2

            jaccard_similarity = compute_jaccard_similarity(set_1, set_2)
            jaccard_containment_1 = compute_jaccard_containment(set_1, set_2)
            jaccard_containment_2 = compute_jaccard_containment(set_2, set_1)

            result_dict['col_list_1'].append(col_list_1)
            result_dict['col_list_2'].append(col_list_2)
            result_dict['jaccard_similarity'].append(jaccard_similarity)
            result_dict['jaccard_containment_1'].append(jaccard_containment_1)
            result_dict['jaccard_containment_2'].append(jaccard_containment_2)

        return pd.DataFrame(result_dict)

    def compute_value_range_overlap(self):
        """
        Compute the intersection of the ranges over the union of the ranges.
        """

        result_dict = {
            'col_list_1': [],
            'col_list_2': [],
            'mean_value_range_overlap': []
        }

        cache_dict_1 = {}
        cache_dict_2 = {}

        for candidate_pair in tqdm(self.candidate_pairs):
            col_list_1 = candidate_pair[0]
            col_list_2 = candidate_pair[1]

            value_range_overlap_list = []
            for col_1, col_2 in zip(col_list_1, col_list_2):
                # we check that column is of valid types (numeric or date)
                if self.df1.loc[:, (col_1,)].select_dtypes(self.numerical_dtypes).shape[1] == 1:
                    if col_1 not in cache_dict_1:
                        min_col_1 = min(self.df1[col_1])
                        max_col_1 = max(self.df1[col_1])
                        cache_dict_1[col_1] = (min_col_1, max_col_1)
                    else:
                        min_col_1, max_col_1 = cache_dict_1[col_1]

                    if col_2 not in cache_dict_2:
                        min_col_2 = min(self.df2[col_2])
                        max_col_2 = max(self.df2[col_2])
                        cache_dict_2[col_2] = (min_col_2, max_col_2)
                    else:
                        min_col_2, max_col_2 = cache_dict_2[col_2]

                    lst = sort([min_col_1, max_col_1, min_col_2, max_col_2])
                    value_range_overlap = (lst[2] - lst[1]) / (lst[3] - lst[0])
                    value_range_overlap_list.append(value_range_overlap)

                else:
                    if col_2 == 'created_at':
                        print('Skipping: ', col_2)
                    continue

            result_dict['col_list_1'].append(col_list_1)
            result_dict['col_list_2'].append(col_list_2)
            result_dict['mean_value_range_overlap'].append(np.mean(value_range_overlap_list))

        return pd.DataFrame(result_dict)

    def compute_col_value_types(self):

        result_dict = {
            'col_list_1': [],
            'col_list_2': [],
            'col_value_types': []
        }

        for candidate_pair in tqdm(self.candidate_pairs):
            col_list_1 = candidate_pair[0]
            col_list_2 = candidate_pair[1]

            # no need to check col_list_2 as they must have the same dtype as col_list_1
            # sort the list so that it's easier for comparison
            col_value_types = sort([v.name for v in self.df1.loc[:, col_list_1].dtypes.values]).tolist()

            result_dict['col_list_1'].append(col_list_1)
            result_dict['col_list_2'].append(col_list_2)
            result_dict['col_value_types'].append(json.dumps(col_value_types))

        return pd.DataFrame(result_dict)

    def compute_leftness(self):
        """
        The positions of the column(s) in the table, in both absolute terms (e.g., the 2nd column from left),
        and relative terms (e.g., 2nd column out of 20 total cols is 2/20=10%)
        """

        result_dict = {
            'col_list_1': [],
            'col_list_2': [],
            'average_absolute_leftness_1': [],
            'average_absolute_leftness_2': [],
            'average_relative_leftness_1': [],
            'average_relative_leftness_2': []
        }

        for candidate_pair in tqdm(self.candidate_pairs):
            col_list_1 = candidate_pair[0]
            col_list_2 = candidate_pair[1]

            indexes_col_list_1 = np.array([self.df1.columns.tolist().index(c) for c in col_list_1])
            indexes_col_list_2 = np.array([self.df2.columns.tolist().index(c) for c in col_list_2])

            average_absolute_leftness_1 = np.mean(indexes_col_list_1)
            average_absolute_leftness_2 = np.mean(indexes_col_list_2)

            average_relative_leftness_1 = np.mean(indexes_col_list_1 / len(self.df1.columns))
            average_relative_leftness_2 = np.mean(indexes_col_list_2 / len(self.df2.columns))

            result_dict['col_list_1'].append(col_list_1)
            result_dict['col_list_2'].append(col_list_2)
            result_dict['average_absolute_leftness_1'].append(average_absolute_leftness_1)
            result_dict['average_absolute_leftness_2'].append(average_absolute_leftness_2)
            result_dict['average_relative_leftness_1'].append(average_relative_leftness_1)
            result_dict['average_relative_leftness_2'].append(average_relative_leftness_2)

        return pd.DataFrame(result_dict)

    def compute_sortedness(self):
        """
        Whether values in candidate columns are sorted
        """

        result_dict = {
            'col_list_1': [],
            'col_list_2': [],
            'sortedness_1': [],
            'sortedness_2': []
        }

        for candidate_pair in tqdm(self.candidate_pairs):
            col_list_1 = candidate_pair[0]
            col_list_2 = candidate_pair[1]

            sortedness_1 = all([self.df1[col].is_monotonic for col in col_list_1])
            sortedness_2 = all([self.df2[col].is_monotonic for col in col_list_2])
            result_dict['col_list_1'].append(col_list_1)
            result_dict['col_list_2'].append(col_list_2)
            result_dict['sortedness_1'].append(sortedness_1)
            result_dict['sortedness_2'].append(sortedness_2)

        return pd.DataFrame(result_dict)

    def compute_single_column_candidate(self):
        """
        Whether a candidate is single-column or not
        """

        result_dict = {
            'col_list_1': [],
            'col_list_2': [],
            'single_column_candidate': []
        }

        for candidate_pair in tqdm(self.candidate_pairs):
            col_list_1 = candidate_pair[0]
            col_list_2 = candidate_pair[1]

            single_column_candidate = len(col_list_1) == 1

            result_dict['col_list_1'].append(col_list_1)
            result_dict['col_list_2'].append(col_list_2)
            result_dict['single_column_candidate'].append(single_column_candidate)

        return pd.DataFrame(result_dict)

    def compute_table_level_statistics(self):
        """
        Compute statistics of input tables:
           * Number of rows
           * Ratio of the two row-counts
        """

        num_rows_1 = len(self.df1)
        num_rows_2 = len(self.df2)
        row_ratio = float(num_rows_1) / num_rows_2

        result_dict = {
            'col_list_1': [c[0] for c in self.candidate_pairs],
            'col_list_2': [c[1] for c in self.candidate_pairs],
            'num_rows_1': [num_rows_1] * len(self.candidate_pairs),
            'num_rows_2': [num_rows_2] * len(self.candidate_pairs),
            'row_ratio': [row_ratio] * len(self.candidate_pairs)
        }

        return pd.DataFrame(result_dict)