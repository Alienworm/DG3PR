import jenkspy
import numpy as np
import pandas as pd
import torch


def discrete_value_to_grade(
        raw_data: pd.DataFrame,
        group_column_names: list,
        column_name: str,
        new_column_name: str,
        grade_num: int
) -> pd.Series:
    """
    discrete value to grade
    :param raw_data:
    :param group_column_names:
    :param column_name:
    :param new_column_name:
    :param grade_num:
    :return:
    """

    group_data = raw_data.groupby(by=group_column_names)
    for group_name, group in group_data:
        values = np.round(group[column_name].values, 3)

        tmp_grade_num = grade_num
        if len(np.unique(values)) < grade_num:
            tmp_grade_num = len(np.unique(values))
        breaks = jenkspy.jenks_breaks(values, n_classes=tmp_grade_num)
        breaks[0] -= 1
        breaks[-1] += 1

        if len(group_column_names) > 1:
            group_name = group_name[0]

        def get_grade(value: float) -> str:
            for index in range(len(breaks) - 1):
                if breaks[index] < value <= breaks[index + 1]:
                    return group_name + '_' + str(index)
            return group_name + '_-1'

        group[new_column_name] = group[column_name].apply(get_grade)
        raw_data.loc[group.index, new_column_name] = group[new_column_name]

    return raw_data[new_column_name]


def dataset_to_tensor(samples: dict, device: str = 'cuda', dataset_type: str = 'train'):
    """
    :param samples:
    :param device:
    :param dataset_type:
    :return:
    """

    new_samples = {}
    for index, sample in samples.items():
        if dataset_type == 'train':
            negative_sample_list = [
                negative_sample.to(device) for negative_sample in sample['negative_sample_list']
            ]
            new_samples.setdefault(index, {
                'positive_sample': sample['positive_sample'].to(device),
                'negative_sample_list': negative_sample_list
            })
        if dataset_type == 'eval':
            new_samples.setdefault(index, {
                'sample': sample['sample'].to(device),
                'purchased': torch.tensor(sample['purchased'], dtype=torch.long, device=device)
            })
    return new_samples