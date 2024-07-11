import time
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data_util import dataset_to_tensor

from models.dg3pr import DG3PR
from knowledge_graph.knowledge_graph_data import KnowledgeGraphData, Sample, SampleNoneInfo, KnowledgeGraphEvalDataset


def calculate_score(
        knowledge_graph_data: KnowledgeGraphData,
        model: DG3PR,
        batch_size: int = 16384,
        device: str = 'cuda'
) -> (pd.DataFrame, float):
    """
    Calculate score
    :param knowledge_graph_data:
    :param model:
    :param batch_size:
    :param device:
    :return:
    """
    start_time = time.time()

    score_map = {}
    samples = knowledge_graph_data.get_samples()
    datasets = KnowledgeGraphEvalDataset(samples, device=device)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        for user_id, item_id, sub_node_id_list, value, purchased in dataloader:
            score = model(user_id, item_id, sub_node_id_list, value).cpu().numpy()

            purchased = purchased.cpu().numpy()
            for i in range(len(user_id)):
                customer_id = user_id[i].cpu().item()
                breed_id = item_id[i].cpu().item()
                tmp_value = value[i].cpu().item()
                score_value = score[i].item()
                purchased_value = purchased[i].item()

                if customer_id not in score_map:
                    score_map[customer_id] = {}
                if breed_id not in score_map[customer_id]:
                    score_map[customer_id][breed_id] = {
                        'value': 0,
                        'score': 0,
                        'purchased': 0
                    }

                score_map[customer_id][breed_id]['value'] = tmp_value
                score_map[customer_id][breed_id]['score'] = score_value
                score_map[customer_id][breed_id]['purchased'] = purchased_value

    # convert score map to dataframe with header: customer_id, breed_id, score, is_buy
    score_dataframe = []
    for customer_id, score_info in score_map.items():
        for breed_id, score in score_info.items():
            score_dataframe.append({
                'customer_id': customer_id,
                'breed_id': breed_id,
                'value': score['value'],
                'score': score['score'],
                'purchased': score['purchased']
            })

    score_dataframe = pd.DataFrame(score_dataframe)
    score_dataframe['value'] = score_dataframe['value'].fillna(0)

    return score_dataframe, time.time() - start_time


def calculate_metrics(score_dataframe: pd.DataFrame, top_k: int = 10) -> (dict, float):
    """
    Calculate precision, recall, f1, hit rate, ndcg, map, mrr
    :param score_dataframe:
    :param top_k:
    :return:
    """

    start_time = time.time()
    metrics_result = {'precision': 0, 'recall': 0, 'f1': 0, 'hit_rate': 0, 'ndcg': 0}
    customer_count = 0

    for customer_id in score_dataframe['customer_id'].unique():
        customer_score_dataframes = score_dataframe[score_dataframe['customer_id'] == customer_id]

        purchase_items = customer_score_dataframes[customer_score_dataframes['purchased'] == 1]

        # sort purchase_items by sale_qty
        purchase_items = purchase_items.sort_values(by='value', ascending=False)

        recommend_items = customer_score_dataframes.sort_values(by='score', ascending=False).head(top_k)
        correct_items = set(purchase_items['breed_id'].values) & set(recommend_items['breed_id'].values)
        purchase_items_num = len(purchase_items)
        recommend_items_num = len(recommend_items)
        correct_items_num = len(correct_items)

        if purchase_items_num <= 0 or recommend_items_num <= 0:
            continue
        customer_count += 1

        metrics_result['precision'] += correct_items_num / top_k
        metrics_result['recall'] += correct_items_num / min(top_k, purchase_items_num)
        metrics_result['hit_rate'] += 1 if correct_items_num > 0 else 0

        # calculate ndcg
        dcg = 0
        for i in range(top_k):
            if recommend_items['breed_id'].values[i] in correct_items:
                dcg += 1 / np.log2(i + 2)
        idcg = 0
        for i in range(min(top_k, purchase_items_num)):
            idcg += 1 / np.log2(i + 2)
        metrics_result['ndcg'] += dcg / idcg

    metrics_result['precision'] /= customer_count
    metrics_result['recall'] /= customer_count
    metrics_result['f1'] = (2 * metrics_result['precision'] * metrics_result['recall']) / \
                           (metrics_result['precision'] + metrics_result['recall'])
    metrics_result['hit_rate'] /= customer_count
    metrics_result['ndcg'] /= customer_count

    return metrics_result, time.time() - start_time