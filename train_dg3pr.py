import os
import time
import torch
import torch_geometric

import numpy as np
import pandas as pd

from utils.logger import Logger
from utils.eval import calculate_score, calculate_metrics
from utils.file_util import load_csv_file, save_to_dill

from knowledge_graph.knowledge_graph_data import KnowledgeGraphData, Sample
from knowledge_graph.knowledge_graph_data_generator import KnowledgeGraphDataGenerator

from models.dg3pr import BPRLoss, DG3PR

from dg3pr_trainer import DG3PRTrainer


if __name__ == '__main__':
    logger = Logger()

    raw_data = load_csv_file('./data/processed_shipping_data.dill')

    raw_data['forsale_breeds_dk'] = raw_data['forsale_breeds_dk'] + '-' + raw_data['breeds_class_nm']
    raw_data['guide_price_round'] = raw_data['guide_price'].apply(lambda x: round(x, 2))
    raw_data.reset_index(drop=True, inplace=True)

    node_mapping = {
        'customer': 'cust_dk',
        'breed': 'forsale_breeds_dk',
        'breed_class': 'breeds_class_nm',
        'guide_price_grade': None,
        'guide_price': 'guide_price_round'
    }

    edge_mapping = {
        'purchased': {'source': {'customer': 'cust_dk'}, 'target': {'breed': 'forsale_breeds_dk'}, 'value': 'sale_qty'},
        'belong_to_1': {'source': {'breed': 'forsale_breeds_dk'}, 'target': {'guide_price_grade': None}, 'value': 'guide_price'},
        'belong_to_2': {'source': {'breed': 'forsale_breeds_dk'}, 'target': {'breed_class': 'breeds_class_nm'}, 'value': 'avg_wt'},
        'belong_to_3': {'source': {'breed': 'forsale_breeds_dk'}, 'target': {'guide_price': 'guide_price_round'}, 'value': 'avg_wt'},
    }

    sample_none_mapping = {
        'user_item': {'source_node_type': 'customer', 'target_node_type': 'breed', 'edge_type': 'purchased'},
        'sub_node_list': [
            {'source_node_type': 'breed', 'target_node_type': 'guide_price_grade', 'edge_type': 'belong_to_1'},
            {'source_node_type': 'breed', 'target_node_type': 'breed_class', 'edge_type': 'belong_to_2'},
            {'source_node_type': 'breed', 'target_node_type': 'guide_price', 'edge_type': 'belong_to_3'},
        ]
    }

    east_province_list = ['上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '台湾']
    middle_province_list = ['河南', '湖北', '湖南', '广东', '广西', '海南', '河北', '山西', '内蒙古']
    west_province_list = ['重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']

    east_data = raw_data[raw_data['province_nm'].isin(east_province_list)]
    middle_data = raw_data[raw_data['province_nm'].isin(middle_province_list)]
    west_data = raw_data[raw_data['province_nm'].isin(west_province_list)]

    use_lstm_list = [False, True]
    grade_num_list = [1, 3, 5]
    kge_embedding_dim_list = [32, 64, 128, 256, 512]
    train_date_days_list = [3, 7, 11, 15]
    eval_date_list = [
        '2021-01-08', '2021-01-09', '2021-01-10', '2021-01-11', '2021-01-12',
        '2021-01-13', '2021-01-14', '2021-01-15', '2021-01-16', '2021-01-17',
    ]

    dataset_list = {'middle': middle_data, 'east': east_data, 'west': west_data}

    for dataset_name, dataset in dataset_list.items():
        for use_lstm in use_lstm_list:
            for grade_num in grade_num_list:
                for kge_embedding_dim in kge_embedding_dim_list:
                    for train_date_days in train_date_days_list:
                        output_path = 'output/kan_train_output_dataset_name-{}_use_lstm-{}_grade_num-{}_kge_dim-{}_train_date_days-{}'.format(dataset_name, use_lstm, grade_num, kge_embedding_dim, train_date_days)
                        metrics_result = {}

                        for eval_date in eval_date_list:
                            eval_date = pd.to_datetime(eval_date)
                            start_date = eval_date - pd.DateOffset(days=train_date_days)
                            end_date = eval_date - pd.DateOffset(days=1)
                            start_date = start_date.strftime('%Y-%m-%d')
                            end_date = end_date.strftime('%Y-%m-%d')
                            eval_date = eval_date.strftime('%Y-%m-%d')

                            logger.divider('Training Date: {} - {}'.format(start_date, end_date))
                            trainer = DG3PRTrainer(
                                raw_data=dataset.copy(),
                                date_range=(start_date, end_date),
                                top_ks=[5, 10, 15, 20, 25, 30],
                                epochs=50,
                                batch_size=1024,
                                use_lstm=use_lstm,
                                learning_rate=0.01,
                                node_mapping=node_mapping,
                                edge_mapping=edge_mapping,
                                sample_none_mapping=sample_none_mapping,
                                kge_embedding_dim=kge_embedding_dim,
                                negative_sample_num=1,
                                discrete_value_columns=[
                                    {
                                        'column_name': 'guide_price',
                                        'new_column_name': 'guide_price_grade',
                                        'grade_num': grade_num,
                                        'group_column_names': ['breeds_class_nm', 'org_inv_dk']
                                    }
                                ],
                                save_path=output_path,
                                logger=logger
                            )
                            tmp_metrics_result = trainer.train()
                            metrics_result.setdefault(eval_date, tmp_metrics_result)
                            logger.divider('Training Date: {} - {} Done'.format(start_date, end_date), end=True)

                        # to dataframe
                        metrics_result_dataframe = pd.DataFrame()
                        for eval_date, top_k_metrics in metrics_result.items():
                            for top_k, metrics in top_k_metrics.items():
                                metrics_result_dataframe.loc[eval_date, f'precision@{top_k}'] = metrics['precision']
                                metrics_result_dataframe.loc[eval_date, f'recall@{top_k}'] = metrics['recall']
                                metrics_result_dataframe.loc[eval_date, f'f1@{top_k}'] = metrics['f1']
                                metrics_result_dataframe.loc[eval_date, f'hit_rate@{top_k}'] = metrics['hit_rate']
                                metrics_result_dataframe.loc[eval_date, f'ndcg@{top_k}'] = metrics['ndcg']
                        metrics_result_dataframe.to_csv(os.path.join(output_path, 'metrics_result_tmp.csv'))
                        logger.divider('Training Done: dataset_name-{}_use_lstm-{}_grade_num-{}_kge_dim-{}_train_date_days-{}'.format(dataset_name, use_lstm, grade_num, kge_embedding_dim, train_date_days), end=True)