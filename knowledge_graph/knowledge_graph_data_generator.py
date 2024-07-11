import time
import jenkspy
import numpy as np
import pandas as pd

import torch
import torch_geometric

from sklearn.preprocessing import LabelEncoder

from utils.logger import Logger
from utils.data_util import discrete_value_to_grade

from knowledge_graph.knowledge_graph_data import KnowledgeGraphData, SampleNoneInfo, Sample
from knowledge_graph.knowledge_graph_trainer import KnowledgeGraphTrainer


class KnowledgeGraphDataGenerator(object):

    def __init__(
            self,
            raw_data: pd.DataFrame,
            node_mapping: dict = None,
            edge_mapping: dict = None,
            sample_none_mapping: dict = None,
            kge_model_name: str = 'DistMult',
            kge_embedding_dim: int = 128,
            kge_epochs: tuple = (200, 100),
            kge_batch_size: int = 1024,
            negative_sample_num: int = 1,
            logger: Logger = None
    ):
        self.raw_data = raw_data
        self.node_mapping = node_mapping
        self.edge_mapping = edge_mapping
        self.sample_none_mapping = sample_none_mapping
        self.negative_sample_num = negative_sample_num
        self.logger = logger if logger is not None else Logger(root_path='../log')

        self.kge_model_name = kge_model_name
        self.kge_embedding_dim = kge_embedding_dim
        self.kge_epochs = kge_epochs
        self.kge_batch_size = kge_batch_size
        self.knowledge_graph_trainer = KnowledgeGraphTrainer(device='cuda:0', logger=self.logger)

        if self.node_mapping is None:
            self.node_mapping = {
                'customer': 'cust_dk',
                'breed': 'forsale_breeds_dk',
                'breed_class': 'breeds_class_nm',
                'guide_price_grade': None
            }
        if self.edge_mapping is None:
            self.edge_mapping = {
                'purchased': {'source': {'customer': 'cust_dk'}, 'target': {'breed': 'forsale_breeds_dk'}, 'value': 'sale_qty'},
                'belong_to_1': {'source': {'breed': 'forsale_breeds_dk'}, 'target': {'guide_price_grade': None}, 'value': 'guide_price'},
                'belong_to_2': {'source': {'breed': 'forsale_breeds_dk'}, 'target': {'breed_class': 'breeds_class_nm'}, 'value': 'avg_wt'}
            }
        if self.sample_none_mapping is None:
            self.sample_none_mapping = {
                'user_item': {'source_node_type': 'customer', 'target_node_type': 'breed', 'edge_type': 'purchased'},
                'sub_node_list': [
                    {'source_node_type': 'breed', 'target_node_type': 'guide_price_grade', 'edge_type': 'belong_to_1'},
                    {'source_node_type': 'breed', 'target_node_type': 'breed_class', 'edge_type': 'belong_to_2'}
                ]
            }

        self.logger.divider('Start check mapping')
        is_valid, msg = self._check_mapping()
        if not is_valid:
            self.logger.error(msg)
            raise ValueError(msg)
        else:
            self.logger.info(msg)
        self.logger.divider('End check mapping')

    @staticmethod
    def _check_node_mapping(raw_data_columns: list, node_mapping: dict) -> bool:
        """
        check if node_mapping is valid
        :param raw_data_columns:
        :param node_mapping:
        :return:
        """

        for _, column_name in node_mapping.items():
            if column_name is None:
                continue
            if column_name not in raw_data_columns:
                return False

        return True

    def _check_mapping(self) -> (bool, str):
        """
        check if node_mapping, edge_mapping is valid
        :return:
        """

        raw_data_columns = self.raw_data.columns.tolist()

        # check node_mapping
        if not self._check_node_mapping(raw_data_columns, self.node_mapping):
            return False, 'node_mapping is invalid'

        # check edge_mapping
        for _, edge in self.edge_mapping.items():
            if not self._check_node_mapping(raw_data_columns, edge['source']):
                return False, 'source node_mapping is invalid'
            if not self._check_node_mapping(raw_data_columns, edge['target']):
                return False, 'target node_mapping is invalid'

        return True, 'node_mapping and edge_mapping is valid'

    def _extract_node_data(self, raw_data: pd.DataFrame) -> tuple:
        """
        extract node data
        :param raw_data:
        :return:
        """

        node_name_map = {}
        node_name_list = []
        for node_type, column_name in self.node_mapping.items():
            if column_name is None:
                column_name = node_type
            tmp_node_name_list = [
                '{}_{}'.format(node_type, node_name) for node_name in raw_data[column_name].unique().tolist()
            ]
            node_name_map[node_type] = tmp_node_name_list
            node_name_list.extend(tmp_node_name_list)
            self.logger.info('Node type: {}, number of nodes: {}'.format(node_type, len(tmp_node_name_list)))

        return node_name_map, node_name_list

    def _extract_edge_data(self, raw_data: pd.DataFrame, node_name_id_map: dict) -> tuple:
        """
        extract edge data
        :param raw_data:
        :return:
        """

        edge_map = {}
        edge_value_map = {}
        edge_node_type_map = {}
        for edge_type, tmp_edge_map in self.edge_mapping.items():
            source_column_type = list(tmp_edge_map['source'].keys())[0]
            target_column_type = list(tmp_edge_map['target'].keys())[0]

            if source_column_type not in node_name_id_map or target_column_type not in node_name_id_map:
                continue

            source_column_name = tmp_edge_map['source'][source_column_type]
            target_column_name = tmp_edge_map['target'][target_column_type]

            if source_column_name is None:
                source_column_name = source_column_type
            if target_column_name is None:
                target_column_name = target_column_type

            edge_node_type_map[edge_type] = {
                'source': source_column_type,
                'target': target_column_type
            }

            tmp_raw_data = raw_data[[source_column_name, target_column_name, tmp_edge_map['value']]]
            tmp_raw_data = tmp_raw_data.drop_duplicates()
            tmp_raw_data = tmp_raw_data.reset_index(drop=True)
            for index, row in tmp_raw_data.iterrows():
                if pd.isna(row[source_column_name]) or pd.isna(row[target_column_name]):
                    continue

                source_node_name = '{}_{}'.format(source_column_type, row[source_column_name])
                target_node_name = '{}_{}'.format(target_column_type, row[target_column_name])
                if source_node_name not in node_name_id_map[source_column_type] or target_node_name not in node_name_id_map[target_column_type]:
                    continue

                source_node_id = node_name_id_map[source_column_type][source_node_name]
                target_node_id = node_name_id_map[target_column_type][target_node_name]
                edge_map.setdefault(edge_type, []).append((source_node_id, target_node_id))
                edge_value_map.setdefault(edge_type, {}).setdefault((source_node_id, target_node_id), row[tmp_edge_map['value']])

            self.logger.info('Edge type: {}, number of edges: {}'.format(edge_type, len(edge_map[edge_type])))

        return edge_map, edge_value_map, edge_node_type_map

    def _generate_sample(
            self,
            source_node_id: int,
            target_node_id: int,
            knowledge_graph_data: KnowledgeGraphData
    ) -> Sample or None:
        user_item_mapping = self.sample_none_mapping['user_item']
        sub_node_mapping_list = self.sample_none_mapping['sub_node_list']
        user_item_node_info = SampleNoneInfo(
            source_node_id=source_node_id,
            source_node_type=user_item_mapping['source_node_type'],
            target_node_id=target_node_id,
            target_node_type=user_item_mapping['target_node_type'],
            edge_type=user_item_mapping['edge_type'],
            value=knowledge_graph_data.get_edge_value(
                source_node_id=source_node_id, target_node_id=target_node_id, edge_type=user_item_mapping['edge_type']
            )
        )

        sub_node_info_list = []
        for sub_node_mapping in sub_node_mapping_list:
            if sub_node_mapping['source_node_type'] == user_item_mapping['source_node_type']:
                tmp_source_node_id = source_node_id
            elif sub_node_mapping['source_node_type'] == user_item_mapping['target_node_type']:
                tmp_source_node_id = target_node_id
            else:
                raise ValueError('Invalid source node type: {}'.format(sub_node_mapping['source_node_type']))
            tmp_target_node_id_list = knowledge_graph_data.get_target_node_list(
                source_node_id=tmp_source_node_id, edge_type=sub_node_mapping['edge_type']
            )
            if len(tmp_target_node_id_list) == 0:
                # tmp_source_node_name = knowledge_graph_data.get_node_name_by_node_id(tmp_source_node_id)
                # self.logger.warring('No target node for source node: {}, edge type: {}'.format(tmp_source_node_name, sub_node_mapping['edge_type']))
                return None

            tmp_target_node_id = tmp_target_node_id_list[0]
            sub_node_info_list.append(SampleNoneInfo(
                source_node_id=tmp_source_node_id,
                source_node_type=sub_node_mapping['source_node_type'],
                target_node_id=tmp_target_node_id,
                target_node_type=sub_node_mapping['target_node_type'],
                edge_type=sub_node_mapping['edge_type'],
                value=knowledge_graph_data.get_edge_value(
                    source_node_id=tmp_source_node_id, target_node_id=tmp_target_node_id,
                    edge_type=sub_node_mapping['edge_type']
                )
            ))

        return Sample(
            user_item_node_info=user_item_node_info,
            sub_node_info_list=sub_node_info_list
        )

    def _generate_knowledge_graph_data(
            self,
            graph_name: str,
            date_range: tuple,
            date_column: str = 'shipping_dt',
            discrete_value_columns: list = None,
            graph_type: str = 'global',
            global_node_id_encoder: LabelEncoder = None
    ) -> (KnowledgeGraphData, LabelEncoder):
        """
        generate knowledge graph data
        :param date_range:
        :param date_column:
        :param discrete_value_columns: [
            {'column_name': 'column_name', 'grade_num': 5, 'group_column_names': ['group_column_name', ...]},
        ]
        :return:
        """

        if graph_type not in ['global', 'dynamic']:
            raise ValueError('graph_type is invalid')
        if graph_type == 'dynamic' and global_node_id_encoder is None:
            raise ValueError('global_node_id_encoder is required when graph_type is dynamic')

        tmp_raw_data = self.raw_data.copy()
        tmp_raw_data = tmp_raw_data[
            (tmp_raw_data[date_column] >= date_range[0]) & (tmp_raw_data[date_column] <= date_range[1])
        ]

        # discrete value to grade
        for discrete_value_column in discrete_value_columns:
            tmp_raw_data[
                discrete_value_column['new_column_name']
            ] = discrete_value_to_grade(
                raw_data=tmp_raw_data,
                group_column_names=discrete_value_column['group_column_names'],
                column_name=discrete_value_column['column_name'],
                new_column_name=discrete_value_column['new_column_name'],
                grade_num=discrete_value_column['grade_num']
            )

        # extract node data
        node_name_map, node_name_list = self._extract_node_data(tmp_raw_data)

        unseen_node_list = []
        if global_node_id_encoder is not None:
            unseen_node_list = [
                node_name for node_name in node_name_list if node_name not in global_node_id_encoder.classes_
            ]
            node_name_list = list(set(node_name_list) - set(unseen_node_list))

        if len(unseen_node_list) > 0:
            self.logger.warring('Unseen node list: {}'.format(unseen_node_list))

        node_name_id_map = {}
        node_id_encoder = LabelEncoder()
        node_id_encoder.fit(node_name_list)
        for node_type, tmp_node_name_list in node_name_map.items():
            tmp_node_name_list = list(set(tmp_node_name_list) & set(node_name_list))
            node_name_id_map[node_type] = dict(zip(tmp_node_name_list, node_id_encoder.transform(tmp_node_name_list)))

        # extract edge data
        edge_map, edge_value_map, edge_node_type_map = self._extract_edge_data(tmp_raw_data, node_name_id_map)

        # generate global node id map
        if graph_type == 'global':
            node_id = node_id_encoder.transform(node_name_list)
            global_node_id_map = {node_id: node_id for node_id in node_id}
        else:
            global_node_id_map = {}
            for node_name in node_name_list:
                node_id = node_id_encoder.transform([node_name])[0]
                global_node_id = global_node_id_encoder.transform([node_name])[0]
                global_node_id_map[node_id] = global_node_id

        # generate knowledge graph data - node
        geometric_data_x = []
        geometric_data_node_index = []
        node_type_encoder = LabelEncoder()
        node_type_encoder.fit(list(node_name_id_map.keys()))
        for node_type, name_id_map in node_name_id_map.items():
            node_type_id = node_type_encoder.transform([node_type])[0]
            for _, node_id in name_id_map.items():
                geometric_data_x.append([node_type_id, node_id])
                geometric_data_node_index.append(node_id)
        geometric_data_x = torch.tensor(geometric_data_x, dtype=torch.long)
        geometric_data_node_index = torch.tensor(geometric_data_node_index, dtype=torch.long)

        # generate knowledge graph data - edge
        geometric_data_edge_type = []
        graph_data_edge_index = [[], []]
        edge_type_encoder = LabelEncoder()
        edge_type_encoder.fit(list(edge_map.keys()))
        for edge_type, edge_list in edge_map.items():
            edge_type_id = edge_type_encoder.transform([edge_type])[0]
            for edge in edge_list:
                geometric_data_edge_type.append(edge_type_id)
                graph_data_edge_index[0].append(edge[0])
                graph_data_edge_index[1].append(edge[1])
        geometric_data_edge_type = torch.tensor(geometric_data_edge_type, dtype=torch.long)
        graph_data_edge_index = torch.tensor(graph_data_edge_index, dtype=torch.long)

        # generate knowledge graph data - graph
        geometric_data = torch_geometric.data.Data(
            x=geometric_data_x,
            node_index=geometric_data_node_index,
            edge_index=graph_data_edge_index,
            edge_type=geometric_data_edge_type,
            num_nodes=geometric_data_x.shape[0],
            num_edges=geometric_data_edge_type.shape[0]
        )

        # train node embedding and edge embedding
        node_embedding, edge_embedding = self.knowledge_graph_trainer.train(
            geometric_data=geometric_data,
            model_name=self.kge_model_name,
            embedding_dim=self.kge_embedding_dim,
            epochs=self.kge_epochs[0] if graph_type == 'global' else self.kge_epochs[1],
            batch_size=self.kge_batch_size,
            verbose=False
        )

        knowledge_graph_data = KnowledgeGraphData(
            graph_name=graph_name,
            geometric_data=geometric_data,
            edge_map=edge_map,
            edge_value_map=edge_value_map,
            edge_node_type_map=edge_node_type_map,
            node_name_id_map=node_name_id_map,
            global_node_id_map=global_node_id_map,
            unseen_node_list=unseen_node_list,
            node_embedding=node_embedding,
            edge_embedding=edge_embedding
        )

        # generate samples
        samples = {}
        for index, (source_node_id, target_node_id) in enumerate(knowledge_graph_data.get_edge_list_by_type('purchased')):
            positive_sample = self._generate_sample(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                knowledge_graph_data=knowledge_graph_data
            )

            if positive_sample is None:
                continue

            # negative sample
            negative_sample_list = []
            negative_target_node_id_list = knowledge_graph_data.generate_negative_sample(
                source_node_id=source_node_id,
                edge_type='purchased',
                negative_sample_num=self.negative_sample_num
            )
            for negative_target_node_id in negative_target_node_id_list:
                negative_sample = self._generate_sample(
                    source_node_id=source_node_id,
                    target_node_id=negative_target_node_id,
                    knowledge_graph_data=knowledge_graph_data
                )
                negative_sample_list.append(negative_sample)

            samples[index] = {
                'positive_sample': positive_sample,
                'negative_sample_list': negative_sample_list
            }

        knowledge_graph_data.set_samples(samples)

        return knowledge_graph_data, node_id_encoder

    def generate_knowledge_graph_data(
            self,
            date_range: tuple,
            date_column: str = 'shipping_dt',
            discrete_value_columns: list = None
    ) -> (KnowledgeGraphData, dict, KnowledgeGraphData):
        """
        generate knowledge graph data
        :param date_range:
        :param date_column:
        :param discrete_value_columns: [
            {'column_name': 'column_name', 'grade_num': 5, 'group_column_names': ['group_column_name', ...]},
        ]
        :return:
        """

        self.logger.divider('Start generate knowledge graph data')
        eval_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

        self.logger.info('Start generate global knowledge graph data')
        start_time = time.time()
        global_knowledge_graph_data, global_node_id_encoder = self._generate_knowledge_graph_data(
            graph_name='{}-{}_global_knowledge_graph'.format(date_range[0], date_range[1]),
            date_range=date_range,
            date_column=date_column,
            discrete_value_columns=discrete_value_columns,
            graph_type='global'
        )
        self.logger.info('Global Graph of date {}-{} node num: {}, edge num: {}'.format(
            date_range[0], date_range[1],
            global_knowledge_graph_data.geometric_data.num_nodes,
            global_knowledge_graph_data.geometric_data.num_edges
        ))
        self.logger.info('Global Graph of date {}-{} Purchased edge num: {}'.format(
            date_range[0], date_range[1],
            len(global_knowledge_graph_data.edge_map['purchased']))
        )
        self.logger.info('End generate global knowledge graph, cost time: {:.2f}s'.format(time.time() - start_time))

        self.logger.info('Start generate dynamic knowledge graph data')
        start_time = time.time()
        dynamic_knowledge_graph_data_dict = {}
        eval_knowledge_graph_data = None
        for date in pd.date_range(date_range[0], eval_date):
            self.logger.info('Generate dynamic knowledge graph data for date: {}'.format(date))
            date = date.strftime('%Y-%m-%d')
            dynamic_knowledge_graph_data, _ = self._generate_knowledge_graph_data(
                graph_name='{}_dynamic_knowledge_graph'.format(date),
                date_range=(date, date),
                date_column=date_column,
                discrete_value_columns=discrete_value_columns,
                graph_type='dynamic',
                global_node_id_encoder=global_node_id_encoder
            )
            if date == eval_date.strftime('%Y-%m-%d'):
                eval_knowledge_graph_data = dynamic_knowledge_graph_data

                # eval samples
                samples = {}
                samples_count = 0
                user_item_mapping = self.sample_none_mapping['user_item']
                sub_node_mapping_list = self.sample_none_mapping['sub_node_list']
                for customer_node_id in eval_knowledge_graph_data.get_node_id_list_by_type('customer'):
                    global_customer_node_id = eval_knowledge_graph_data.get_global_node_id_by_node_id(customer_node_id)
                    if global_customer_node_id is None:
                        continue

                    for breed_node_id in eval_knowledge_graph_data.get_node_id_list_by_type('breed'):
                        global_breed_node_id = eval_knowledge_graph_data.get_global_node_id_by_node_id(breed_node_id)
                        if global_breed_node_id is None:
                            continue

                        purchased = eval_knowledge_graph_data.check_edge_existence(
                            edge_type='purchased', source_node_id=customer_node_id, target_node_id=breed_node_id
                        )

                        user_item_node_info = SampleNoneInfo(
                            source_node_id=global_customer_node_id,
                            source_node_type=user_item_mapping['source_node_type'],
                            target_node_id=global_breed_node_id,
                            target_node_type=user_item_mapping['target_node_type'],
                            edge_type=user_item_mapping['edge_type'],
                            value=eval_knowledge_graph_data.get_edge_value(
                                source_node_id=customer_node_id, target_node_id=breed_node_id, edge_type=user_item_mapping['edge_type']
                            )
                        )

                        sub_node_info_list = []
                        for sub_node_mapping in sub_node_mapping_list:
                            if sub_node_mapping['source_node_type'] == user_item_mapping['source_node_type']:
                                tmp_source_node_id = customer_node_id
                                global_tmp_source_node_id = global_customer_node_id
                            elif sub_node_mapping['source_node_type'] == user_item_mapping['target_node_type']:
                                tmp_source_node_id = breed_node_id
                                global_tmp_source_node_id = global_breed_node_id
                            else:
                                raise ValueError('Invalid source node type: {}'.format(sub_node_mapping['source_node_type']))

                            tmp_target_node_id_list = eval_knowledge_graph_data.get_target_node_list(
                                source_node_id=tmp_source_node_id, edge_type=sub_node_mapping['edge_type']
                            )

                            if len(tmp_target_node_id_list) == 0:
                                # tmp_source_node_name = eval_knowledge_graph_data.get_node_name_by_node_id(tmp_source_node_id)
                                # self.logger.warring('No target node for source node: {}, edge type: {}'.format(tmp_source_node_name, sub_node_mapping['edge_type']))
                                continue

                            tmp_target_node_id = tmp_target_node_id_list[0]
                            global_tmp_target_node_id = eval_knowledge_graph_data.get_global_node_id_by_node_id(
                                node_id=tmp_target_node_id
                            )
                            sub_node_info_list.append(SampleNoneInfo(
                                source_node_id=global_tmp_source_node_id,
                                source_node_type=sub_node_mapping['source_node_type'],
                                target_node_id=global_tmp_target_node_id,
                                target_node_type=sub_node_mapping['target_node_type'],
                                edge_type=sub_node_mapping['edge_type'],
                                value=eval_knowledge_graph_data.get_edge_value(
                                    source_node_id=tmp_source_node_id, target_node_id=tmp_target_node_id,
                                    edge_type=sub_node_mapping['edge_type']
                                )
                            ))

                        if len(sub_node_info_list) != len(sub_node_mapping_list):
                            continue

                        sample = Sample(
                            user_item_node_info=user_item_node_info,
                            sub_node_info_list=sub_node_info_list
                        )
                        samples[samples_count] = {
                            'sample': sample,
                            'purchased': purchased
                        }
                        samples_count += 1
                eval_knowledge_graph_data.set_samples(samples)
            else:
                dynamic_knowledge_graph_data_dict[date] = dynamic_knowledge_graph_data
            self.logger.info('Dynamic Graph of date {} node num: {}, edge num: {}'.format(
                date,
                dynamic_knowledge_graph_data.geometric_data.num_nodes,
                dynamic_knowledge_graph_data.geometric_data.num_edges
            ))
            self.logger.info('Dynamic Graph of date {} Purchased edge num: {}'.format(
                date,
                len(dynamic_knowledge_graph_data.edge_map['purchased']))
            )
        self.logger.info('End generate dynamic knowledge graph, cost time: {:.2f}s'.format(time.time() - start_time))

        self.logger.divider('End generate knowledge graph data')

        return global_knowledge_graph_data, dynamic_knowledge_graph_data_dict, eval_knowledge_graph_data