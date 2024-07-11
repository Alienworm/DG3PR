import torch

import numpy as np
import torch_geometric

from dataclasses import dataclass
from typing import List, Dict, Any, Union

# from utils.data_util import dataset_to_tensor


torch.manual_seed(0)
np.random.seed(0)


class SampleNoneInfo(object):

    def __init__(
            self,
            source_node_id: int or torch.Tensor,
            source_node_type: str,
            target_node_id: int or torch.Tensor,
            target_node_type: str,
            edge_type: str,
            value: float or torch.Tensor = None
    ):
        self.source_node_id = source_node_id
        self.source_node_type = source_node_type
        self.target_node_id = target_node_id
        self.target_node_type = target_node_type
        self.edge_type = edge_type
        self.value = -1 if value is None else value

    def to(self, device):
        self.source_node_id = torch.tensor(self.source_node_id, dtype=torch.long, device=device)
        self.target_node_id = torch.tensor(self.target_node_id, dtype=torch.long, device=device)
        self.value = torch.tensor(self.value, dtype=torch.float, device=device)

        return SampleNoneInfo(
            source_node_id=self.source_node_id,
            source_node_type=self.source_node_type,
            target_node_id=self.target_node_id,
            target_node_type=self.target_node_type,
            edge_type=self.edge_type,
            value=self.value
        )


class Sample(object):

    def __init__(self, user_item_node_info: SampleNoneInfo, sub_node_info_list: List[SampleNoneInfo]):
        self.user_item_node_info = user_item_node_info
        self.sub_node_info_list = sub_node_info_list

    def to(self, device):
        self.user_item_node_info = self.user_item_node_info.to(device)
        self.sub_node_info_list = [
            sub_node_info.to(device) for sub_node_info in self.sub_node_info_list
        ]
        return Sample(
            user_item_node_info=self.user_item_node_info,
            sub_node_info_list=self.sub_node_info_list
        )


class KnowledgeGraphDataset(torch.utils.data.Dataset):

    def __init__(self, samples: dict, negative_sample_num: int, device: str):
        self.samples = samples
        self.negative_sample_num = negative_sample_num
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        positive_sample = self.samples[idx]['positive_sample']
        positive_user_id = positive_sample.user_item_node_info.source_node_id
        positive_item_id = positive_sample.user_item_node_info.target_node_id
        positive_sub_node_id_list = [
            sub_node_info.target_node_id for sub_node_info in positive_sample.sub_node_info_list
        ]
        positive_value = positive_sample.user_item_node_info.value

        negative_sample_list = []
        for negative_sample in self.samples[idx]['negative_sample_list']:
            negative_user_id = negative_sample.user_item_node_info.source_node_id
            negative_item_id = negative_sample.user_item_node_info.target_node_id
            negative_sub_node_id_list = [
                sub_node_info.target_node_id for sub_node_info in negative_sample.sub_node_info_list
            ]
            negative_value = -1 * positive_value
            negative_sample_list.append((negative_user_id, negative_item_id, negative_sub_node_id_list, negative_value))

        return positive_user_id, positive_item_id, positive_sub_node_id_list, positive_value, negative_sample_list


class KnowledgeGraphEvalDataset(torch.utils.data.Dataset):

    def __init__(self, samples: dict, device: str):
        self.samples = samples
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.samples[idx]['sample']
        purchased = self.samples[idx]['purchased']

        user_id = sample.user_item_node_info.source_node_id
        item_id = sample.user_item_node_info.target_node_id
        sub_node_id_list = [
            sub_node_info.target_node_id for sub_node_info in sample.sub_node_info_list
        ]
        value = sample.user_item_node_info.value

        return user_id, item_id, sub_node_id_list, value, purchased


class KnowledgeGraphData(object):

    def __init__(
            self,
            graph_name: str,
            geometric_data: torch_geometric.data.Data,
            edge_map: dict,
            edge_value_map: dict,
            edge_node_type_map: dict,
            node_name_id_map: dict,
            global_node_id_map: dict,
            unseen_node_list: list,
            node_embedding: np.ndarray = None,
            edge_embedding: np.ndarray = None,
            samples: dict = None
    ):
        self.graph_name = graph_name
        self.geometric_data = geometric_data

        # {edge_type: [(source_node_id, target_node_id), ...], ...}
        self.edge_map = edge_map
        # {edge_type: {(source_node_id, target_node_id): value, ...}, ...}
        self.edge_value_map = edge_value_map
        # {edge_type: {'source': source_node_type, 'target': target_node_type}, ...}
        self.edge_node_type_map = edge_node_type_map

        # {node_type: {node_name: node_id}, ...}
        self.node_name_id_map = node_name_id_map
        # {node_type: {node_id: node_name}, ...}
        self.node_id_name_map = {k: {v: k for k, v in v.items()} for k, v in node_name_id_map.items()}
        # {node_id: {'node_type': node_type, 'node_name': node_name}, ...}
        self.node_id_info_map = {
            node_id: {'node_type': node_type, 'node_name': node_name} for node_type, node_name_id in node_name_id_map.items() for node_name, node_id in node_name_id.items()
        }

        # {node_id: global_node_id, ...}
        self.global_node_id_map = global_node_id_map
        # {global_node_id: node_id, ...}
        self.global_node_id_reverse_map = {v: k for k, v in global_node_id_map.items()}

        # unseen node list
        self.unseen_node_list = unseen_node_list

        self.node_embedding = node_embedding
        self.edge_embedding = edge_embedding

        self.samples = samples

    def get_graph_name(self) -> str:
        return self.graph_name

    def get_geometric_data(self) -> torch_geometric.data.Data:
        return self.geometric_data

    def get_edge_type(self, source_node_id: int, target_node_id: int) -> str or None:
        for edge_type, edge_list in self.edge_map.items():
            if (source_node_id, target_node_id) in edge_list:
                return edge_type
        return None

    def get_edge_value(self, source_node_id: int, target_node_id: int, edge_type: str = None) -> float or None:
        if edge_type is None:
            edge_type = self.get_edge_type(source_node_id=source_node_id, target_node_id=target_node_id)
        if edge_type is not None and (source_node_id, target_node_id) in self.edge_value_map[edge_type]:
            return self.edge_value_map[edge_type][(source_node_id, target_node_id)]
        return None

    def get_edge_type_list(self) -> list:
        return list(self.edge_map.keys())

    def get_edge_list_by_type(self, edge_type: str) -> list or None:
        if edge_type not in self.edge_map:
            return None
        return self.edge_map[edge_type]

    def get_edge_node_type_by_edge_type(self, edge_type: str) -> dict or None:
        if edge_type not in self.edge_node_type_map:
            return None
        return self.edge_node_type_map[edge_type]

    def check_edge_existence(self, edge_type: str, source_node_id: int, target_node_id: int) -> bool:
        if edge_type not in self.edge_map:
            return False
        return (source_node_id, target_node_id) in self.edge_map[edge_type]

    def get_node_type_list(self) -> list:
        return list(self.node_name_id_map.keys())

    def get_node_id_list_by_type(self, node_type: str) -> list or None:
        if node_type not in self.node_name_id_map:
            return None
        return list(self.node_name_id_map[node_type].values())

    def get_node_name_by_node_id(self, node_id: int) -> str or None:
        if node_id not in self.node_id_info_map:
            return None
        return self.node_id_info_map[node_id]['node_name']

    def get_node_id_by_name_and_type(self, node_name: str, node_type: str) -> int or None:
        if node_type not in self.node_name_id_map:
            return None
        if node_name not in self.node_name_id_map[node_type]:
            return None
        return self.node_name_id_map[node_type][node_name]

    def get_node_type_by_node_id(self, node_id: int) -> str or None:
        if node_id not in self.node_id_info_map:
            return None
        return self.node_id_info_map[node_id]['node_type']

    def get_node_id_by_global_node_id(self, global_node_id: int) -> int or None:
        if global_node_id not in self.global_node_id_reverse_map:
            return None
        return self.global_node_id_reverse_map[global_node_id]

    def get_global_node_id_by_node_id(self, node_id: int) -> int or None:
        if node_id not in self.global_node_id_map:
            return None
        return self.global_node_id_map[node_id]

    def get_node_name_by_global_node_id(self, global_node_id: int) -> str or None:
        node_id = self.get_node_id_by_global_node_id(global_node_id)
        return self.get_node_name_by_node_id(node_id)

    def get_global_node_id_by_node_name_and_type(self, node_name: str, node_type: str) -> int or None:
        node_id = self.get_node_id_by_name_and_type(node_name, node_type)
        return self.get_global_node_id_by_node_id(node_id)

    def get_node_type_by_global_node_id(self, global_node_id: int) -> str or None:
        node_id = self.get_node_id_by_global_node_id(global_node_id)
        return self.get_node_type_by_node_id(node_id)

    def get_target_node_list(self, source_node_id: int, edge_type: str) -> list or None:
        if edge_type not in self.edge_map:
            return None
        return [
            _target_node_id for _source_node_id, _target_node_id in self.edge_map[edge_type]
            if source_node_id == _source_node_id
        ]

    def get_node_embedding(self) -> np.ndarray:
        return self.node_embedding

    def get_edge_embedding(self) -> np.ndarray:
        return self.edge_embedding

    def get_node_embedding_by_node_id(self, node_id: int) -> np.ndarray or None:
        if self.node_embedding is None:
            return None
        return self.node_embedding[node_id]

    def get_node_embedding_by_global_node_id(self, global_node_id: int) -> np.ndarray or None:
        node_id = self.get_node_id_by_global_node_id(global_node_id)
        return self.get_node_embedding_by_node_id(node_id)

    def set_node_embedding(self, node_embedding: np.ndarray) -> None:
        self.node_embedding = node_embedding

    def set_edge_embedding(self, edge_embedding: np.ndarray) -> None:
        self.edge_embedding = edge_embedding

    def get_unseen_node_list(self) -> list:
        return self.unseen_node_list

    def generate_negative_sample(self, source_node_id: int, edge_type: str, negative_sample_num: int) -> list:
        negative_sample_list = []
        target_node_list = self.get_target_node_list(source_node_id=source_node_id, edge_type=edge_type)
        target_node_type = self.get_edge_node_type_by_edge_type(edge_type)['target']
        all_target_node_list = self.get_node_id_list_by_type(target_node_type)
        while len(negative_sample_list) < negative_sample_num:
            negative_sample = np.random.choice(all_target_node_list)
            if negative_sample not in target_node_list:
                negative_sample_list.append(negative_sample)
        return negative_sample_list

    def set_samples(self, samples: dict) -> None:
        self.samples = samples

    def get_samples(self) -> dict or None:
        return self.samples

    def __str__(self):
        return f'KnowledgeGraphData(graph_name={self.graph_name}, node_num={self.geometric_data.num_nodes}, edge_num={self.geometric_data.num_edges})'