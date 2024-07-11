import torch
import torch_geometric

import numpy as np

from knowledge_graph.knowledge_graph_data import KnowledgeGraphData, Sample


class BPRLoss(torch.nn.Module):

    def __init__(self, gamma: float = 1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class DG3PR(torch.nn.Module):

    def __init__(
            self,
            global_graph: KnowledgeGraphData,
            dynamic_graph_dict: dict,
            output_layer_node_num: int = None,
            lstm_hidden_num_layers: int = 1,
            batch_first: bool = True,
            use_lstm: bool = True
    ):
        super(DG3PR, self).__init__()

        self.global_graph = global_graph
        self.node_type_list = global_graph.get_node_type_list()
        self.output_layer_node_num = len(self.node_type_list) if output_layer_node_num is None else output_layer_node_num
        self.node_embedding_dim = global_graph.get_node_embedding().shape[1]
        self.dynamic_graph_dict = dynamic_graph_dict
        self.use_lstm = use_lstm

        self.lstm_layers = torch.nn.ModuleDict()
        for node_type in self.node_type_list:
            self.lstm_layers[node_type] = torch.nn.LSTM(
                input_size=self.node_embedding_dim,
                hidden_size=self.node_embedding_dim,
                num_layers=lstm_hidden_num_layers,
                batch_first=batch_first
            ).to('cuda')

        self.multi_head_attention_layers = torch.nn.ModuleDict()
        for node_type in self.node_type_list:
            self.multi_head_attention_layers[node_type] = torch.nn.Sequential(
                torch.nn.Linear(self.node_embedding_dim * 2, self.node_embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                torch.nn.Sigmoid()
            ).to('cuda')

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.node_embedding_dim * output_layer_node_num, self.node_embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.node_embedding_dim, 1)
        )

        self.node_embedding_map = torch.nn.Embedding(
            self.global_graph.get_geometric_data().num_nodes,
            self.node_embedding_dim,
            sparse=True
        )
        self.update_node_embedding()

    def update_node_embedding(self):
        if not self.use_lstm:
            self.node_embedding_map = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(self.global_graph.get_node_embedding()).to('cuda')
            )
            return

        date_index_map = {
            date: index for index, date in enumerate(self.dynamic_graph_dict.keys())
        }

        # use LSTM to aggregate dynamic node embeddings
        dynamic_node_embedding_map = {}
        for node_type, lstm_layer in self.lstm_layers.items():
            for global_node_id in self.global_graph.get_node_id_list_by_type(node_type):
                dynamic_node_embedding_list = [
                    np.zeros(self.node_embedding_dim) for _ in range(len(self.dynamic_graph_dict))
                ]

                for date in self.dynamic_graph_dict.keys():
                    dynamic_graph = self.dynamic_graph_dict[date]
                    dynamic_node_id = dynamic_graph.get_node_id_by_global_node_id(global_node_id)
                    if dynamic_node_id is not None:
                        dynamic_node_embedding_list[date_index_map[date]] = dynamic_graph.get_node_embedding_by_node_id(dynamic_node_id)

                lstm_input = torch.tensor(dynamic_node_embedding_list, dtype=torch.float32).to('cuda').view(
                    len(dynamic_node_embedding_list), 1, self.node_embedding_dim
                )

                output, (_, _) = lstm_layer(lstm_input)
                dynamic_node_embedding_map[global_node_id] = output[-1]

        # use multi-head attention to aggregate dynamic node embeddings and global node embeddings
        node_embedding_map = {}
        for node_type, multi_head_attention_layer in self.multi_head_attention_layers.items():
            for global_node_id in self.global_graph.get_node_id_list_by_type(node_type):
                global_node_embedding = torch.from_numpy(self.global_graph.get_node_embedding_by_node_id(global_node_id)).to('cuda')
                dynamic_node_embedding = dynamic_node_embedding_map[global_node_id][0]

                node_embedding = torch.cat([global_node_embedding, dynamic_node_embedding], dim=0)
                attention_weight = multi_head_attention_layer(node_embedding)
                node_embedding_map[global_node_id] = attention_weight * dynamic_node_embedding + \
                                                      (1 - attention_weight) * global_node_embedding

        # update node embedding
        self.node_embedding_map = torch.nn.Embedding.from_pretrained(
            torch.stack([node_embedding_map[node_id] for node_id in range(self.global_graph.get_geometric_data().num_nodes)]
        ))

    def forward(self, user_id, item_id, sub_node_id_list, value):
        user_node_embedding = self.node_embedding_map(user_id)
        item_node_embedding = self.node_embedding_map(item_id)
        user_item_node_embedding = torch.cat([user_node_embedding, item_node_embedding], dim=1)

        sub_node_embedding_list = [
            self.node_embedding_map(sub_node_id) for sub_node_id in sub_node_id_list
        ]

        sub_node_embedding = torch.stack(sub_node_embedding_list)
        sub_node_embedding = sub_node_embedding.view(user_id.shape[0], -1)

        output = self.output_layer(torch.cat([user_item_node_embedding, sub_node_embedding], dim=1))

        return output