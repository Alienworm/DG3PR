import time
import numpy as np
from typing import Tuple

import torch
import torch_geometric
from torch_geometric.nn.kge import KGEModel, TransE, DistMult, ComplEx
from torch_geometric.nn.models import LightGCN

from utils.logger import Logger


class KnowledgeGraphTrainer(object):

    def __init__(self, device: str, logger: Logger = None):
        self.device = device
        self.logger = logger or Logger(root_path='../log')
        self.model_mapping: Dict[str, KGEModel] = {
            'TransE': TransE,
            'DistMult': DistMult,
            'ComplEx': ComplEx
        }

    def evaluate(
            self,
            epoch: int,
            kge_model: KGEModel,
            knowledge_graph: torch_geometric.data.Data,
            eval_batch_size: int = 1024,
            eval_topk: list = None
    ):
        self.logger.divider('Evaluating Knowledge Graph Embedding Model on Epoch: {}'.format(epoch))
        for topk in eval_topk:
            eval_result = kge_model.test(
                head_index=knowledge_graph.edge_index[0],
                rel_type=knowledge_graph.edge_type,
                tail_index=knowledge_graph.edge_index[1],
                batch_size=eval_batch_size,
                k=topk,
                log=False
            )
            self.logger.info('Top-{} Evaluation Result: {}'.format(topk, eval_result))
        self.logger.divider('Evaluation Finished')

    def train(
            self,
            geometric_data: torch_geometric.data.Data,
            model_name: str = 'TransE',
            embedding_dim: int = 64,
            epochs: int = 100,
            batch_size: int = 1024,
            learning_rate: float = 0.01,
            verbose: bool = True,
            eval_interval: int = 10,
            eval_batch_size: int = 1024,
            eval_topk: list = None,
            eval_verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train a knowledge graph embedding model.
        :param geometric_data:
        :param model_name:
        :param embedding_dim:
        :param epochs:
        :param batch_size:
        :param learning_rate:
        :param verbose:
        :param eval_interval:
        :param eval_batch_size:
        :param eval_topk:
        :param eval_verbose:
        :return:
        """

        if eval_topk is None:
            eval_topk = [5, 10, 15, 20, 25, 30]

        self.logger.divider('Training Knowledge Graph Embedding Model')
        self.logger.info('KGE Model: {}, Embedding Dimension: {}, Epochs: {}, Batch Size: {}, Learning Rate: {}'.format(
            model_name, embedding_dim, epochs, batch_size, learning_rate
        ))

        start_time = time.time()

        if model_name not in self.model_mapping:
            raise ValueError(f'Invalid model name: {model_name}')

        geometric_data = geometric_data.to(self.device)
        kge_model: KGEModel = self.model_mapping[model_name](
            num_nodes=geometric_data.num_nodes,
            num_relations=geometric_data.num_edges,
            hidden_channels=embedding_dim,
        ).to(self.device)

        data_loader = kge_model.loader(
            head_index=geometric_data.edge_index[0],
            rel_type=geometric_data.edge_type,
            tail_index=geometric_data.edge_index[1],
            batch_size=batch_size,
            shuffle=True
        )

        optimizer_mapping = {
            'TransE': torch.optim.Adam(kge_model.parameters(), lr=learning_rate),
            'DistMult': torch.optim.Adam(kge_model.parameters(), lr=learning_rate, weight_decay=1e-6),
            'ComplEx': torch.optim.Adagrad(kge_model.parameters(), lr=learning_rate, weight_decay=1e-6)
        }
        optimizer = optimizer_mapping[model_name]

        for epoch in range(epochs):
            kge_model.train()
            total_loss = 0
            for head_index, rel_type, tail_index in data_loader:
                optimizer.zero_grad()
                loss = kge_model.loss(head_index, rel_type, tail_index)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose:
                self.logger.info('Epoch: {}, Loss: {:.4f}'.format(epoch, total_loss))

            if (epoch % eval_interval == 0 and eval_verbose) or (epoch == epochs):
                kge_model.eval()
                self.evaluate(
                    epoch=epoch,
                    kge_model=kge_model,
                    knowledge_graph=geometric_data,
                    eval_batch_size=eval_batch_size,
                    eval_topk=eval_topk
                )

        self.logger.divider('Training Finished, Cost Time: {:.4f}s'.format(time.time() - start_time))

        # get node embedding and edge embedding
        kge_model.eval()
        with torch.no_grad():
            node_embedding = kge_model.node_emb.weight.cpu().detach().numpy()
            edge_embedding = kge_model.rel_emb.weight.cpu().detach().numpy()

        return node_embedding, edge_embedding
