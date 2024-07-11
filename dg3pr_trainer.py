import gevent
from gevent.pool import Pool
from gevent import monkey
monkey.patch_all()

import os
import time
import torch
import pandas as pd

from tqdm import tqdm

from utils.logger import Logger
from utils.file_util import save_to_dill
from utils.eval import calculate_score, calculate_metrics
from utils.data_util import dataset_to_tensor

from models.dg3pr import BPRLoss, DG3PR

from knowledge_graph.knowledge_graph_data import KnowledgeGraphData, Sample, SampleNoneInfo, KnowledgeGraphDataset
from knowledge_graph.knowledge_graph_data_generator import KnowledgeGraphDataGenerator


torch.manual_seed(0)


class DG3PRTrainer(object):

    def __init__(
            self,
            raw_data: pd.DataFrame,
            date_range: tuple,
            top_ks: list,
            epochs: int = 100,
            batch_size: int = 128,
            learning_rate: float = 0.05,
            use_lstm: bool = False,
            lstm_hidden_num_layers: int = 1,
            negative_sample_num: int = 1,
            device: str = 'cuda',
            save_path: str = 'train_output',
            node_mapping: dict = None,
            edge_mapping: dict = None,
            sample_none_mapping: dict = None,
            discrete_value_columns: list = None,
            kge_model_name: str = 'DistMult',
            kge_embedding_dim: int = 128,
            kge_epochs: tuple = (200, 100),
            kge_batch_size: int = 1024,
            logger: Logger = None
    ):
        self.raw_data = raw_data
        self.date_range = date_range
        self.top_ks = top_ks
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_lstm = use_lstm
        self.lstm_hidden_num_layers = lstm_hidden_num_layers
        self.learning_rate = learning_rate
        self.negative_sample_num = negative_sample_num
        self.device = device
        self.save_path = save_path + '/{}'.format(date_range[0] + '-' + date_range[1])

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.node_mapping = node_mapping
        self.edge_mapping = edge_mapping
        self.sample_none_mapping = sample_none_mapping
        self.discrete_value_columns = discrete_value_columns
        self.kge_model_name = kge_model_name
        self.kge_embedding_dim = kge_embedding_dim
        self.kge_epochs = kge_epochs
        self.kge_batch_size = kge_batch_size

        self.logger = logger if logger is not None else Logger(root_path='../log')

    def _train_data_preprocessing(self) -> tuple:
        self.logger.divider('Start data preprocessing', flag='*', end=False)
        knowledge_graph_data_generator = KnowledgeGraphDataGenerator(
            raw_data=self.raw_data,
            node_mapping=self.node_mapping,
            edge_mapping=self.edge_mapping,
            sample_none_mapping=self.sample_none_mapping,
            kge_model_name=self.kge_model_name,
            kge_embedding_dim=self.kge_embedding_dim,
            kge_epochs=self.kge_epochs,
            kge_batch_size=self.kge_batch_size,
            negative_sample_num=self.negative_sample_num,
            logger=self.logger
        )

        global_knowledge_graph_data, dynamic_knowledge_graph_data_dict, eval_knowledge_graph_data = \
            knowledge_graph_data_generator.generate_knowledge_graph_data(
                date_range=self.date_range,
                discrete_value_columns=self.discrete_value_columns
            )

        save_to_dill(global_knowledge_graph_data, os.path.join(self.save_path, 'global_knowledge_graph_data.dill'))
        # save_to_dill(dynamic_knowledge_graph_data_dict, os.path.join(self.save_path, 'dynamic_knowledge_graph_data_dict.dill'))
        save_to_dill(eval_knowledge_graph_data, os.path.join(self.save_path, 'eval_knowledge_graph_data.dill'))
        self.logger.divider('End data preprocessing', flag='*', end=False)

        return global_knowledge_graph_data, dynamic_knowledge_graph_data_dict, eval_knowledge_graph_data

    def train(self):
        global_knowledge_graph_data, dynamic_knowledge_graph_data_dict, eval_knowledge_graph_data = \
            self._train_data_preprocessing()

        start_time = time.time()
        self.logger.divider('Start Training...')
        samples = global_knowledge_graph_data.get_samples()
        samples = dataset_to_tensor(samples, device=self.device, dataset_type='train')
        train_dataset = KnowledgeGraphDataset(
            samples=samples,
            negative_sample_num=self.negative_sample_num,
            device=self.device
        )
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        eval_samples = eval_knowledge_graph_data.get_samples()
        eval_samples = dataset_to_tensor(eval_samples, device=self.device, dataset_type='eval')
        eval_knowledge_graph_data.set_samples(eval_samples)

        model = DG3PR(
            global_graph=global_knowledge_graph_data,
            dynamic_graph_dict=dynamic_knowledge_graph_data_dict,
            output_layer_node_num=2 + len(self.sample_none_mapping['sub_node_list']),
            lstm_hidden_num_layers=self.lstm_hidden_num_layers,
            batch_first=True,
            use_lstm=self.use_lstm
        ).to(self.device)

        bpr_loss = BPRLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=5e-8, betas=(0.5, 0.99))

        self.logger.info('Done init train params, Cost time: {:.2f}s'.format(time.time() - start_time))

        best_loss, best_model, best_epoch = float('inf'), None, 0
        for epoch in range(self.epochs):
            start_time = time.time()

            model.train()
            total_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
            for positive_user_id, positive_item_id, positive_sub_node_id_list, positive_value,\
                negative_sample_list in train_dataloader:

                optimizer.zero_grad()

                tmp_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
                for negative_user_id, negative_item_id, negative_sub_node_id_list, negative_value in negative_sample_list:
                    positive_score = model(positive_user_id, positive_item_id, positive_sub_node_id_list, positive_value)
                    negative_score = model(negative_user_id, negative_item_id, negative_sub_node_id_list, negative_value)
                    loss = bpr_loss(positive_score, negative_score)
                    tmp_loss = tmp_loss + loss

                tmp_loss.backward()
                optimizer.step()

                total_loss = total_loss + tmp_loss.item()

            if self.use_lstm:
                model.update_node_embedding()

            self.logger.info('Epoch: {}, Loss: {}, Lr: {}, Cost time: {:.2f}s'.format(epoch, total_loss.item(), optimizer.param_groups[0]['lr'], time.time() - start_time))
            self.logger.tensorboard_logger.add_scalar('BPR Loss', total_loss.item(), epoch)

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_model = model
                best_epoch = epoch

            model.eval()
            score_dataframe, eval_cost_time = calculate_score(
                knowledge_graph_data=eval_knowledge_graph_data,
                model=model
            )
            for top_k in self.top_ks:
                metrics, metrics_cost_time = calculate_metrics(score_dataframe, top_k=top_k)
                self.logger.info('Epoch: {}, Top-{} Metrics: {}'.format(epoch, top_k, metrics))
                for metric_name, metric_value in metrics.items():
                    self.logger.tensorboard_logger.add_scalar('{}@{}'.format(metric_name, top_k), metric_value, epoch)
                eval_cost_time += metrics_cost_time

            self.logger.info('Epoch: {}, Eval Cost Time: {:.2f}s'.format(epoch, eval_cost_time))

        best_model.eval()
        score_dataframe, eval_cost_time = calculate_score(
            knowledge_graph_data=eval_knowledge_graph_data,
            model=best_model
        )
        metrics_result = {}
        for top_k in self.top_ks:
            metrics, metrics_cost_time = calculate_metrics(score_dataframe, top_k=top_k)
            self.logger.info('Best Epoch: {} Top-{} Metrics: {}'.format(best_epoch, top_k, metrics))
            metrics_result.setdefault(top_k, metrics)
            eval_cost_time += metrics_cost_time

        torch.save(best_model, os.path.join(self.save_path, 'best_model.pth'))
        self.logger.info('Eval Cost Time: {:.2f}s'.format(eval_cost_time))

        return metrics_result
