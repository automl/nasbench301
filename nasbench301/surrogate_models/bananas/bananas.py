import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nasbench301.surrogate_models import utils
from nasbench301.surrogate_models.bananas.bananas_utils import BANANASDataset, BANANASPT
from nasbench301.surrogate_models.gnn.gnn_utils import Patience
from nasbench301.surrogate_models.surrogate_model import SurrogateModel


class BANANASModel(SurrogateModel):
    def __init__(self, data_root, log_dir, seed, model_config, data_config):
        super(BANANASModel, self).__init__(data_root, log_dir, seed, model_config, data_config)
        test_queue = self.load_results_from_result_paths(['surrogate_models/test/results_fidelity_0/results_0.json'])
        Xs, ys = next(iter(test_queue))
        self.device = torch.device('cpu')

        # Instantiate the GNN
        model = BANANASPT(Xs.shape[-1], num_layers=self.model_config['num_layers'],
                          layer_width=self.model_config['layer_width'])
        self.model = model.to(self.device)

    def load_results_from_result_paths(self, result_paths):
        # Instantiate dataset
        dataset = BANANASDataset(result_paths=result_paths, config_loader=self.config_loader)

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.model_config['batch_size'], pin_memory=True)
        return dataloader

    def train(self):
        '''
        if self.model_config['loss_function'] == 'L1':
            criterion = torch.nn.L1Loss()
        elif self.model_config['loss_function'] == 'L2':
            criterion = torch.nn.MSELoss()
        elif self.model_config['loss_function'] == 'HUBER':
            criterion = torch.nn.SmoothL1Loss()
        else:
            raise NotImplementedError('Unknown loss function used.')
        '''
        # Create early stopper
        early_stopper = Patience(patience=20, use_loss=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.model_config['epochs'], eta_min=self.model_config['learning_rate_min'])

        # Load training data
        train_queue = self.load_results_from_result_paths(self.train_paths)
        valid_queue = self.load_results_from_result_paths(self.val_paths)

        # Start train loop
        for epoch in tqdm(range(self.model_config['epochs'])):
            logging.info('Starting epoch {}'.format(epoch))
            lr = scheduler.get_last_lr()[0]

            # training
            train_obj, train_results = self.train_epoch(train_queue, valid_queue, self.model, None, optimizer, lr,
                                                        epoch)

            logging.info('train metrics: %s', train_results)
            scheduler.step()

            # validation
            valid_obj, valid_results = self.infer(train_queue, valid_queue, self.model, None, optimizer, lr, epoch)
            logging.info('validation metrics: %s', valid_results)

            # save the model
            # self.save()

            # Early Stopping
            if early_stopper is not None and early_stopper.stop(epoch, val_loss=valid_obj,
                                                                val_acc=valid_results["kendall_tau"]):
                logging.info(
                    'Early Stopping at epoch {}, best is {}'.format(epoch, early_stopper.get_best_vl_metrics()))
                break

        return valid_results

    def normalize_data(self, val_accuracy, val_min=None):
        if val_min is None:
            return torch.log(1 - val_accuracy)
        else:
            return torch.log(1 - val_accuracy / val_min)

    def unnormalize_data(self, normalized_accuracy):
        return 1 - np.exp(normalized_accuracy)

    def train_epoch(self, train_queue, valid_queue, model, criterion, optimizer, lr, epoch):
        objs = utils.AvgrageMeter()

        # TRAINING
        preds = []
        targets = []

        model.train()

        for step, (arch_path_enc, y_true) in enumerate(train_queue):
            arch_path_enc = arch_path_enc.to(self.device).float()
            y_true = y_true.to(self.device).float()

            pred = self.model(arch_path_enc)
            if self.model_config['loss:loss_log_transform']:
                loss = torch.mean(torch.abs((self.normalize_data(pred) / self.normalize_data(y_true / 100)) - 1))
            else:
                loss = criterion(1 - pred, 1 - y_true / 100)
            if self.model_config['loss:pairwise_ranking_loss']:
                m = 0.1
                pairwise_ranking_loss = []
                sort_idx = torch.argsort(y_true, descending=True)
                for idx, idx_y_i in enumerate(sort_idx):
                    for idx_y_i_p1 in sort_idx[idx + 1:]:
                        pairwise_ranking_loss.append(torch.max(torch.tensor(0.0, dtype=torch.float),
                                                               m - (pred[idx_y_i] - pred[idx_y_i_p1])))
                pairwise_ranking_loss = torch.mean(torch.stack(pairwise_ranking_loss))

                loss += pairwise_ranking_loss
                if step % self.data_config['report_freq'] == 0:
                    logging.info('Pairwise ranking loss {}'.format(pairwise_ranking_loss))

            preds.extend(pred.detach().cpu().numpy() * 100)
            targets.extend(y_true.detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            objs.update(loss.data.item(), len(arch_path_enc))

            if step % self.data_config['report_freq'] == 0:
                logging.info('train %03d %e', step, objs.avg)

        fig = utils.scatter_plot(np.array(preds), np.array(targets), xlabel='Predicted', ylabel='True', title='')
        fig.savefig(os.path.join(self.log_dir, 'pred_vs_true_train_{}.jpg'.format(epoch)))
        plt.close()
        train_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)

        return objs.avg, train_results

    def infer(self, train_queue, valid_queue, model, criterion, optimizer, lr, epoch):
        objs = utils.AvgrageMeter()

        # VALIDATION
        preds = []
        targets = []

        for step, (arch_path_enc, y_true) in enumerate(valid_queue):
            arch_path_enc = arch_path_enc.to(self.device).float()
            y_true = y_true.to(self.device).float()
            pred = self.model(arch_path_enc)
            loss = torch.mean(torch.abs((self.normalize_data(pred) / self.normalize_data(y_true / 100)) - 1))
            preds.extend(pred.detach().cpu().numpy() * 100)
            targets.extend(y_true.detach().cpu().numpy())
            objs.update(loss.data.item(), len(arch_path_enc))

            if step % self.data_config['report_freq'] == 0:
                logging.info('valid %03d %e ', step, objs.avg)

        fig = utils.scatter_plot(np.array(preds), np.array(targets), xlabel='Predicted', ylabel='True', title='')
        fig.savefig(os.path.join(self.log_dir, 'pred_vs_true_valid_{}.jpg'.format(epoch)))
        plt.close()

        val_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)

        return objs.avg, val_results

    def test(self):
        preds = []
        targets = []
        self.model.eval()

        test_queue = self.load_results_from_result_paths(self.test_paths)
        for step, (arch_path_enc, y_true) in enumerate(test_queue):
            arch_path_enc = arch_path_enc.to(self.device).float()
            y_true = y_true.to(self.device).float()
            pred = self.model(arch_path_enc)
            preds.extend(pred.detach().cpu().numpy() * 100)
            targets.extend(y_true.detach().cpu().numpy())

        fig = utils.scatter_plot(np.array(preds), np.array(targets), xlabel='Predicted', ylabel='True', title='')
        fig.savefig(os.path.join(self.log_dir, 'pred_vs_true_test.jpg'))
        plt.close()

        test_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
        logging.info('test metrics %s', test_results)

        return test_results

    def validate(self):
        preds = []
        targets = []
        self.model.eval()

        valid_queue = self.load_results_from_result_paths(self.val_paths)
        for step, (arch_path_enc, y_true) in enumerate(valid_queue):
            arch_path_enc = arch_path_enc.to(self.device).float()
            y_true = y_true.to(self.device).float()

            pred = self.model(arch_path_enc)
            preds.extend(pred.detach().cpu().numpy() * 100)
            targets.extend(y_true.detach().cpu().numpy())

        fig = utils.scatter_plot(np.array(preds), np.array(targets), xlabel='Predicted', ylabel='True', title='')
        fig.savefig(os.path.join(self.log_dir, 'pred_vs_true_valid.jpg'))
        plt.close()

        val_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
        logging.info('validation metrics %s', val_results)

        return val_results

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'surrogate_model.model'))

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def evaluate(self, result_paths):
        # Get evaluation data
        eval_queue = self.load_results_from_result_paths(result_paths)

        preds = []
        targets = []
        self.model.eval()
        for step, (arch_path_enc, y_true) in enumerate(eval_queue):
            arch_path_enc = arch_path_enc.to(self.device).float()
            y_true = y_true.to(self.device).float()
            pred = self.model(arch_path_enc)
            preds.extend(pred.detach().cpu().numpy() * 100)
            targets.extend(y_true.detach().cpu().numpy())

        test_metrics = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
        return test_metrics, preds, targets

    def query(self, config_dict):
        # Get evaluation data
        config_space_instance = self.config_loader.query_config_dict(config_dict)
        dataset = BANANASDataset(result_paths=None, config_loader=self.config_loader)

        bananas_enc = dataset.convert_to_bananas_paths_format(config_space_instance)
        single_item_batch = torch.from_numpy(bananas_enc).reshape(1, -1)

        self.model.eval()
        single_item_batch = single_item_batch.to(self.device)

        if self.model_config['model'] == 'gnn_vs_gae_classifier':
            pred_bin, pred_normalized = self.model(graph_batch=single_item_batch)
            pred = self.unnormalize_data(pred_normalized)
        else:
            pred = self.model(graph_batch=single_item_batch) * 100
        pred = pred.detach().cpu().numpy()

        return pred
