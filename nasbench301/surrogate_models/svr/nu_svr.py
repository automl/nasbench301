import logging
import os
import pickle

import numpy as np
from sklearn.svm import NuSVR as sklearn_NuSVR
import matplotlib.pyplot as plt

from nasbench301.surrogate_models import utils
from nasbench301.surrogate_models.svr.svr import SVR


class NuSVR(SVR):
    def __init__(self, data_root, log_dir, seed, model_config, data_config):
        super(NuSVR, self).__init__(data_root, log_dir, seed, model_config, data_config)
        # Instantiate model
        svr_config = {k:eval(v) for k,v in model_config.items() if v=='False' or
                      v=='True'}
        self.model = sklearn_NuSVR(**svr_config)

    def train(self):

        from nasbench301.surrogate_models import utils
        X_train, y_train, _ = self.load_results_from_result_paths(self.train_paths)
        X_val, y_val, _ = self.load_results_from_result_paths(self.val_paths)
        self.model.fit(X_train, y_train)

        train_pred, var_train = self.model.predict(X_train), None
        val_pred, var_val = self.model.predict(X_val), None

        #self.save()

        fig_train = utils.scatter_plot(np.array(train_pred), np.array(y_train), xlabel='Predicted', ylabel='True', title='')
        fig_train.savefig(os.path.join(self.log_dir, 'pred_vs_true_train.jpg'))
        plt.close()

        fig_val = utils.scatter_plot(np.array(val_pred), np.array(y_val), xlabel='Predicted', ylabel='True', title='')
        fig_val.savefig(os.path.join(self.log_dir, 'pred_vs_true_val.jpg'))
        plt.close()

        train_metrics = utils.evaluate_metrics(y_train, train_pred, prediction_is_first_arg=False)
        valid_metrics = utils.evaluate_metrics(y_val, val_pred, prediction_is_first_arg=False)

        logging.info('train metrics: %s', train_metrics)
        logging.info('valid metrics: %s', valid_metrics)

        return valid_metrics

    def test(self):
        from nasbench301.surrogate_models import utils
        X_test, y_test, _ = self.load_results_from_result_paths(self.test_paths)
        test_pred, var_test = self.model.predict(X_test), None

        fig = utils.scatter_plot(np.array(test_pred), np.array(y_test), xlabel='Predicted', ylabel='True', title='')
        fig.savefig(os.path.join(self.log_dir, 'pred_vs_true_test.jpg'))
        plt.close()

        test_metrics = utils.evaluate_metrics(y_test, test_pred, prediction_is_first_arg=False)

        logging.info('test metrics %s', test_metrics)

        return test_metrics
