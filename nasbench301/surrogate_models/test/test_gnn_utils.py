import unittest

import numpy as np

from nasbench301.surrogate_models.gnn.gnn_utils import NASBenchDataset


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.nasbench_dataset = NASBenchDataset('surrogate_models/test/', result_paths=[
            'surrogate_models/test/results_fidelity_0/results_0.json'], config_space_path='configspace.json')

    def test_length(self):
        self.assertEqual(1, len(self.nasbench_dataset))

    def test_correct_adjacency_matrix(self):
        config_space_instance, val_accuracy, test_accuracy, json_file = self.nasbench_dataset.config_loader[
            'surrogate_models/test/results_fidelity_0/results_0.json']
        normal_cell, reduction_cell = self.nasbench_dataset.create_darts_adjacency_matrix_from_config(
            config_space_instance)
        gt_normal_adjacency_matrix = np.array([[0, 0, 1, 1, 1, 1, 0],
                                               [0, 0, 1, 0, 0, 1, 0],
                                               [0, 0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)
        self.assertTrue((normal_cell[0] == gt_normal_adjacency_matrix).all())

    def test_correct_coo_format(self):
        config_space_instance, val_accuracy, test_accuracy, json_file = self.nasbench_dataset.config_loader[
            'surrogate_models/test/results_fidelity_0/results_0.json']
        normal_cell, _ = self.nasbench_dataset.create_darts_adjacency_matrix_from_config(
            config_space_instance)
        normal_cell_pt = self.nasbench_dataset.convert_to_pytorch_format(normal_cell)

        x = [0., 7., 0., 8., 0., 9., 0., 10., 1., 11., 1., 12., 2., 13., 3., 14., 2., 3., 4., 5.]
        y = [7., 2., 8., 3., 9., 4., 10., 5., 11., 2., 12., 5., 13., 3., 14., 4., 6., 6., 6., 6.]
        expected_coo_format = np.array([x, y], dtype=np.float64)

        self.assertTrue((normal_cell_pt[0] == expected_coo_format).all())


if __name__ == '__main__':
    unittest.main()
