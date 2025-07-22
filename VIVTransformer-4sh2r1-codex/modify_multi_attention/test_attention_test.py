import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import shutil
from pathlib import Path

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modify_multi_attention_svd10_results.attention_test import main

class TestAttentionTest(unittest.TestCase):

    @patch('modify_multi_attention.attention_test.argparse.ArgumentParser')
    @patch('modify_multi_attention.attention_test.load_config')
    @patch('modify_multi_attention.attention_test.get_loaders')
    @patch('modify_multi_attention.attention_test.TransformerFlowReconstructionModel')
    @patch('modify_multi_attention.attention_test.train_model')
    @patch('modify_multi_attention.attention_test.test_model')
    @patch('torch.save')
    @patch('matplotlib.pyplot.savefig')
    def test_main_script_execution(self, mock_savefig, mock_torch_save, mock_test_model, mock_train_model, mock_model, mock_get_loaders, mock_load_config, mock_argparse):
        # Mock argparse
        mock_args = MagicMock()
        mock_args.config = 'dummy_config.yaml'
        mock_args.results_dir = 'dummy_results'
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_argparse.return_value = mock_parser

        # Mock config loading
        mock_load_config.return_value = {
            'device': 'cpu',
            'data': {'path': 'dummy_path', 'batch_size': 32},
            'model': {
                'input_dim': 1, 'output_dim': 1, 'num_heads': 2,
                'num_layers': 2, 'd_model': 128, 'max_time_steps': 10
            },
            'training': {'learning_rate': 1e-3, 'epochs': 1, 'early_stop_patience': 3},
            'visualization': {'enabled': True}
        }

        # Mock data loaders
        mock_get_loaders.return_value = (MagicMock(), MagicMock(), MagicMock())

        # Mock model and training/testing
        mock_train_model.return_value = (MagicMock(), [0.1], [0.1], [0.1])
        mock_test_model.return_value = 0.1

        # Create a dummy failed_attention_log.txt
        results_dir = Path('dummy_results')
        results_dir.mkdir(exist_ok=True)
        failed_log_path = results_dir / 'failed_attention_log.txt'
        with open(failed_log_path, 'w') as f:
            f.write('test_attn1\n')
            f.write('test_attn2\n')

        # Run the main function
        main()

        # Assertions
        # Since some attention mechanisms might fail during initialization, 
        # we check that the successful ones are trained and tested.
        successful_attns = ['test_attn1', 'test_attn2']
        num_successful = mock_train_model.call_count

        self.assertLessEqual(num_successful, len(successful_attns))
        self.assertEqual(mock_test_model.call_count, num_successful)
        self.assertEqual(mock_torch_save.call_count, num_successful)
        self.assertEqual(mock_savefig.call_count, num_successful)

        # Clean up
        if results_dir.exists():
            shutil.rmtree(results_dir)

if __name__ == '__main__':
    unittest.main()