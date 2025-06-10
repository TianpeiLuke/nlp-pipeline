# test/test_dist_utils.py
import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np

# Import the functions to be tested
from src.lightning_models.dist_utils import (
    get_world_size,
    get_rank,
    is_main_process,
    synchronize,
    all_gather,
    gather,
    shared_random_seed,
    reduce_dict,
    print_gpu_memory_usage,
    print_gpu_memory_stats
)

# A mock for the torch.distributed module
class DistMock:
    def __init__(self, rank=0, world_size=1, backend='nccl'):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
    
    def get_rank(self, group=None):
        return self.rank
    
    def get_world_size(self, group=None):
        return self.world_size
    
    def is_available(self):
        return True
        
    def is_initialized(self):
        return self.world_size > 1

    def get_backend(self):
        return self.backend
        
    def barrier(self, *args, **kwargs):
        pass # Mock barrier
        
    def all_gather_object(self, output_list, obj, group=None):
        # Simulate all_gather by populating the output list
        for i in range(len(output_list)):
            output_list[i] = obj
            
    def gather_object(self, data, output_list=None, dst=0, group=None):
        # Simulate gather by populating the output list if we are the destination rank
        if self.rank == dst and output_list is not None:
            for i in range(len(output_list)):
                output_list[i] = data
                
    def reduce(self, tensor, dst, op=None, group=None):
        # The real `reduce` operation sums tensors from all processes to the destination.
        # The user code then handles averaging. For this mock, since we are only on one process,
        # we don't need to modify the tensor. The user code will correctly average it.
        pass

    def new_group(self, *args, **kwargs):
        return "mock_group"


class TestDistUtils(unittest.TestCase):

    # --- Test in a simulated single-process environment ---
    @patch('src.lightning_models.dist_utils.dist', DistMock(world_size=1))
    def test_single_process_environment(self):
        self.assertEqual(get_world_size(), 1)
        self.assertEqual(get_rank(), 0)
        self.assertTrue(is_main_process())
        # Test all_gather in single process
        self.assertEqual(all_gather("data"), ["data"])
        # Test gather in single process
        self.assertEqual(gather("data"), ["data"])
        # Test reduce_dict in single process
        input_dict = {'a': torch.tensor(1.0), 'b': torch.tensor(2.0)}
        self.assertEqual(reduce_dict(input_dict), input_dict)

    # --- Test in a simulated multi-process (distributed) environment ---
    @patch('src.lightning_models.dist_utils.dist', DistMock(rank=1, world_size=4))
    def test_multi_process_environment(self):
        self.assertEqual(get_world_size(), 4)
        self.assertEqual(get_rank(), 1)
        self.assertFalse(is_main_process())

    @patch('src.lightning_models.dist_utils.dist')
    def test_synchronize(self, mock_dist):
        # Test with NCCL backend
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 2
        mock_dist.get_backend.return_value = "nccl"
        synchronize()
        mock_dist.barrier.assert_called_once()

        # Test with Gloo backend
        mock_dist.reset_mock()
        mock_dist.get_backend.return_value = "gloo"
        synchronize()
        mock_dist.barrier.assert_called_once()

    @patch('src.lightning_models.dist_utils.dist', DistMock(rank=1, world_size=4))
    def test_all_gather_multi_process(self):
        data_to_gather = "test_data"
        result = all_gather(data_to_gather)
        self.assertEqual(len(result), 4)
        self.assertEqual(result, ["test_data", "test_data", "test_data", "test_data"])

    @patch('src.lightning_models.dist_utils.dist', DistMock(rank=1, world_size=4))
    def test_gather_non_destination_rank(self):
        """Test gather on a rank that is not the destination."""
        data_to_gather = "test_data"
        result = gather(data_to_gather, dst=0)
        self.assertEqual(result, [])

    @patch('src.lightning_models.dist_utils.dist', DistMock(rank=0, world_size=4))
    def test_gather_destination_rank(self):
        """Test gather on the destination rank."""
        data_to_gather = "test_data"
        result = gather(data_to_gather, dst=0)
        self.assertEqual(len(result), 4)
        self.assertEqual(result, ["test_data", "test_data", "test_data", "test_data"])

    @patch('src.lightning_models.dist_utils.all_gather')
    def test_shared_random_seed(self, mock_all_gather):
        mock_all_gather.return_value = [123, 456, 789]
        seed = shared_random_seed()
        self.assertEqual(seed, 123)
        mock_all_gather.assert_called_once()
        
    @patch('src.lightning_models.dist_utils.dist', DistMock(rank=0, world_size=2))
    def test_reduce_dict_multi_process(self):
        """Test the reduce_dict function in a simulated multi-process environment."""
        input_dict = {'loss': torch.tensor(2.0), 'acc': torch.tensor(0.8)}
        
        # Call the function being tested
        result_dict = reduce_dict(input_dict, average=True)
        
        # Check the *returned* dictionary, which contains the averaged values.
        self.assertIn('acc', result_dict)
        self.assertIn('loss', result_dict)
        self.assertAlmostEqual(result_dict['acc'].item(), 0.4)
        self.assertAlmostEqual(result_dict['loss'].item(), 1.0)

    @patch('src.lightning_models.dist_utils.torch.cuda')
    @patch('builtins.print')
    def test_print_gpu_memory_usage(self, mock_print, mock_cuda):
        """Test that the GPU memory printing function calls print with expected stats."""
        mock_cuda.is_available.return_value = True
        mock_cuda.memory_allocated.return_value = 1024**2 * 100 # 100 MB
        mock_cuda.memory_reserved.return_value = 1024**2 * 200 # 200 MB
        mock_cuda.max_memory_allocated.return_value = 1024**2 * 150
        mock_cuda.max_memory_reserved.return_value = 1024**2 * 250
        
        print_gpu_memory_usage(device_id=0)
        
        # Check if print was called multiple times
        self.assertGreater(mock_print.call_count, 4)
        
        # Check if some expected substrings are in the printed output
        all_printed_text = " ".join([call.args[0] for call in mock_print.call_args_list])
        self.assertIn("Allocated: 100.00 MB", all_printed_text)
        self.assertIn("Reserved:  200.00 MB", all_printed_text)
        self.assertIn("Max Allocated: 150.00 MB", all_printed_text)

    @patch('src.lightning_models.dist_utils.torch.cuda')
    @patch('builtins.print')
    def test_print_gpu_memory_stats(self, mock_print, mock_cuda):
        """Test the GPU memory stats printing function."""
        mock_cuda.is_available.return_value = True
        mock_cuda.memory_stats.return_value = {"allocated_bytes.all.current": 100}

        print_gpu_memory_stats(device_id=0)
        mock_cuda.memory_stats.assert_called_once_with(device=0)
        
        # Check if print was called with the stats
        all_printed_text = " ".join([call.args[0] for call in mock_print.call_args_list])
        self.assertIn("allocated_bytes.all.current: 100", all_printed_text)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)