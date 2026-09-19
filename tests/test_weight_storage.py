import unittest
import torch
from benchmarks.weights import storage_bytes


class StorageTests(unittest.TestCase):
    def test_tied_storage_counted_once_and_buffers_included(self):
        model = torch.nn.Module()
        model.a = torch.nn.Parameter(torch.zeros(10, dtype=torch.float16))
        model.b = torch.nn.Parameter(model.a[:5])
        model.register_buffer('buffer', torch.ones(3, dtype=torch.float32))
        self.assertEqual(storage_bytes(model), 20 + 12)

    def test_quantization_metadata_included(self):
        class State:
            def as_dict(self, packed=False):
                return {'absmax': scale, 'nested': [code]}
        scale = torch.ones(2, dtype=torch.float32)
        code = torch.ones(16, dtype=torch.float32)
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.zeros(8, dtype=torch.uint8), requires_grad=False)
        model.weight.quant_state = State()
        self.assertEqual(storage_bytes(model), 8 + 8 + 64)


if __name__ == '__main__':
    unittest.main()
