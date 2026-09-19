import unittest
import torch
import torch.nn.functional as F
from inference_engine.attention import flash_attention_triton, flash_decoding_triton


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class AttentionTests(unittest.TestCase):
    @torch.inference_mode()
    def test_prefill_and_decode(self):
        torch.manual_seed(123)
        # BF16 needs Ampere+; FP16 keeps the kernels usable on Turing (T4).
        dtypes = [torch.float16]
        if torch.cuda.get_device_capability()[0] >= 8:
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            for n in (1, 17, 127, 129, 513):
                for d in (32, 64, 128):
                    for decode in (False, True):
                        with self.subTest(dtype=dtype, n=n, d=d, decode=decode):
                            # Strided inputs exercise wrapper strides and partial final tiles.
                            q = torch.randn(1, 2, 1 if decode else n, d * 2, device='cuda', dtype=dtype)[..., ::2]
                            k = torch.randn(1, 2, n, d * 2, device='cuda', dtype=dtype)[..., ::2]
                            v = torch.randn_like(k)
                            fn = flash_decoding_triton if decode else flash_attention_triton
                            actual = fn(q, k, v)
                            ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
                            torch.testing.assert_close(actual.float(), ref, atol=0.015, rtol=0.025)


if __name__ == '__main__':
    unittest.main()
