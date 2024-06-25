import numpy as np
import torch
import unittest

from deepdna.nn.layers import MultiHeadAttention, RelativeMultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):

    def test_io_projection_shapes(self):
        embed_dim = 64
        num_heads = 8
        layer = MultiHeadAttention(embed_dim, num_heads)
        self.assertEqual(layer.w_query.in_features, embed_dim)
        self.assertEqual(layer.w_query.out_features, embed_dim)
        self.assertEqual(layer.w_key.in_features, embed_dim)
        self.assertEqual(layer.w_key.out_features, embed_dim)
        self.assertEqual(layer.w_value.in_features, embed_dim)
        self.assertEqual(layer.w_value.out_features, embed_dim)
        self.assertEqual(layer.w_output.in_features, embed_dim)
        self.assertEqual(layer.w_output.out_features, embed_dim)

    def test_io_projection_shapes_with_explicit_head_embed_dim(self):
        embed_dim = 64
        num_heads = 8
        head_dim = 64
        layer = MultiHeadAttention(embed_dim, num_heads, head_embed_dim=head_dim)
        self.assertEqual(layer.w_query.in_features, embed_dim)
        self.assertEqual(layer.w_query.out_features, head_dim*num_heads)
        self.assertEqual(layer.w_key.in_features, embed_dim)
        self.assertEqual(layer.w_key.out_features, head_dim*num_heads)
        self.assertEqual(layer.w_value.in_features, embed_dim)
        self.assertEqual(layer.w_value.out_features, head_dim*num_heads)
        self.assertEqual(layer.w_output.in_features, head_dim*num_heads)
        self.assertEqual(layer.w_output.out_features, embed_dim)

    def test_merge_mask(self):
        layer = MultiHeadAttention(64, 8)
        attention_mask = None
        key_padding_mask = None
        self.assertIsNone(layer.merge_mask(attention_mask, key_padding_mask))

        torch.manual_seed(0)
        attention_mask = torch.randint(0, 2, (2, 10, 10), dtype=torch.bool)
        key_padding_mask = None
        self.assertTrue(torch.all(layer.merge_mask(attention_mask, key_padding_mask) == attention_mask))

        torch.manual_seed(0)
        attention_mask = None
        key_padding_mask = torch.randint(0, 2, (2, 10), dtype=torch.bool)
        self.assertTrue(torch.all(layer.merge_mask(attention_mask, key_padding_mask) == torch.repeat_interleave(key_padding_mask.unsqueeze(-2), key_padding_mask.shape[-1], -2)))

        torch.manual_seed(0)
        attention_mask = torch.randint(0, 2, (2, 10, 10), dtype=torch.bool)
        key_padding_mask = torch.randint(0, 2, (2, 10), dtype=torch.bool)
        self.assertTrue(torch.all(layer.merge_mask(attention_mask, key_padding_mask) == torch.logical_or(attention_mask, torch.repeat_interleave(key_padding_mask.unsqueeze(-2), key_padding_mask.shape[-1], -2))))

    def test_compute_attention_weights(self):
        layer = MultiHeadAttention(64, 8)

        # Test without masking
        torch.manual_seed(0)
        query = torch.rand(2, 10, 64)
        key = torch.rand(2, 10, 64)
        value = torch.rand(2, 10, 64)
        attention_mask = None
        key_padding_mask = None
        output, attention_weights = layer(query, key, value, key_padding_mask=key_padding_mask, attention_mask=attention_mask, return_attention_weights=True)
        self.assertEqual(output.shape, (2, 10, 64))
        self.assertEqual(attention_weights.shape, (2, 10, 10))

        # Test with attention_mask and key_padding_mask
        torch.manual_seed(0)
        query = torch.rand(2, 10, 64)
        key = torch.rand(2, 10, 64)
        value = torch.rand(2, 10, 64)
        attention_mask = torch.randint(0, 10, (2, 10, 10)) > 7
        key_padding_mask = torch.randint(0, 10, (2, 10,)) > 7
        output, attention_weights = layer(query, key, value, key_padding_mask=key_padding_mask, attention_mask=attention_mask, return_attention_weights=True)
        mask: torch.Tensor = layer.merge_mask(attention_mask, key_padding_mask) # type: ignore
        self.assertEqual(output.shape, (2, 10, 64))
        self.assertEqual(attention_weights.shape, (2, 10, 10))
        self.assertTrue(torch.all(attention_weights[torch.where(mask)] == 0.0))

        # Test with head mask
        torch.manual_seed(0)
        query = torch.rand(2, 10, 64)
        key = torch.rand(2, 10, 64)
        value = torch.rand(2, 10, 64)
        attention_mask = None
        key_padding_mask = None
        attention_head_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        output, attention_weights = layer(query, key, value, key_padding_mask=key_padding_mask, attention_mask=attention_mask, attention_head_mask=attention_head_mask, average_attention_weights=False, return_attention_weights=True)
        self.assertEqual(output.shape, (2, 10, 64))
        self.assertEqual(attention_weights.shape, (2, 8, 10, 10))
        self.assertTrue(torch.all(attention_weights[:,-1,:,:] == 0.0))

    def test_gradients_are_not_none_and_not_nan(self):
        layer = MultiHeadAttention(64, 8)
        torch.manual_seed(0)
        query = torch.rand(2, 10, 64, requires_grad=True)
        key = torch.rand(2, 10, 64, requires_grad=True)
        value = torch.rand(2, 10, 64, requires_grad=True)
        output = layer(query, key, value)
        output.sum().backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertFalse(torch.any(torch.isnan(query.grad)))
        self.assertFalse(torch.any(torch.isnan(key.grad)))
        self.assertFalse(torch.any(torch.isnan(value.grad)))


class TestRelativeMultiHeadAttention(unittest.TestCase):

    def test_io_projection_shapes(self):
        embed_dim = 64
        num_heads = 8
        layer = RelativeMultiHeadAttention(embed_dim, num_heads, 10)
        self.assertEqual(layer.w_query.in_features, embed_dim)
        self.assertEqual(layer.w_query.out_features, embed_dim)
        self.assertEqual(layer.w_key.in_features, embed_dim)
        self.assertEqual(layer.w_key.out_features, embed_dim)
        self.assertEqual(layer.w_value.in_features, embed_dim)
        self.assertEqual(layer.w_value.out_features, embed_dim)
        self.assertEqual(layer.w_output.in_features, embed_dim)
        self.assertEqual(layer.w_output.out_features, embed_dim)

    def test_io_projection_shapes_with_explicit_head_embed_dim(self):
        embed_dim = 64
        num_heads = 8
        head_dim = 64
        layer = RelativeMultiHeadAttention(embed_dim, num_heads, 10, head_embed_dim=head_dim)
        self.assertEqual(layer.w_query.in_features, embed_dim)
        self.assertEqual(layer.w_query.out_features, head_dim*num_heads)
        self.assertEqual(layer.w_key.in_features, embed_dim)
        self.assertEqual(layer.w_key.out_features, head_dim*num_heads)
        self.assertEqual(layer.w_value.in_features, embed_dim)
        self.assertEqual(layer.w_value.out_features, head_dim*num_heads)
        self.assertEqual(layer.w_output.in_features, head_dim*num_heads)
        self.assertEqual(layer.w_output.out_features, embed_dim)

    def test_merge_mask(self):
        layer = RelativeMultiHeadAttention(64, 8, 10)
        attention_mask = None
        key_padding_mask = None
        self.assertIsNone(layer.merge_mask(attention_mask, key_padding_mask))

        torch.manual_seed(0)
        attention_mask = torch.randint(0, 2, (2, 10, 10), dtype=torch.bool)
        key_padding_mask = None
        self.assertTrue(torch.all(layer.merge_mask(attention_mask, key_padding_mask) == attention_mask))

        torch.manual_seed(0)
        attention_mask = None
        key_padding_mask = torch.randint(0, 2, (2, 10), dtype=torch.bool)
        self.assertTrue(torch.all(layer.merge_mask(attention_mask, key_padding_mask) == torch.repeat_interleave(key_padding_mask.unsqueeze(-2), key_padding_mask.shape[-1], -2)))

        torch.manual_seed(0)
        attention_mask = torch.randint(0, 2, (2, 10, 10), dtype=torch.bool)
        key_padding_mask = torch.randint(0, 2, (2, 10), dtype=torch.bool)
        self.assertTrue(torch.all(layer.merge_mask(attention_mask, key_padding_mask) == torch.logical_or(attention_mask, torch.repeat_interleave(key_padding_mask.unsqueeze(-2), key_padding_mask.shape[-1], -2))))

    def test_compute_attention_weights(self):
        layer = RelativeMultiHeadAttention(64, 8, 10)

        # Test without masking
        torch.manual_seed(0)
        query = torch.rand(2, 10, 64)
        key = torch.rand(2, 10, 64)
        value = torch.rand(2, 10, 64)
        attention_mask = None
        key_padding_mask = None
        output, attention_weights = layer(query, key, value, key_padding_mask=key_padding_mask, attention_mask=attention_mask, return_attention_weights=True)
        self.assertEqual(output.shape, (2, 10, 64))
        self.assertEqual(attention_weights.shape, (2, 10, 10))

        # Test with attention_mask and key_padding_mask
        torch.manual_seed(0)
        query = torch.rand(2, 10, 64)
        key = torch.rand(2, 10, 64)
        value = torch.rand(2, 10, 64)
        attention_mask = torch.randint(0, 10, (2, 10, 10)) > 7
        key_padding_mask = torch.randint(0, 10, (2, 10,)) > 7
        output, attention_weights = layer(query, key, value, key_padding_mask=key_padding_mask, attention_mask=attention_mask, return_attention_weights=True)
        mask: torch.Tensor = layer.merge_mask(attention_mask, key_padding_mask) # type: ignore
        self.assertEqual(output.shape, (2, 10, 64))
        self.assertEqual(attention_weights.shape, (2, 10, 10))
        self.assertTrue(torch.all(attention_weights[torch.where(mask)] == 0.0))

        # Test with head mask
        torch.manual_seed(0)
        query = torch.rand(2, 10, 64)
        key = torch.rand(2, 10, 64)
        value = torch.rand(2, 10, 64)
        attention_mask = None
        key_padding_mask = None
        attention_head_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        output, attention_weights = layer(query, key, value, key_padding_mask=key_padding_mask, attention_mask=attention_mask, attention_head_mask=attention_head_mask, average_attention_weights=False, return_attention_weights=True)
        self.assertEqual(output.shape, (2, 10, 64))
        self.assertEqual(attention_weights.shape, (2, 8, 10, 10))
        self.assertTrue(torch.all(attention_weights[:,-1,:,:] == 0.0))

    def test_gradients_are_not_none_and_not_nan(self):
        layer = MultiHeadAttention(64, 8)
        torch.manual_seed(0)
        query = torch.rand(2, 10, 64, requires_grad=True)
        key = torch.rand(2, 10, 64, requires_grad=True)
        value = torch.rand(2, 10, 64, requires_grad=True)
        output = layer(query, key, value)
        output.sum().backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertFalse(torch.any(torch.isnan(query.grad)))
        self.assertFalse(torch.any(torch.isnan(key.grad)))
        self.assertFalse(torch.any(torch.isnan(value.grad)))




if __name__ == "__main__":
    unittest.main()
