import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming MoELayer is in a module named 'moe' at the same level
from moe import MoELayer 

# Define a simple FFN expert for testing purposes
class SimpleFFNExpert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, *args, **kwargs): # Add *args, **kwargs for compatibility
        return self.fc2(self.relu(self.fc1(x)))

# Define a mock expert that can be configured (e.g., identity or scaling)
class MockExpert(nn.Module):
    def __init__(self, scale_factor=1.0, add_factor=0.0, output_transform=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.add_factor = add_factor
        self.output_transform = output_transform # A function to apply to output

    def forward(self, x, *args, **kwargs): # Add *args, **kwargs for compatibility
        x = x * self.scale_factor + self.add_factor
        if self.output_transform:
            return self.output_transform(x)
        return x

class TestMoELayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 10
        self.hidden_size = 32
        self.num_experts = 4
        self.top_k_default = 2

        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.hidden_size)

    def test_initialization(self):
        experts = nn.ModuleList([MockExpert() for _ in range(self.num_experts)])
        router = nn.Linear(self.hidden_size, self.num_experts)
        moe_layer = MoELayer(experts, router, top_k=self.top_k_default)
        
        self.assertEqual(len(moe_layer.experts), self.num_experts)
        self.assertEqual(moe_layer.top_k, self.top_k_default)
        self.assertIsInstance(moe_layer.router, nn.Linear)

    def test_forward_pass_shape(self):
        experts = nn.ModuleList([SimpleFFNExpert(self.hidden_size, self.hidden_size * 2, self.hidden_size) for _ in range(self.num_experts)])
        router = nn.Linear(self.hidden_size, self.num_experts)
        moe_layer = MoELayer(experts, router, top_k=self.top_k_default)
        
        output = moe_layer(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)

    def test_forward_pass_identity_experts_top1(self):
        """Test with identity experts and top_k=1, ensuring output matches input if one expert is chosen."""
        identity_experts = nn.ModuleList([MockExpert(scale_factor=1.0) for _ in range(self.num_experts)])
        
        # Design a router that strongly selects the first expert for all tokens
        router_mock = nn.Linear(self.hidden_size, self.num_experts)
        torch.nn.init.zeros_(router_mock.weight)
        torch.nn.init.zeros_(router_mock.bias)
        router_mock.bias.data[0] = 100.0 # Strong bias for expert 0
        
        moe_layer = MoELayer(identity_experts, router_mock, top_k=1)
        output = moe_layer(self.input_tensor)
        
        # Output should be very close to input because only expert 0 (identity) is chosen with weight ~1.0
        self.assertTrue(torch.allclose(output, self.input_tensor, atol=1e-6))

    def test_forward_pass_identity_experts_top_k_sum_to_one(self):
        """Test with identity experts and top_k > 1, output should still be input if all chosen experts are identity."""
        identity_experts = nn.ModuleList([MockExpert(scale_factor=1.0) for _ in range(self.num_experts)])
        
        # Router that distributes weights somewhat, but all chosen experts are identity
        router = nn.Linear(self.hidden_size, self.num_experts) 
        moe_layer = MoELayer(identity_experts, router, top_k=self.top_k_default)
        output = moe_layer(self.input_tensor)
        
        # Since all experts are identity and top_k weights sum to 1, output should be input
        self.assertTrue(torch.allclose(output, self.input_tensor, atol=1e-6))


    def test_output_combination_scaling_experts_top1(self):
        """Test with scaling experts and top_k=1."""
        scaling_experts = nn.ModuleList([MockExpert(scale_factor=float(i + 1)) for i in range(self.num_experts)])
        
        # Router that strongly selects expert 1 (scale_factor=2)
        router_mock = nn.Linear(self.hidden_size, self.num_experts)
        torch.nn.init.zeros_(router_mock.weight)
        torch.nn.init.zeros_(router_mock.bias)
        router_mock.bias.data[1] = 100.0 
        
        moe_layer = MoELayer(scaling_experts, router_mock, top_k=1)
        output = moe_layer(self.input_tensor)
        
        expected_output = self.input_tensor * 2.0 # Expert 1 scales by 2.0
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_top_k_routing_selection(self):
        """Verify that only top_k experts contribute for each token."""
        num_experts = 3
        top_k = 2
        input_val = 1.0
        # Experts add a unique value: expert_i adds 10**(i)
        # Expert 0 adds 1, Expert 1 adds 10, Expert 2 adds 100
        experts = nn.ModuleList([
            MockExpert(scale_factor=0.0, add_factor=1.0),  # Adds 1
            MockExpert(scale_factor=0.0, add_factor=10.0), # Adds 10
            MockExpert(scale_factor=0.0, add_factor=100.0) # Adds 100
        ])

        # Create a single token input: (1, 1, hidden_size)
        single_token_input = torch.full((1, 1, self.hidden_size), input_val)

        # Router logits: [batch_size, seq_len, num_experts]
        # For our single token, make logits that select expert 0 and expert 2 as top 2
        router_logits_val = torch.tensor([[[10.0, 0.0, 20.0]]]) # Expert 2, then Expert 0

        # Mock the router's forward pass
        class MockRouter(nn.Module):
            def __init__(self, logits_to_return):
                super().__init__()
                self.logits_to_return = logits_to_return
            def forward(self, x):
                # Repeat logits for batch and sequence if necessary, but here input is (1,1,H)
                return self.logits_to_return.to(x.device)

        router = MockRouter(router_logits_val)
        moe_layer = MoELayer(experts, router, top_k=top_k)
        output = moe_layer(single_token_input)

        # Router softmax will give high weights to expert 0 and 2.
        # Let w0, w1, w2 be softmax weights. w1 will be near zero.
        # Output = w0 * (input_val*0 + 1) + w1 * (input_val*0 + 10) + w2 * (input_val*0 + 100)
        # Since top_k=2, one weight (for expert 1) in the top_k_weights calculation will be zeroed out effectively.
        # The top_k_weights are normalized.
        # Softmax of [10, 0, 20] is roughly [e^10, e^0, e^20] / sum(...)
        # e^10 ~= 22026, e^0 = 1, e^20 ~= 4.85e8
        # So, weights are approx [small, negligible, large]
        # Top 2 indices are 2 and 0.
        # Top 2 weights (normalized) would be w_expert2_norm, w_expert0_norm
        
        # Expected output will be a combination of expert 0's output (input_val + 1.0) and expert 2's output (input_val + 100.0)
        # (after scaling by 0, so just 1.0 and 100.0)
        # Weighted sum: w_norm_0 * 1.0 + w_norm_2 * 100.0
        
        # Calculate expected normalized weights for expert 0 and 2
        probs = F.softmax(router_logits_val.squeeze(0).squeeze(0), dim=-1) # Shape [num_experts]
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        normalized_top_k_probs = top_k_probs / top_k_probs.sum()

        expected_output_val = 0.0
        if 0 in top_k_indices:
            idx_in_topk = (top_k_indices == 0).nonzero(as_tuple=True)[0]
            expected_output_val += normalized_top_k_probs[idx_in_topk].item() * experts[0](single_token_input).mean().item()
        if 1 in top_k_indices: # Should not happen with logits [10,0,20] and top_k=2
             pass 
        if 2 in top_k_indices:
            idx_in_topk = (top_k_indices == 2).nonzero(as_tuple=True)[0]
            expected_output_val += normalized_top_k_probs[idx_in_topk].item() * experts[2](single_token_input).mean().item()
        
        # The output tensor elements should all be equal to expected_output_val
        # Ensure expected_output_val is a float for torch.full_like
        self.assertTrue(torch.allclose(output, torch.full_like(output, float(expected_output_val)), atol=1e-3))

    def test_initialization_top_k_clamping(self):
        """Test that top_k is clamped to num_experts."""
        num_experts = 3
        experts = nn.ModuleList([MockExpert() for _ in range(num_experts)])
        router = nn.Linear(self.hidden_size, num_experts)
        
        # Test when top_k > num_experts
        moe_layer_clamped = MoELayer(experts, router, top_k=num_experts + 2)
        self.assertEqual(moe_layer_clamped.top_k, num_experts)

        # Test when top_k < num_experts
        moe_layer_normal = MoELayer(experts, router, top_k=num_experts -1)
        self.assertEqual(moe_layer_normal.top_k, num_experts -1)
        
        # Test when top_k == num_experts
        moe_layer_equal = MoELayer(experts, router, top_k=num_experts)
        self.assertEqual(moe_layer_equal.top_k, num_experts)

    def test_forward_with_actual_ffn_expert_from_gpt2_act(self):
        """Test MoELayer with _FeedForwardExpert from gpt2_act.py."""
        try:
            from gpt2_act import _FeedForwardExpert as GPT2FeedForwardExpert
        except ImportError:
            self.skipTest("_FeedForwardExpert could not be imported from gpt2_act.py. Skipping this test.")
            return

        experts = nn.ModuleList([
            GPT2FeedForwardExpert(self.hidden_size, self.hidden_size * 2, self.hidden_size) 
            for _ in range(self.num_experts)
        ])
        router = nn.Linear(self.hidden_size, self.num_experts)
        moe_layer = MoELayer(experts, router, top_k=self.top_k_default)
        
        output = moe_layer(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)
        # Basic check that output is not NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
