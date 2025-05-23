import unittest
import torch
import torch.nn as nn
import tempfile
import os

from gpt2_act import GPT2ACTConfig, GPT2ACTLMHeadModel, _FeedForwardExpert
from moe import MoELayer # For type checking

class TestGPT2ACTMoELayerIntegration(unittest.TestCase):

    def _create_config_and_model(self, moe_layer_type="MoELayer", expert_type="ffn"):
        vocab_size = 50  # Small vocab for testing
        n_embd = 16      # Small embedding size
        n_positions = 30 # Small context size
        n_layer = 2      # Minimal layers
        n_head = 2       # Minimal heads
        num_experts = 2
        top_k = 1
        
        # Expert config for MoELayer (FFN)
        # These will be defaulted by GPT2ACTConfig if not set, but explicit for clarity
        expert_hidden_size = n_embd 
        expert_intermediate_size = n_embd * 2 # Smaller intermediate for tests

        config = GPT2ACTConfig(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_positions=n_positions,
            n_layer=n_layer,
            n_head=n_head,
            # MoE specific
            moe_layer_type=moe_layer_type,
            expert_type=expert_type,
            num_experts=num_experts,
            experts_top_k=top_k,
            expert_hidden_size=expert_hidden_size if moe_layer_type == "MoELayer" else None,
            expert_intermediate_size=expert_intermediate_size if moe_layer_type == "MoELayer" else None,
            # ACT specific (less relevant if MoELayer doesn't produce these losses)
            act_commitment_cost=0.001,
            # Other params to avoid warnings or issues
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0, # Assuming pad_token_id is 0 for test model
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            # Set low activation function related params if any (not directly in GPT2Config)
        )
        model = GPT2ACTLMHeadModel(config)
        model.eval() # Default to eval mode, switch to train if needed
        return config, model

    def test_model_initialization_with_moelayer(self):
        config, model = self._create_config_and_model(moe_layer_type="MoELayer", expert_type="ffn")
        
        self.assertIsInstance(model.transformer.act_f, MoELayer, "act_f should be MoELayer")
        self.assertEqual(model.transformer.moe_layer_type, "MoELayer")
        
        # Check experts type if MoELayer and FFN
        if isinstance(model.transformer.act_f, MoELayer) and config.expert_type == "ffn":
            self.assertTrue(len(model.transformer.act_f.experts) > 0)
            for expert in model.transformer.act_f.experts:
                self.assertIsInstance(expert, _FeedForwardExpert, "Experts should be _FeedForwardExpert")

    def test_forward_pass_training_moelayer(self):
        config, model = self._create_config_and_model(moe_layer_type="MoELayer")
        model.train() # Switch to train mode for loss calculation
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone() # Standard LM objective

        outputs = model(input_ids=input_ids, labels=labels)

        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.ndim, 0) # Scalar tensor
        self.assertTrue(outputs.loss.requires_grad)

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape, (batch_size, seq_len, config.vocab_size))

        # For MoELayer, act_loss and ponder_cost should be present and zero
        self.assertIsNotNone(outputs.act_loss)
        self.assertEqual(outputs.act_loss.item(), 0.0)
        self.assertIsNotNone(outputs.ponder_cost)
        self.assertEqual(outputs.ponder_cost.item(), 0.0)

    def test_backward_pass_moelayer(self):
        config, model = self._create_config_and_model(moe_layer_type="MoELayer")
        model.train()

        # Ensure parameters require gradients
        for param in model.parameters():
            param.requires_grad = True
            
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Zero gradients before backward pass
        model.zero_grad()
        loss.backward()

        # Check some gradients
        self.assertIsNotNone(model.transformer.act_f.router.weight.grad)
        self.assertTrue(torch.sum(torch.abs(model.transformer.act_f.router.weight.grad)).item() > 0)
        
        first_expert = model.transformer.act_f.experts[0]
        if isinstance(first_expert, _FeedForwardExpert):
            self.assertIsNotNone(first_expert.fc1.weight.grad)
            self.assertTrue(torch.sum(torch.abs(first_expert.fc1.weight.grad)).item() > 0)
        
        # Check a non-MoE layer param too
        self.assertIsNotNone(model.transformer.wte.weight.grad)


    def test_text_generation_moelayer(self):
        config, model = self._create_config_and_model(moe_layer_type="MoELayer")
        model.eval()

        # Input IDs should be within the model's vocab_size
        input_ids = torch.tensor([[config.vocab_size // 2, config.vocab_size // 3, 1, 0]], dtype=torch.long)
        max_length = 15
        
        # Ensure model is on CPU if no CUDA available, or tests might fail if model is on GPU but input isn't
        # Forcing CPU for this test for simplicity as CUDA is not guaranteed in test env.
        model.to("cpu")
        input_ids = input_ids.to("cpu")

        with torch.no_grad(): # Generation doesn't require grads
            output_sequences = model.generate(
                input_ids, 
                max_length=max_length,
                pad_token_id=config.pad_token_id # Ensure pad_token_id is set
            )

        self.assertIsInstance(output_sequences, torch.Tensor)
        # Expected shape: (batch_size, max_length)
        self.assertEqual(output_sequences.shape[0], input_ids.shape[0])
        self.assertTrue(output_sequences.shape[1] <= max_length) 
        
        # Check if all generated token IDs are valid (less than vocab_size)
        self.assertTrue(torch.all(output_sequences < config.vocab_size))
        self.assertTrue(torch.all(output_sequences >= 0))

    def test_model_save_load_moelayer(self):
        config, model_original = self._create_config_and_model(moe_layer_type="MoELayer")
        model_original.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_original.save_pretrained(tmpdirname)
            
            # Ensure the config reflects the MoELayer choice
            loaded_config = GPT2ACTConfig.from_pretrained(tmpdirname)
            self.assertEqual(loaded_config.moe_layer_type, "MoELayer")

            loaded_model = GPT2ACTLMHeadModel.from_pretrained(tmpdirname)
            loaded_model.eval()
        
        self.assertIsInstance(loaded_model.transformer.act_f, MoELayer)
        self.assertEqual(loaded_model.config.moe_layer_type, "MoELayer")
        self.assertEqual(loaded_model.config.num_experts, config.num_experts)

        # Perform a forward pass to ensure it works
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        with torch.no_grad():
            outputs_original = model_original(input_ids)
            outputs_loaded = loaded_model(input_ids)
        
        self.assertTrue(torch.allclose(outputs_original.logits, outputs_loaded.logits, atol=1e-6))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
