import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from act import ACTBlock

class MoEACTBlock(nn.Module):
    def __init__(self, block, layers, hiddens, num_experts=4, top_k=2, **act_kwargs):
        super().__init__()
        
        # Ensure top_k is never greater than num_experts
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create multiple expert ACT blocks
        self.experts = nn.ModuleList([
            ACTBlock(copy.deepcopy(block), layers, hiddens, **act_kwargs)
            for _ in range(num_experts)
        ])
        
        # Router network
        self.router = nn.Linear(hiddens, num_experts)
        
    def forward(self, hidden_states, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Calculate routing probabilities
        router_logits = self.router(hidden_states)  # [batch, seq_len, num_experts]
        
        # Get sparse dispatch weights with load balancing
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts per token
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, k=self.top_k, dim=-1
        )
        
        # Normalize the weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        combined_output = torch.zeros_like(hidden_states)
        act_loss = 0.0
        ponder_cost = 0.0
        
        # Process tokens through their assigned experts
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if not expert_mask.any():
                continue
                
            # Get the weights for this expert
            expert_weights = torch.zeros(
                (batch_size, seq_len, 1), device=hidden_states.device, dtype=top_k_weights.dtype
            )
            
            for k in range(self.top_k):
                mask = (top_k_indices[..., k] == expert_idx)
                weight_indices = torch.nonzero(mask)
                if weight_indices.size(0) > 0:  # Check if there are any tokens routed to this expert
                    batch_indices = weight_indices[:, 0]
                    seq_indices = weight_indices[:, 1]
                    expert_weights[batch_indices, seq_indices] = top_k_weights[batch_indices, seq_indices, k].unsqueeze(-1)
            
            # Only process tokens assigned to this expert
            if expert_mask.any():
                # Process through the ACT block
                expert_output = self.experts[expert_idx](
                    hidden_states,
                    **kwargs
                )
                
                # Extract outputs and loss components
                if isinstance(expert_output, tuple):
                    expert_hidden, expert_act_loss, expert_ponder = expert_output
                    act_loss = act_loss + expert_act_loss * expert_weights.mean()
                    ponder_cost = ponder_cost + expert_ponder * expert_weights.mean()
                else:
                    expert_hidden = expert_output
                
                # Weight and combine the expert outputs - ensure tensors have compatible shapes
                if isinstance(expert_hidden, torch.Tensor):
                    combined_output = combined_output + expert_hidden * expert_weights
                else:
                    # Handle the case where expert_hidden might be a list or tuple
                    if isinstance(expert_hidden, (list, tuple)) and len(expert_hidden) > 0 and isinstance(expert_hidden[0], torch.Tensor):
                        # Use the first tensor if it's a list/tuple of tensors
                        combined_output = combined_output + expert_hidden[0] * expert_weights
        
        # Return combined output and losses
        return combined_output, act_loss, ponder_cost


class MoELayer(nn.Module):
    def __init__(self, experts: nn.ModuleList, router: nn.Module, top_k: int = 2):
        super().__init__()
        self.experts = experts
        self.router = router
        self.num_experts = len(experts)
        self.top_k = min(top_k, self.num_experts)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get routing logits from the router
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]

        # Get routing weights (probabilities) by applying softmax
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts and their weights
        top_k_weights, top_k_indices = torch.topk(routing_weights, k=self.top_k, dim=-1)

        # Normalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Initialize final output tensor
        final_output = torch.zeros_like(hidden_states)

        # Iterate over each expert
        for i in range(self.num_experts):
            # Create a mask for tokens routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1) # True for tokens that have this expert in their top_k

            if not expert_mask.any():
                continue

            # Create a mask for selecting the weights for the current expert
            # This mask will be [batch_size, seq_len, top_k]
            idx_mask = top_k_indices == i
            
            # Gather the weights for the current expert.
            # Weights are [batch_size, seq_len, top_k]. We want to sum weights if a token is routed to expert 'i' multiple times (though top_k usually < num_experts, so less likely unless k=num_experts)
            # For simplicity, we assume each expert is chosen at most once in top_k for a given token, or if chosen multiple times, its weight appears multiple times in top_k_weights.
            # The current approach correctly sums weights if an expert is listed multiple times in top_k_indices for a token.
            current_expert_weights = (top_k_weights * idx_mask).sum(dim=-1, keepdim=True) # [batch_size, seq_len, 1]

            # Select tokens for the current expert
            # hidden_states_expert = hidden_states[expert_mask] # This would flatten the batch_size and seq_len dimensions
            
            # Instead, we want to pass all hidden_states and let the expert process them.
            # The output will be weighted, so only tokens routed to this expert will contribute meaningfully.
            # This is a simplification. A more optimized approach would be to select and then process.
            # However, for generality of expert (e.g. if expert expects full sequence), this is safer.
            
            # Process with the expert
            expert_output = self.experts[i](hidden_states, *args, **kwargs) # Pass through any additional args/kwargs

            # Ensure expert_output is a tensor
            if isinstance(expert_output, tuple): # If expert returns multiple values (e.g. output, other_losses)
                expert_output_tensor = expert_output[0]
            else:
                expert_output_tensor = expert_output
            
            # Weight the expert output
            weighted_output = expert_output_tensor * current_expert_weights
            
            # Add to the final output
            final_output += weighted_output
            
        return final_output