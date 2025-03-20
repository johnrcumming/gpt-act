import torch
import torch.nn as nn
import copy

from act import ACTBlock

class MoEACTBlock(nn.Module):
    def __init__(self, block, layers, hiddens, num_experts=4, top_k=2, **act_kwargs):
        super().__init__()
        
        # Create multiple expert ACT blocks
        self.experts = nn.ModuleList([
            ACTBlock(copy.deepcopy(block), layers, hiddens, **act_kwargs)
            for _ in range(num_experts)
        ])
        
        # Router network
        self.router = nn.Linear(hiddens, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts
        
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
                (batch_size, seq_len, 1), device=hidden_states.device
            )
            
            for k in range(self.top_k):
                mask = (top_k_indices[..., k] == expert_idx)
                expert_weights[mask] = top_k_weights[mask, k].unsqueeze(-1)
            
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
                
                # Weight and combine the expert outputs
                combined_output = combined_output + expert_hidden * expert_weights
        
        # Return combined output and losses
        return combined_output, act_loss, ponder_cost