
import torch.nn as nn
import torch


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # Query Key and Value weight matrices
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization

        # Causal mask to ensure that attention is only applied to previous tokens in the sequence
        self.register_buffer('mask', torch.triu(
            torch.ones(context_length, context_length), diagonal=1))
        '''
        diagonal = 0 → keep the main diagonal and everything above it.
        diagonal = 1 → keep everything strictly above the main diagonal (excludes the main diagonal).
        diagonal < 0 → shifts downward (includes some elements below the main diagonal).
        '''

    def forward(self, x):
        b, num_tokens, d_in = x.size()

        # Compute queries, keys, and values matrices
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Compute scaled dot-product attention
        # Change for multiplication
        attn_scores = queries @ keys.transpose(1, 2)
        # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
