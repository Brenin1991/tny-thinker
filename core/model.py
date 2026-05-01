import torch
import torch.nn as nn

class GPTConfig:
    def __init__(self, vocab_size, d_model=256, n_heads=4,
                 n_layers=4, d_ff=1024, max_seq_len=128, dropout=0.1):
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.n_layers    = n_layers
        self.d_ff        = d_ff
        self.max_seq_len = max_seq_len
        self.dropout     = dropout

class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)
        decoder_layer  = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=config.n_layers)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        device = idx.device
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(positions)
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        x = self.ln_f(x)
        return self.head(x)