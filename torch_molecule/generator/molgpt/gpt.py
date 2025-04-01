"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num = int(bool(config.num_task)) + int(config.scaffold_maxlen)   #int(config.lstm_layers)    #  int(config.use_scaffold) 
        # num = 1
        self.register_buffer("mask", torch.tril(torch.ones(config.max_len + num, config.max_len + num))
                                     .view(1, 1, config.max_len + num, config.max_len + num))

        self.n_head = config.num_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, max_len, **kwargs):
        self.vocab_size = vocab_size
        self.max_len = max_len
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT(nn.Module):
    """  the full GPT language model, with a context size of max_len """

    def __init__(self, vocab_size, max_len, num_task, num_layer, num_head, hidden_size, use_scaffold, scaffold_maxlen, use_lstm, lstm_layers):
        super().__init__()
        config = GPTConfig(vocab_size, max_len, num_task = num_task, num_layer = num_layer, num_head = num_head, hidden_size = hidden_size, use_scaffold = use_scaffold, scaffold_maxlen = scaffold_maxlen, use_lstm = use_lstm, lstm_layers = lstm_layers)

        # input embedding stem
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.type_emb = nn.Embedding(2, config.hidden_size)
        if config.num_task:
            self.prop_nn = nn.Linear(config.num_task, config.hidden_size)
     
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_len, config.hidden_size))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.max_len = config.max_len

        if config.use_lstm:
            self.lstm = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size, num_layers = config.lstm_layers, dropout = 0.3, bidirectional = False)
    
    def get_max_len(self):
        return self.max_len

    def initialize_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config["weight_decay"]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config["learning_rate"], betas=train_config["betas"])
        return optimizer

    def compute_loss(self, x, targets, prop = None, scaffold = None):
        logits, attn_maps = self.forward(x, targets, prop, scaffold)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))
        return loss

    def forward(self, idx, targets=None, prop = None, scaffold = None):
        b, t = idx.size()
        assert t <= self.max_len, "Cannot forward, model max_len is exhausted."

        if self.config.num_task:
            assert prop.size(-1) == self.config.num_task, "num_task should be equal to last dim of property vector"           

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b,t), dtype = torch.long, device = idx.device))
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)

        if self.config.num_task:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))
            if prop.ndim == 2:
                p = self.prop_nn(prop.unsqueeze(1))    # for single property
            else:
                p = self.prop_nn(prop)    # for multiproperty
            p += type_embd
            x = torch.cat([p, x], 1)

        if self.config.use_scaffold:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))

            scaffold_embeds = self.tok_emb(scaffold)     # .mean(1, keepdim = True)
            if self.config.use_lstm:
                scaffold_embeds = self.lstm(scaffold_embeds.permute(1,0,2))[1][0]
                # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], 2, self.config.hidden_size).mean(2)
                scaffold_embeds = scaffold_embeds.permute(1,0,2)   # mean(0, keepdim = True)
                # scaffold_embeds = scaffold_embeds.reshape(self.config.lstm_layers, 1, -1, self.config.hidden_size)[-1].permute(1,0,2)
                # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], self.config.hidden_size)
            scaffold_embeds += type_embd
            x = torch.cat([scaffold_embeds, x], 1)

        # x = self.blocks(x)
        attn_maps = []

        for layer in self.blocks:
            x, attn = layer(x)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        # print(logits.shape)
        if self.config.num_task and self.config.use_scaffold:
            num = int(bool(self.config.num_task)) + int(self.config.scaffold_maxlen)
        elif self.config.num_task:
            num = int(bool(self.config.num_task))
        elif self.config.use_scaffold:
            num = int(self.config.scaffold_maxlen) 
        else:
            num = 0

        logits = logits[:, num:, :]

        return logits, attn_maps # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)