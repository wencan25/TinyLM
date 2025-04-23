import torch
import torch.nn as nn
import os
import json
from safetensors.torch import save_file


class Qwen2Config:
    def __init__(
        self,
        vocab_size=10000,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=12,
        num_key_value_heads=2,
        intermediate_size=2048,
        attention_dropout=0,
        rms_norm_eps=1e-06,
        padding_idx=0,
        max_position_embeddings=512,
        rope_theta=10000.0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.padding_idx = padding_idx
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_forward(query, key, value, dropout=0.0, num_key_value_groups=1):
    key = repeat_kv(key, num_key_value_groups)
    value = repeat_kv(value, num_key_value_groups)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=dropout,
        is_causal=True,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(self, hidden_states, position_embeddings, attention_mask):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        attn_output = sdpa_attention_forward(
            query_states,
            key_states,
            value_states,
            self.attention_dropout,
            self.num_key_value_groups,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(self, hidden_states, attention_mask, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states
        return outputs


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        base = config.rope_theta
        dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
                1, 2
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def loss_function(logits, labels, vocab_size, ignore_index=-100):
    logits = logits.float()
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = nn.functional.cross_entropy(
        logits, shift_labels, ignore_index=ignore_index, reduction="mean"
    )
    return loss


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super(Qwen2Model, self).__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.padding_idx
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

    def forward(self, input_ids, attention_mask=None):
        # Embedding
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_ids = torch.arange(
            0,
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
        ).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Transformer blocks
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        # Final layer norm
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.model(input_ids, attention_mask)

        # Output layer
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = loss_function(logits, labels, self.config.vocab_size)

        return {"logits": logits, "loss": loss}

    def generate(self, input_ids, max_new_tokens=256):
        next_input_ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(next_input_ids)["logits"]
            next_tokens = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            next_input_ids = torch.cat([next_input_ids, next_tokens], -1)
        result = next_input_ids
        return result

    def save_pretrained(self, save_path, **kwargs):
        os.makedirs(save_path, exist_ok=True)
        state_dict = self.state_dict()
        save_file(state_dict, os.path.join(save_path, "model.safetensors"))
        save_pretrained_config(self.config, save_path)

    def num_parameters(self, only_trainable=True):
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


def save_pretrained_config(config: Qwen2Config, save_path):
    result = {
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": config.attention_dropout,
        "hidden_act": "silu",
        "hidden_size": config.hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "max_window_layers": 28,
        "model_type": "qwen2",
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_scaling": None,
        "rope_theta": config.rope_theta,
        "sliding_window": 4096,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.3",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": config.vocab_size,
    }
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(result, f, indent=4)
    generation_config = {"_from_model_config": False, "transformers_version": "4.51.3"}
    with open(os.path.join(save_path, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=4)
