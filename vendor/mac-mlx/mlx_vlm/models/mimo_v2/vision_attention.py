"""MiMo ViT attention with the learned null sink used by its serving backend.

The published Transformers reference adds the sink bias to the first real key.
The checkpoint instead uses an extra softmax logit with a zero value (SGLang
VisionAttention's s_aux). Keep that distinction on MPS as well as CUDA/CPU.
"""


def attention_with_sink(q, k, v, *, sink=None, window_size=-1, scale=None):
    import torch
    from torch.nn import functional as F

    mask = None
    if window_size > 0:
        row = torch.arange(q.shape[-2], device=q.device)[:, None]
        col = torch.arange(k.shape[-2], device=q.device)[None, :]
        mask = torch.zeros(q.shape[-2], k.shape[-2], device=q.device, dtype=q.dtype)
        mask.masked_fill_((row - col).abs() > window_size, float('-inf'))
    if sink is not None:
        # The extra key contributes only its learned logit to normalization.
        # Its value is zero, so it cannot bias a real image patch's content.
        k = torch.cat((k, torch.zeros_like(k[..., :1, :])), dim=-2)
        v = torch.cat((v, torch.zeros_like(v[..., :1, :])), dim=-2)
        shape = (q.shape[0], q.shape[1], q.shape[-2], k.shape[-2] - 1)
        mask = torch.zeros(shape, dtype=q.dtype, device=q.device) if mask is None else mask.expand(shape)
        bias = sink.to(q.dtype).view(1, -1, 1, 1).expand(*shape[:-1], 1)
        mask = torch.cat((mask, bias), dim=-1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def vision_attention_forward(self, hidden_states, cu_seqlens, position_embeddings, full_attn=False):
    import torch

    length = hidden_states.shape[0]
    qkv = self.qkv(hidden_states)
    qdim, kvdim = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
    q, k, v = qkv.split((qdim, kvdim, kvdim), dim=-1)
    q = q.reshape(length, self.num_heads, self.head_dim)
    k = k.reshape(length, self.num_kv_heads, self.head_dim)
    v = v.reshape(length, self.num_kv_heads, self.head_dim)
    cos, sin = (x.float().unsqueeze(1) for x in position_embeddings)
    def rotary(x):
        dtype = x.dtype
        x = x.float()
        left, right = x.chunk(2, dim=-1)
        return (x * cos + torch.cat((-right, left), dim=-1) * sin).to(dtype)
    q, k = rotary(q), rotary(k)
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    outputs = []
    for qc, kc, vc in zip(q.split(lengths), k.split(lengths), v.split(lengths)):
        qc, kc, vc = (x.transpose(0, 1).unsqueeze(0) for x in (qc, kc, vc))
        if self.num_kv_groups > 1:
            kc = kc.repeat_interleave(self.num_kv_groups, dim=1)
            vc = vc.repeat_interleave(self.num_kv_groups, dim=1)
        sink = self.sinks if not full_attn else None
        output = attention_with_sink(qc, kc, vc, sink=sink,
            window_size=self.window_size if sink is not None else -1, scale=self.scaling)
        outputs.append(output.squeeze(0).transpose(0, 1))
    return self.proj(torch.cat(outputs, dim=0).reshape(length, -1))


def install_vision_attention(visual):
    from types import MethodType

    for block in visual.blocks:
        block.attn.forward = MethodType(vision_attention_forward, block.attn)
