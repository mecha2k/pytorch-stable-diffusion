import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from tensorflow.python.ops.numpy_ops.np_array_ops import cumsum

from sd import encoder, diffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"{device} is available in torch and cuda version : {torch.version.cuda}")


def self_attention(x, n_heads, in_proj_bias=True, out_proj_bias=True, causal_mask=False):
    # x: # (Batch_Size, Seq_Len, Dim)
    # (Batch_Size, Seq_Len, Dim)
    input_shape = x.shape

    # (Batch_Size, Seq_Len, Dim)
    batch_size, sequence_length, d_embed = input_shape

    # This combines the Wq, Wk and Wv matrices into one matrix
    in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
    # This one represents the Wo matrix
    out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
    d_head = d_embed // n_heads

    # (Batch_Size, Seq_Len, H, Dim / H)
    interim_shape = (batch_size, sequence_length, n_heads, d_head)

    # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
    q, k, v = in_proj(x).chunk(3, dim=-1)

    # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
    q = q.view(interim_shape).transpose(1, 2)
    k = k.view(interim_shape).transpose(1, 2)
    v = v.view(interim_shape).transpose(1, 2)

    # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
    weight = q @ k.transpose(-1, -2)

    if causal_mask:
        # Mask where the upper triangle (above the principal diagonal) is 1
        mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
        # Fill the upper triangle with -inf
        weight.masked_fill_(mask, -torch.inf)

        # Divide by d_k (Dim / H).
    # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
    weight /= math.sqrt(d_head)

    # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
    weight = F.softmax(weight, dim=-1)

    # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
    output = weight @ v

    # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
    output = output.transpose(1, 2)

    # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
    output = output.reshape(input_shape)

    # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
    output = out_proj(output)

    # (Batch_Size, Seq_Len, Dim)
    return output


if __name__ == "__main__":
    inputs = torch.randn(64, 3, 32, 32).to(device)
    ns = inputs.shape
    noise_shape = (ns[0], 4, ns[2] / 8, ns[3] / 8)
    noise = torch.randn(1).to(device)
    model = encoder.VAE_Encoder().to(device)
    outputs = model(inputs, noise)
    print(outputs.shape)
    print("=" * 100)

    inputs = torch.randn(64, 128, 512)  # (Batch_Size, Seq_Len, Dim)
    outputs = self_attention(inputs, 4)
    print(outputs.shape)
    print("=" * 100)

    # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
    weight = torch.randn(4, 4)
    # Mask where the upper triangle (above the principal diagonal) is 1
    mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
    # Fill the upper triangle with -inf
    weight.masked_fill_(mask, -torch.inf)
    print(mask)
    print(weight)

    out_lin = nn.Linear(512, 512)
    outputs = out_lin(inputs)
    print(out_lin.weight.shape)
    print(outputs.shape)
    print("=" * 100)

    time = torch.ones(1, 320)
    time_embedding = diffusion.TimeEmbedding(320)
    outputs = time_embedding(time)
    print(outputs.shape)
    print("=" * 100)

    beta_start = 0.00085
    beta_end = 0.0120
    num_training_steps = 1000

    betas = (
        torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32) ** 2
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    one = torch.tensor(1.0)

    betas = np.arange(1, 5, step=1)
    print(betas)
    betas = torch.from_numpy(betas)
    cumsum = torch.cumsum(betas, dim=0)
    cumprod = torch.cumprod(betas, dim=0)
    print(cumsum, cumprod)
