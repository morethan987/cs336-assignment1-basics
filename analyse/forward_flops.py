def compute_transformer_forward_flops(
    batch_size: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
):
    """
    Transformer Forward Pass FLOPs
    """
    d_ff = int(8 / 3 * d_model)
    tokens = batch_size * context_length

    breakdown = {}

    # ==========================================================
    # 1. Single Transformer Block
    # ==========================================================

    # --- (1) Pre-Attn RMSNorm ---
    flops_rmsnorm_attn = 4 * tokens * d_model

    # --- (2) Multi-Head Self-Attention (MHSA) ---
    flops_qkv_proj = 3 * (2 * tokens * d_model * d_model)  # 6 * tokens * d_model^2

    # Q @ K^T: [B, H, S, d_k] x [B, H, d_k, S] -> [B, H, S, S]
    # d_k = d_model / num_heads
    flops_attn_score = 2 * batch_size * num_heads * (context_length**2) * (d_model // num_heads)
    flops_attn_softmax = 4 * batch_size * num_heads * (context_length**2)

    # (P @ V): [B, H, S, S] x [B, H, S, d_k] -> [B, H, S, d_k]
    flops_attn_context = 2 * batch_size * num_heads * (context_length**2) * (d_model // num_heads)

    # W_O: [B*S, D] x [D, D]
    flops_out_proj = 2 * tokens * d_model * d_model

    # residual add
    flops_attn_residual = tokens * d_model

    flops_mhsa_layer = (
        flops_qkv_proj
        + flops_attn_score
        + flops_attn_softmax
        + flops_attn_context
        + flops_out_proj
        + flops_attn_residual
    )

    # --- (3) Pre-FFN RMSNorm ---
    flops_rmsnorm_ffn = 4 * tokens * d_model

    # --- (4) SwiGLU FFN ---
    # Gate and Up project: 2 * [B*S, D] x [D, d_ff]
    flops_gate_up_proj = 2 * (2 * tokens * d_model * d_ff)  # 4 * tokens * d_model * (8/3 * d_model)

    # SwiGLU: Swish(Gate) * Up = (Gate * Sigmoid(Gate)) * Up
    flops_swiglu_act = 4 * tokens * d_ff

    # Down project: [B*S, d_ff] x [d_ff, D]
    flops_down_proj = 2 * tokens * d_ff * d_model

    # residual add
    flops_ffn_residual = tokens * d_model

    flops_swiglu_layer = flops_gate_up_proj + flops_swiglu_act + flops_down_proj + flops_ffn_residual

    flops_per_layer = flops_rmsnorm_attn + flops_mhsa_layer + flops_rmsnorm_ffn + flops_swiglu_layer
    flops_all_layers = num_layers * flops_per_layer

    # ==========================================================
    # 2. Final Norm, LM Head, Cross Entropy
    # ==========================================================
    # Final RMSNorm
    flops_final_rmsnorm = 4 * tokens * d_model

    # Output Embedding / LM Head 投影: [B*S, D] x [D, vocab_size]
    flops_lm_head = 2 * tokens * d_model * vocab_size

    # Cross-Entropy Loss: Softmax + NLL Loss (约 3 FLOPs / vocab_element)
    flops_cross_entropy = 3 * tokens * vocab_size

    # ==========================================================
    # 3. Total
    # ==========================================================
    total_forward_flops = flops_all_layers + flops_final_rmsnorm + flops_lm_head + flops_cross_entropy

    breakdown["Attention (QKV & Out Proj)"] = num_layers * (flops_qkv_proj + flops_out_proj)
    breakdown["Attention (QK^T & PV Score)"] = num_layers * (flops_attn_score + flops_attn_context)
    breakdown["SwiGLU Projections (Gate, Up, Down)"] = num_layers * (flops_gate_up_proj + flops_down_proj)
    breakdown["LM Head Projection"] = flops_lm_head
    breakdown["RMSNorms (All)"] = num_layers * (flops_rmsnorm_attn + flops_rmsnorm_ffn) + flops_final_rmsnorm
    breakdown["Activations & Softmax & Residuals"] = (
        num_layers * (flops_attn_softmax + flops_attn_residual + flops_swiglu_act + flops_ffn_residual)
        + flops_cross_entropy
    )

    return {
        "tokens": tokens,
        "flops_per_layer": flops_per_layer,
        "total_forward_flops": total_forward_flops,
        "breakdown": breakdown,
    }


def format_flops(flops_val):
    tflops = flops_val / 1e12
    pflops = flops_val / 1e15
    return f"{flops_val:>18,} FLOPs | {tflops:>10.3f} TFLOPs | {pflops:>8.4f} PFLOPs"


if __name__ == "__main__":
    config = {
        "batch_size": 1024,
        "vocab_size": 50257,
        "context_length": 1024,
        "num_layers": 48,
        "d_model": 1600,
        "num_heads": 25,
    }

    print("=" * 80)
    print("Model hyperparams:")
    for k, v in config.items():
        print(f"  - {k:<18}: {v}")
    print("=" * 80)

    res = compute_transformer_forward_flops(**config)
    total_f = res["total_forward_flops"]

    print(f"Single forward Token count: {res['tokens']:,} (Batch Size x Context Length)")
    print("-" * 80)
    print("Forward FLOPs details:")
    for name, f_val in res["breakdown"].items():
        ratio = (f_val / total_f) * 100
        print(f"  - {name:<36}: {format_flops(f_val)} ({ratio:>5.2f} %)")

    print("-" * 80)
    print(f"  Forward Total: {format_flops(total_f)}")
    print("=" * 80)
    print("Explaination:")
    print("  1. Backward usually costs 2 times FLOPs as Forward (Backward = 2 * Forward)")
    print(f"  2. Single step total (Forward + Backward) = {format_flops(total_f * 3)}")
    print("=" * 80)
