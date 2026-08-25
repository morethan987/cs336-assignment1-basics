def compute_adamw_fp32_peak_memory(
    batch_size: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
):
    """
    GPU memory cost by Transformer with float32 and AdamW
    """
    bytes_per_fp32 = 4

    # 1. Trainable parameters
    # (1) Input Embedding + Output Embedding (LM Head)
    params_embeddings = 2 * vocab_size * d_model

    # (2) Single Transformer Block
    # - 2 RMSNorm (Pre-Attn, Pre-FFN)
    params_rmsnorms_per_layer = 2 * d_model
    # - MHSA (W_Q, W_K, W_V, W_O)
    params_mhsa_per_layer = 4 * (d_model**2)
    # - SwiGLU: d_ff = (8/3) * d_model -> 3 linear layer (gate, up, down)
    params_swiglu_per_layer = int(3 * d_model * (8 / 3 * d_model))  # 8 * d_model^2

    params_per_layer = params_rmsnorms_per_layer + params_mhsa_per_layer + params_swiglu_per_layer
    params_all_layers = num_layers * params_per_layer

    # (3) Final RMSNorm
    params_final_rmsnorm = d_model

    total_params = params_embeddings + params_all_layers + params_final_rmsnorm
    memory_params = total_params * bytes_per_fp32

    # 2. Gradients
    # every param has a FP32 gradient
    memory_gradients = total_params * bytes_per_fp32

    # 3. AdamW Optimizer State
    # every param has two FP32 state: m_t and v_t
    memory_optimizer = 2 * total_params * bytes_per_fp32

    # 4. Activations
    # (1) Single Transformer Block
    # - RMSNorm (Pre-Attn)
    act_rmsnorm1 = batch_size * context_length * d_model
    # - MHSA: QKV input and output (4*B*S*D) + Attn/Softmax (2*B*H*S^2) + Out_proj (1*B*S*D)
    act_mhsa = 5 * batch_size * context_length * d_model + 2 * batch_size * num_heads * (context_length**2)
    # - RMSNorm (Pre-FFN)
    act_rmsnorm2 = batch_size * context_length * d_model
    # - SwiGLU: input (B*S*D) + 3 internal tensor (B*S*d_ff = 8/3*B*S*D) -> 9*B*S*D
    act_swiglu = batch_size * context_length * d_model + int(3 * batch_size * context_length * (8 / 3 * d_model))

    act_elements_per_layer = act_rmsnorm1 + act_mhsa + act_rmsnorm2 + act_swiglu
    act_elements_all_layers = num_layers * act_elements_per_layer

    # (2) other
    act_final_rmsnorm = batch_size * context_length * d_model
    act_output_embedding = batch_size * context_length * d_model
    act_cross_entropy = batch_size * context_length * vocab_size

    total_act_elements = act_elements_all_layers + act_final_rmsnorm + act_output_embedding + act_cross_entropy
    memory_activations = total_act_elements * bytes_per_fp32

    # 5. Total Peak Memory
    memory_total = memory_params + memory_gradients + memory_optimizer + memory_activations

    return {
        "total_params": total_params,
        "memory_params_bytes": memory_params,
        "memory_gradients_bytes": memory_gradients,
        "memory_optimizer_bytes": memory_optimizer,
        "memory_activations_bytes": memory_activations,
        "memory_total_bytes": memory_total,
    }


def format_bytes(bytes_val):
    mb = bytes_val / (1024**2)
    gb = bytes_val / (1024**3)
    return f"{bytes_val:>15,} Bytes | {mb:>10.2f} MB | {gb:>8.2f} GB"


if __name__ == "__main__":
    config = {
        "batch_size": 4,
        "vocab_size": 50257,
        "context_length": 1024,
        "num_layers": 48,
        "d_model": 1600,
        "num_heads": 25,
    }

    print("=" * 70)
    print("Model hyperparams:")
    for k, v in config.items():
        print(f"  - {k:<18}: {v}")
    print("=" * 70)

    res = compute_adamw_fp32_peak_memory(**config)

    print(f"Parameters: {res['total_params']:,} ({res['total_params'] / 1e6:.2f} M)")
    print("-" * 70)
    print("GPU memory cost details:")
    print(f"  1. Params      : {format_bytes(res['memory_params_bytes'])}")
    print(f"  2. Gradients   : {format_bytes(res['memory_gradients_bytes'])}")
    print(f"  3. AdamW       : {format_bytes(res['memory_optimizer_bytes'])}")
    print(f"  4. Activations : {format_bytes(res['memory_activations_bytes'])}")
    print("-" * 70)
    print(f"  Total cost     : {format_bytes(res['memory_total_bytes'])}")
    print("=" * 70)

    # 打印占比
    total = res["memory_total_bytes"]
    print("Percentages:")
    print(f"  - Params       : {res['memory_params_bytes'] / total * 100:>6.2f} %")
    print(f"  - Gradients    : {res['memory_gradients_bytes'] / total * 100:>6.2f} %")
    print(f"  - Optimizer    : {res['memory_optimizer_bytes'] / total * 100:>6.2f} %")
    print(f"  - Activations  : {res['memory_activations_bytes'] / total * 100:>6.2f} %")
    print("=" * 70)
