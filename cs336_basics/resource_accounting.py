from cs336_basics.model import TransformerLM

def model_size(vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int) -> int:
    embedding = vocab_size * d_model
    d_k = d_model // num_heads
    transformer_block = num_layers * (2 * d_model + 4 * num_heads * d_k * d_model + 3 * d_model * d_ff)
    ln_final = d_model
    lm_head = d_model * vocab_size
    return embedding + transformer_block + ln_final + lm_head

def compute_adamw_peak_memory(context_len, num_heads, num_layers, d_model, vocab_size, d_ff, batch_size):
    # assume all linear transformations' resultants are kept for backprop
    storage = ((8 * context_len * d_model + num_heads * context_len**2 + 3 * context_len * d_ff) * num_layers +
            3 * context_len * d_model + context_len * vocab_size) * batch_size + 4 * (12 * num_layers * d_model**2 + (2 * num_layers + 2 * vocab_size + 1) * d_model)
    return storage * 4 * 1e-9

def compute_forward_pass_flop(context_len, num_heads, num_layers, d_model, vocab_size, d_ff):
    d_k = d_model // num_heads
    return ((8 * context_len * d_model * num_heads * d_k + 4 * num_heads * context_len**2 * d_k + 6 *
            context_len * d_model * d_ff) * num_layers + 2 * context_len * d_model * vocab_size)

if __name__ == "__main__":
    vocab_size = 50257
    context_len = 1024
    num_layers = 45
    d_model = 1600
    num_heads = 25
    d_ff = 4288
    # 手写的计算模型参数量的函数
    size = model_size(vocab_size, d_model, num_heads, d_ff, num_layers)
    print("Model size: %d" % size)
    '''
    # 用内置函数进行统计
    model = TransformerLM(num_layers, vocab_size, d_model, num_heads, d_ff, 1e+4, context_len)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model size: %d" % trainable_params)
    '''

    print("batch size = 1, peak memory: %.2fGB" % compute_adamw_peak_memory(context_len, num_heads, num_layers, d_model,
                                                                        vocab_size, d_ff, batch_size=1))
    print("batch size = 2, peak memory: %.2fGB" % compute_adamw_peak_memory(context_len, num_heads, num_layers, d_model,
                                                                        vocab_size, d_ff, batch_size=2))
    flops = 400000 * 1024 * 3 * compute_forward_pass_flop(context_len, num_heads, num_layers, d_model, vocab_size, d_ff) * 1e-12
    print("With batch size = 1024, 400K steps of forward pass + backward pass results in %f TFLOPs, requiring %f days of training." %
          (flops, flops / (495 * 0.5) / 3600 / 24))

    vocab_size = 10000
    context_len = 256
    num_layers = 4
    d_model = 512
    num_heads = 16
    d_ff = 1344
    batch_size = 32
    num_steps = 40000
    flops = num_steps * batch_size * 3 * compute_forward_pass_flop(context_len, num_heads, num_layers, d_model, vocab_size, d_ff) * 1e-12
    compute_RTX4090D = 59 # unit = TFLOPS
    MFU = 0.5
    print("With batch size = %d, %d steps of forward pass + backward pass results in %f TFLOPs, requiring %f hours of training." %
          (batch_size, num_steps, flops, flops / (compute_RTX4090D * MFU) / 3600))
