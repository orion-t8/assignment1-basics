from cs336_basics.utils import model_size
from cs336_basics.model import TransformerLM

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
