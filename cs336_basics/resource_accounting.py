from cs336_basics.utils import model_size
from cs336_basics.model import TransformerLM

if __name__ == "__main__":
    vocab_size = 50257
    context_length = 1024
    num_layers = 45
    d_model = 1600
    num_heads = 25
    d_ff = 4288
    # 手写的计算模型参数量的函数
    size = model_size(vocab_size, d_model, num_heads, d_ff, num_layers)
    print("Model size: %d" % size)
    # 用内置函数进行统计
    model = TransformerLM(num_layers, vocab_size, d_model, num_heads, d_ff, 1e+4, context_length)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model size: %d" % trainable_params)