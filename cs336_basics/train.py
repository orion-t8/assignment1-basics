import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import numpy as np
import torch
import random
from cs336_basics.utils import data_loading, cross_entropy, learning_rate_schedule, gradient_clipping, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW

@hydra.main(version_base=None, config_path="conf", config_name="config")
def training_loop(cfg: DictConfig) -> None:
    random.seed(cfg.training.seed) 
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if cfg.training.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.training.seed)
    
    train_data = np.load(to_absolute_path(cfg.data.train_path), mmap_mode='r')
    valid_data = np.load(to_absolute_path(cfg.data.valid_path), mmap_mode='r')
    batch_size = cfg.training.batch_size
    context_length = cfg.model.context_length
    max_steps = cfg.training.max_tokens_processed // (cfg.training.batch_size * cfg.model.context_length)

    transformer_lm = TransformerLM(cfg.model.num_layers, cfg.model.vocab_size, cfg.model.d_model, cfg.model.num_heads,
                        cfg.model.d_ff, cfg.model.rope_theta, cfg.model.context_length, 1e-5, cfg.training.device)
    max_lr = cfg.lr_scheduler.max_learning_rate
    min_lr = cfg.lr_scheduler.min_learning_rate
    warmup_iters = cfg.lr_scheduler.warmup_iters
    cosine_cycle_iters = cfg.lr_scheduler.cosine_cycle_iters

    optimizer = AdamW(transformer_lm.parameters(), learning_rate_schedule(0, max_lr, min_lr, warmup_iters, cosine_cycle_iters),
                      cfg.optimizer.weight_decay, (cfg.optimizer.beta1, cfg.optimizer.beta2), cfg.optimizer.eps)

    for t in range(max_steps):
        x, y = data_loading(train_data, batch_size, context_length, cfg.training.device)
        optimizer.zero_grad()
        logits = transformer_lm.forward(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        gradient_clipping(transformer_lm.parameters(), cfg.training.max_grad_norm)
        lr = learning_rate_schedule(t, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        completed_steps = t+1
        if completed_steps % cfg.training.eval_interval == 0:
            transformer_lm.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for _ in range(cfg.training.eval_iters):
                    eval_x, eval_y = data_loading(valid_data, batch_size, context_length, cfg.training.device)
                    eval_logits = transformer_lm.forward(eval_x)
                    eval_loss += cross_entropy(eval_logits, eval_y).item()
            eval_loss /= cfg.training.eval_iters
            print("After step %d, training loss: %f, validation loss: %f" % (t+1, loss.item(), eval_loss))
            transformer_lm.train()

        if completed_steps % cfg.training.save_interval == 0:
            save_checkpoint(transformer_lm, optimizer, completed_steps, to_absolute_path(cfg.training.ckpt_path))

if __name__ == "__main__":
    training_loop()