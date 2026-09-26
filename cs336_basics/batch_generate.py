import torch
import einops
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from jaxtyping import Int, Float, Bool
from cs336_basics.model import TransformerLM
from cs336_basics.utils import softmax, load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.optimizer import AdamW

def sample_top_p(logits: Float[torch.Tensor, "... vocab_size"], temperature: float, p: float) -> Int[torch.Tensor, "..."]:
    probs = softmax(logits/temperature, -1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulatvie_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulatvie_probs > p
    # move along the last dim by one position becasue we are looking for the *first* position exceeding p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    sorted_probs[sorted_indices_to_remove] = 0.0
    filtered_probs = torch.zeros_like(probs).scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    filtered_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)
    original_shape = filtered_probs.shape
    prob_2d = einops.rearrange(filtered_probs, "... vocab_size -> (...) vocab_size")
    return torch.multinomial(prob_2d, num_samples=1).reshape(original_shape[:-1])


def batch_generate_tokens(input_ids: Int[torch.Tensor, "... sequence_length"],
                          model: TransformerLM, context_length: int, max_tokens: int, temperature: float,
                          p: float, eot_id: int) -> tuple[Int[torch.Tensor, "... sequence_length"], Bool[torch.Tensor, "..."]]:
    # prerequisite: input_ids have the same length
    generated_ids: Int[torch.Tensor, "..."] = input_ids.clone() 
    is_eot_generated = torch.zeros(generated_ids.shape[:-1], dtype=torch.bool, device=generated_ids.device)
    num_tokens_generated = 0
    while not is_eot_generated.all().item() and num_tokens_generated < max_tokens:
        # if the length of current generated ids exceed context_length, we truncate
        start_idx = max(0, generated_ids.shape[-1] - context_length)
        logits = model.forward(generated_ids[..., start_idx:]) # shape "... sequence_length vocab_size"
        next_token = sample_top_p(logits[..., -1, :], temperature, p) # shape "..."
        num_tokens_generated += 1
        is_eot_generated[next_token == eot_id] = True
        next_token[is_eot_generated] = eot_id
        generated_ids = torch.concat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
    return generated_ids, is_eot_generated


def batch_decode(input_ids: Int[torch.Tensor, "batch_size sequence_length"],
                 generated_ids: Int[torch.Tensor, "batch_size sequence_length"],
                 tokenizer: Tokenizer, eot_id: int, device: torch.device,
                 is_eot_generated: Bool[torch.Tensor, "batch_size"]) -> list[str]:
    # only consider EOT after the input token positions
    id_mask = torch.arange(generated_ids.shape[-1], device=device) >= input_ids.shape[-1] # 1d tensor
    value_mask = generated_ids == eot_id # shape "batch_size sequence_length"
    first_occurrence = torch.argmax(id_mask&value_mask, dim=-1)
    # for those that do not contain EOT, first_occurrence should be generated_ids.shape[-1]
    first_occurrence[~is_eot_generated] = generated_ids.shape[-1]
    return [tokenizer.decode(generated_ids[i, :first_occurrence[i]].cpu().tolist()) for i in range(len(generated_ids))]


def batch_encode(prompts: list[str], tokenizer: Tokenizer, pad_value: int, device: torch.device) -> Int[torch.Tensor, "batch_size sequence_length"]:
    max_len = 0
    input_ids: list[list[int]] = []
    for prompt in prompts:
        encoded = tokenizer.encode(prompt)
        input_ids.append(encoded)
        max_len = max(max_len, len(encoded))
    # left pad to input_ids
    for i in range(len(input_ids)):
        input_ids[i] = [pad_value] * (max_len - len(input_ids[i])) + input_ids[i]
    return torch.tensor(input_ids, device=device)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=" * 60)
    print("Type your prompt and press Enter.")
    print("To quit, type 'exit', 'q', or press Ctrl+C.")
    print("=" * 60)

    eot = "<|endoftext|>"
    tokenizer = Tokenizer.from_file(cfg.tokenizer.vocab_path, cfg.tokenizer.merges_path, special_tokens=[eot])
    device = torch.device(cfg.training.device)
    model = TransformerLM(cfg.model.num_layers, cfg.model.vocab_size, cfg.model.d_model, cfg.model.num_heads,
                          cfg.model.d_ff, cfg.model.rope_theta, cfg.model.context_length, device=cfg.training.device)
    load_checkpoint(cfg.model.ckpt_path, model)    
    eot_id = tokenizer.encode(eot)[0]
    model.eval()
    with torch.inference_mode():
        try:
            while True:
                # 1. Capture user prompt
                user_input = input("\nUser ❯ ").strip()
                
                # 2. Check for exit keywords/keys
                if user_input.lower() in ['q', 'exit', 'quit']:
                    print("\nExiting script. Goodbye!")
                    break
                    
                if not user_input:
                    continue  # Handle empty enters cleanly
                
                # 3. Run your custom autoregressive decoding function
                print("Model ❯ Generating...", end="\r")
                input_ids = batch_encode([user_input], tokenizer, 0, device)
                output_ids, is_eot_generated = batch_generate_tokens(input_ids, model, cfg.model.context_length,
                                                                     cfg.inference.max_tokens,
                                                                     cfg.inference.temperature, cfg.inference.top_p,
                                                                     eot_id)
                responses = batch_decode(input_ids, output_ids, tokenizer, eot_id, device, is_eot_generated)
                # 4. Print results
                print(f"Model ❯ {responses}")

        except KeyboardInterrupt:
            # Handles physical Ctrl+C press cleanly without crashing out with an ugly traceback
            print("\n\nSession interrupted by user (Ctrl+C). Exiting!")


if __name__ == "__main__":
   main()
