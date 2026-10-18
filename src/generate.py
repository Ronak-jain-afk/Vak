from typing import Iterator, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from src.model import VakModel
from src.tokenizer import VakTokenizer


def sample_next_token(
    logits: torch.Tensor,
    generated_tokens: List[int],
    temperature: float = 0.5,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> int:
    logits = logits.clone().float()

    if repetition_penalty != 1.0 and generated_tokens:
        for tid in set(generated_tokens):
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty
            else:
                logits[tid] *= repetition_penalty

    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k > 0 and top_k < logits.size(-1):
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = -float("Inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return int(next_token)


@torch.no_grad()
def stream_generate(
    model: VakModel,
    tokenizer: VakTokenizer,
    prompt_tokens: List[int],
    max_new_tokens: int = 256,
    temperature: float = 0.5,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: Optional[torch.device] = None,
) -> Iterator[str]:
    if device is None:
        device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    prompt_len = len(prompt_tokens)
    if prompt_len >= model.cfg.max_seq_len:
        raise ValueError(f"Prompt length {prompt_len} exceeds max sequence length {model.cfg.max_seq_len}")

    kv_caches = model.init_kv_cache(batch_size=1, device=device, dtype=model_dtype)

    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    logits, _ = model(input_tensor, start_pos=0, kv_caches=kv_caches)
    next_logits = logits[0, -1, :]

    generated_tokens: List[int] = []
    current_pos = prompt_len

    for _ in range(max_new_tokens):
        if current_pos >= model.cfg.max_seq_len:
            return False

        penalty_window = generated_tokens[-64:] if generated_tokens else []
        next_token = sample_next_token(
            next_logits,
            generated_tokens=penalty_window,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if next_token == tokenizer.end_id:
            return True

        generated_tokens.append(next_token)
        token_str = tokenizer.decode([next_token])
        yield token_str

        # Incremental decode: single token forward pass with cached keys and values
        next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, _ = model(next_input, start_pos=current_pos, kv_caches=kv_caches)
        next_logits = logits[0, -1, :]
        current_pos += 1

    return False


@torch.no_grad()
def generate_response(
    model: VakModel,
    tokenizer: VakTokenizer,
    prompt_tokens: List[int],
    max_new_tokens: int = 256,
    temperature: float = 0.5,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: Optional[torch.device] = None,
    return_end_flag: bool = False,
) -> Union[str, Tuple[str, bool]]:
    tokens = []
    num_generated = 0
    gen = stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        device=device,
    )
    reached_end = False
    try:
        while True:
            token_piece = next(gen)
            tokens.append(token_piece)
            num_generated += 1
    except StopIteration as e:
        reached_end = bool(e.value)

    text = "".join(tokens)
    if return_end_flag:
        return text, reached_end
    return text
