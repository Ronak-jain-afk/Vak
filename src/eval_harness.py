import os
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from src.generate import stream_generate
from src.model import VakModel
from src.tokenizer import VakTokenizer


class VakLMEvalAdapter(LM):
    """lm-evaluation-harness LM adapter wrapping VakModel and VakTokenizer."""

    def __init__(
        self,
        model: VakModel,
        tokenizer: VakTokenizer,
        device: Optional[torch.device] = None,
        max_seq_len: int = 1024,
        batch_size: int = 1,
    ):
        super().__init__()
        self._device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self._device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context."""
        results: List[Tuple[float, bool]] = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tokenizer.encode(context) if context else []
            cont_ids = self.tokenizer.encode(continuation) if continuation else []

            if not cont_ids:
                results.append((0.0, True))
                continue

            all_ids = ctx_ids + cont_ids
            if len(all_ids) > self.max_seq_len:
                all_ids = all_ids[-self.max_seq_len :]
                cont_len = min(len(cont_ids), len(all_ids) - 1)
            else:
                cont_len = len(cont_ids)

            input_ids = torch.tensor([all_ids[:-1]], dtype=torch.long, device=self._device)
            target_ids = torch.tensor(all_ids[1:], dtype=torch.long, device=self._device)

            with torch.no_grad():
                logits, _ = self.model(input_ids)
                log_probs = F.log_softmax(logits[0], dim=-1)

            cont_log_probs = log_probs[-cont_len:]
            cont_targets = target_ids[-cont_len:]

            selected_lp = cont_log_probs.gather(1, cont_targets.unsqueeze(1)).squeeze(1)
            sum_lp = float(selected_lp.sum().item())
            is_greedy = bool((cont_log_probs.argmax(dim=-1) == cont_targets).all().item())
            results.append((sum_lp, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        """Compute rolling log-likelihood across sequences for perplexity evaluation."""
        results: List[float] = []
        for req in requests:
            (string,) = req.args
            token_ids = self.tokenizer.encode(string)
            if len(token_ids) <= 1:
                results.append(0.0)
                continue

            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[-self.max_seq_len :]

            input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long, device=self._device)
            target_ids = torch.tensor(token_ids[1:], dtype=torch.long, device=self._device)

            with torch.no_grad():
                logits, _ = self.model(input_ids)
                log_probs = F.log_softmax(logits[0], dim=-1)

            selected_lp = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            results.append(float(selected_lp.sum().item()))

        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate tokens until a stop token or max tokens limit is reached."""
        results: List[str] = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks", 64)
            temperature = gen_kwargs.get("temperature", 0.0)
            top_p = gen_kwargs.get("top_p", 1.0)

            ctx_ids = self.tokenizer.encode(context)
            if not ctx_ids:
                ctx_ids = [self.tokenizer.bos_id]

            tokens: List[str] = []
            for piece in stream_generate(
                self.model,
                self.tokenizer,
                ctx_ids,
                max_new_tokens=max_gen_toks,
                temperature=temperature,
                top_p=top_p,
                device=self._device,
            ):
                tokens.append(piece)
                current_text = "".join(tokens)
                if any(stop_str in current_text for stop_str in until):
                    break

            results.append("".join(tokens).strip())

        return results
