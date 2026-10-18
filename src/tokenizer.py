import os
from typing import Dict, Iterator, List, Optional, Tuple, Union
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|bos|>", "<|user|>", "<|assistant|>", "<|end|>"]
BOS_TOKEN = "<|bos|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"


class VakTokenizer:
    def __init__(self, tokenizer_file: Optional[str] = None):
        if tokenizer_file is not None and os.path.exists(tokenizer_file):
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            self.tokenizer = self._create_empty_tokenizer()

        self.bos_id = self.tokenizer.token_to_id(BOS_TOKEN)
        self.user_id = self.tokenizer.token_to_id(USER_TOKEN)
        self.assistant_id = self.tokenizer.token_to_id(ASSISTANT_TOKEN)
        self.end_id = self.tokenizer.token_to_id(END_TOKEN)

    def _create_empty_tokenizer(self) -> Tokenizer:
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        return tok

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def train_from_iterator(self, iterator: Iterator[str], vocab_size: int = 8192) -> None:
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        self.tokenizer.train_from_iterator(iterator, trainer=trainer)
        self.bos_id = self.tokenizer.token_to_id(BOS_TOKEN)
        self.user_id = self.tokenizer.token_to_id(USER_TOKEN)
        self.assistant_id = self.tokenizer.token_to_id(ASSISTANT_TOKEN)
        self.end_id = self.tokenizer.token_to_id(END_TOKEN)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.tokenizer.save(path)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        enc = self.tokenizer.encode(text)
        tokens = enc.ids
        if add_special_tokens:
            tokens = [self.bos_id] + tokens
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def format_chat(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        formatted = BOS_TOKEN
        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()
            if role == "user":
                formatted += f"{USER_TOKEN}{content}{END_TOKEN}"
            elif role == "assistant":
                formatted += f"{ASSISTANT_TOKEN}{content}{END_TOKEN}"
        if add_generation_prompt:
            formatted += ASSISTANT_TOKEN
        return formatted

    def prepare_chat_input(self, messages: List[Dict[str, str]], max_prompt_len: int = 768) -> Tuple[Optional[List[int]], Optional[str]]:
        if not messages:
            return None, "Empty input provided."

        last_msg = messages[-1]
        if last_msg["role"] != "user" or not last_msg["content"].strip():
            return None, "Latest turn must be a non-empty user message."

        latest_text = f"{BOS_TOKEN}{USER_TOKEN}{last_msg['content'].strip()}{END_TOKEN}{ASSISTANT_TOKEN}"
        latest_ids = self.encode(latest_text)
        if len(latest_ids) > max_prompt_len:
            return None, f"Input message exceeds maximum prompt length of {max_prompt_len} tokens."

        history = list(messages)
        while len(history) > 1:
            candidate_text = self.format_chat(history, add_generation_prompt=True)
            candidate_ids = self.encode(candidate_text)
            if len(candidate_ids) <= max_prompt_len:
                return candidate_ids, None
            # Drop oldest full user-assistant turn to preserve role alternation
            history.pop(0)
            if len(history) > 1:
                history.pop(0)

        return latest_ids, None
