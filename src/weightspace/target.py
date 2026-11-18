import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from weightspace.tasks import Completion

END_OF_REASONING = "</think>"
QWEN_RECOMMENDED = {"temp": 0.6, "top_p": 0.95, "top_k": 20}


def split_answer(text: str, thinking: bool) -> str | None:
    if END_OF_REASONING in text:
        return text.split(END_OF_REASONING)[-1].strip()
    return None if thinking else text.strip()


class Target:
    def __init__(self, path: str):
        self.model, self.tokenizer = load(path)

    def encode(self, prompt: str, thinking: bool) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=thinking,
        )
        return self.tokenizer.encode(text)

    def complete(self, prompts: list[str], *, thinking: bool, max_tokens: int, batch_size: int = 4, sampled: bool = False) -> list[Completion]:
        mx.random.seed(0)
        sampler = make_sampler(**QWEN_RECOMMENDED) if sampled else None
        completions = []
        for start in range(0, len(prompts), batch_size):
            batch = [self.encode(p, thinking) for p in prompts[start : start + batch_size]]
            response = batch_generate(self.model, self.tokenizer, batch, max_tokens=max_tokens, return_token_ids=True, sampler=sampler)
            for text, ids in zip(response.texts, response.token_ids):
                completions.append(
                    Completion(
                        text=text,
                        answer=split_answer(text, thinking),
                        finished=len(ids) < max_tokens,
                        tokens=len(ids),
                    )
                )
        return completions
