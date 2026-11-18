import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

from weightspace.tasks import Completion

END_OF_REASONING = "</think>"
QWEN_RECOMMENDED = {"temp": 0.6, "top_p": 0.95, "top_k": 20}
LOOP_WINDOW = 50
LOOP_REPEATS = 3
LOOP_CHECK_EVERY = 16


def split_answer(text: str, thinking: bool) -> str | None:
    if END_OF_REASONING in text:
        return text.split(END_OF_REASONING)[-1].strip()
    return None if thinking else text.strip()


def looping(tokens: list[int]) -> bool:
    if len(tokens) < LOOP_WINDOW * LOOP_REPEATS:
        return False
    window = tokens[-LOOP_WINDOW:]
    hits = sum(tokens[i : i + LOOP_WINDOW] == window for i in range(len(tokens) - LOOP_WINDOW + 1))
    return hits >= LOOP_REPEATS


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

    def complete(
        self,
        prompts: list[str],
        *,
        thinking: bool,
        max_tokens: int,
        batch_size: int = 4,
        sampled: bool = False,
    ) -> list[Completion]:
        mx.random.seed(0)
        generator = BatchGenerator(
            self.model,
            stop_tokens=[[t] for t in self.tokenizer.eos_token_ids],
            sampler=make_sampler(**QWEN_RECOMMENDED) if sampled else None,
            completion_batch_size=batch_size,
            prefill_batch_size=batch_size,
        )
        uids = generator.insert([self.encode(p, thinking) for p in prompts], [max_tokens] * len(prompts))
        tokens = {uid: [] for uid in uids}
        ended = {}
        flagged = set()
        while responses := generator.next_generated():
            for r in responses:
                if r.finish_reason == "stop":
                    ended[r.uid] = "stop"
                    continue
                tokens[r.uid].append(r.token)
                if r.finish_reason == "length":
                    ended[r.uid] = "length"
                elif r.uid not in flagged and len(tokens[r.uid]) % LOOP_CHECK_EVERY == 0 and looping(tokens[r.uid]):
                    flagged.add(r.uid)
        generator.close()
        completions = []
        for uid in uids:
            text = self.tokenizer.decode(tokens[uid])
            completions.append(
                Completion(
                    text=text,
                    answer=split_answer(text, thinking),
                    finished=ended.get(uid) == "stop",
                    tokens=len(tokens[uid]),
                    looped=uid in flagged,
                )
            )
        return completions
