"""Supervised fine-tuning examples (prompt -> completion) for BIRD, in the engine's batch interface.

Each example is ``prompt + target + end-of-text``, tokenized as two pieces so training sees exactly the
token boundary that generation does (the prompt is encoded alone at inference). Inputs are the sequence
without its last token, targets the sequence without its first; targets inside the prompt and padding are
-100, so the loss covers only the SQL and the end-of-text token. Right padding needs no attention mask
with causal attention: real tokens never attend to later padding.

The schema with sample rows is used when the example fits ``max_len``, else the plain schema; an example
that does not fit either way is dropped and counted.

Step s still takes the same ``global_batch`` examples from ``dlt.data.packed.GlobalOrder`` (resume needs
only the step). Within the step, examples are sorted by length and dealt round-robin to ranks, then cut
into micro-batches, which keeps padding low and the ranks' work balanced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from dlt.bird.prompt import prompt, target
from dlt.data.packed import GlobalOrder


@dataclass(frozen=True)
class Example:
    ids: tuple[int, ...]  # prompt + target + eos
    n_prompt: int
    question_id: int


class SFTExamples:
    def __init__(self, jsonl: str | Path, schemas: str | Path, tokenizer, max_len: int, limit: int = 0):
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        rows = [json.loads(line) for line in Path(jsonl).read_text().splitlines() if line.strip()]
        if limit:
            rows = sorted(rows, key=lambda r: r["question_id"])[:limit]
        schema = json.loads(Path(schemas).read_text())
        self.items: list[Example] = []
        self.dropped = 0
        self.used_plain_schema = 0
        eos = tokenizer.eos_token_id
        for r in rows:
            t = tokenizer(target(r)).input_ids + [eos]
            for i, key in enumerate(("ddl_rows", "ddl")):
                p = tokenizer(prompt(r, schema[r["db_id"]][key])).input_ids
                if len(p) + len(t) <= max_len:
                    self.items.append(Example(tuple(p + t), len(p), r["question_id"]))
                    self.used_plain_schema += i
                    break
            else:
                self.dropped += 1

    def __len__(self) -> int:
        return len(self.items)

    def length(self, i: int) -> int:
        return len(self.items[i].ids) - 1

    def batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ex = [self.items[i] for i in indices.tolist()]
        width = max(len(e.ids) for e in ex) - 1
        x = torch.full((len(ex), width), self.pad_id, dtype=torch.long)
        y = torch.full((len(ex), width), -100, dtype=torch.long)
        for row, e in enumerate(ex):
            ids = torch.tensor(e.ids, dtype=torch.long)
            n = len(e.ids) - 1
            x[row, :n] = ids[:-1]
            y[row, :n] = ids[1:]
            y[row, : e.n_prompt - 1] = -100  # positions that predict prompt tokens
        return x, y

    def micro_batches(
        self, order: GlobalOrder, step: int, rank: int, world: int, micro: int
    ) -> list[torch.Tensor]:
        idx = order.global_indices(step).tolist()
        if len(idx) % (world * micro):
            raise ValueError(f"global batch {len(idx)} not divisible by world {world} x micro {micro}")
        idx.sort(key=self.length)
        mine = torch.tensor(idx[rank::world])
        return list(mine.split(micro))

    def target_tokens(self, indices: torch.Tensor) -> int:
        return sum(len(self.items[i].ids) - self.items[i].n_prompt for i in indices.tolist())
