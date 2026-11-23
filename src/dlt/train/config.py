"""Training configuration: a dataclass, optionally loaded from YAML, overridden with --set key=value."""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PINNED = {
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "Qwen/Qwen2.5-1.5B": "8faed761d45a263340a0528343f099c05c9a4323",
    "Qwen/Qwen2.5-3B": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
    "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796",
    "Qwen/Qwen2.5-14B": "97e1e76335b7017d8f67c08a19d103c0504298c9",
    "Qwen/Qwen2.5-72B": "efba10c8e54e91e0d9570ab5f7b51a958474d4cb",
    "Qwen/Qwen2.5-72B-Instruct": "495f39366efef23836d0cfae4fbe635880d2be31",  # eval reference only
}
BACKENDS = ("single", "ddp", "fsdp2", "fsdp1")  # fsdp1: adapter runs on a sharded NF4 base (Phase 7)


@dataclass
class TrainConfig:
    # model
    model: str = "Qwen/Qwen2.5-0.5B"
    revision: str | None = None  # None: the pinned revision from PINNED
    init: str = "pretrained"  # pretrained | random | tiny (2-layer toy, CPU tests)
    backend: str = "ddp"  # single | ddp | fsdp2
    attn: str = "sdpa"  # sdpa | eager (eager is deterministic)
    activation_checkpointing: bool = False
    ac_every: int = 1  # with activation_checkpointing: checkpoint decoder layers 0, k, 2k, ...
    sharded_load: bool = True  # FSDP2 + pretrained: meta init, each rank reads only its shard
    sdpa_backend: str = "auto"  # auto | flash | efficient | cudnn | math (attn=sdpa only)
    loss_chunk: int = 0  # >0: lm_head + cross-entropy over this many tokens at a time (no full logits)
    # distributed
    ddp_bucket_mb: float = 25.0  # DDP gradient bucket size
    ddp_bucket_view: bool = False  # DDP gradient_as_bucket_view (gradients live in the buckets)
    reshard_after_forward: bool = True  # FSDP2: free each layer's gathered parameters after forward
    fsdp_prefetch: int = 0  # FSDP2: explicitly prefetch this many layers ahead (0: FSDP2's default)
    fsdp_hold_params: bool = False  # FSDP2: keep layers unsharded across the micro-batches of a step
    compile: bool = False  # torch.compile each decoder layer (fuses the elementwise ops)
    compile_backend: str = "inductor"  # aot_eager in CPU tests (no C++ toolchain needed)
    # QLoRA (Phases 6-7): a frozen, optionally NF4-quantized base with LoRA adapters
    quant: str = ""  # "" | nf4 (bitsandbytes, double quant, BF16 compute)
    peft: str = ""  # "" | lora
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    adapter_out: str = ""  # save the final adapter here (peft save_pretrained) when the run completes
    # data
    task: str = "lm"  # lm: packed tokens (train.bin) | sft: prompt -> completion examples (BIRD jsonl)
    schemas: str = ""  # sft: BIRD schemas.json
    sft_limit: int = 0  # sft: use only the first N examples by question id (the overfit test); 0 = all
    data: str = ""  # lm: train.bin (flat uint32 tokens); sft: train.jsonl
    seq_len: int = 2048
    global_batch: int = 32  # sequences per optimizer step, across all ranks
    micro_batch: int = 4  # sequences per forward/backward on one rank
    seed: int = 0
    val_data: str = ""  # val.bin; with eval_samples > 0, val loss at step 0 and at the end of each run
    eval_samples: int = 0  # the first N val samples, split across ranks (multiple of world x micro_batch)
    # optimization
    steps: int = 100
    lr: float = 3e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 10
    weight_decay: float = 0.1
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    accum_sync: str = "last"  # last: sync gradients on the last micro-batch only | every: each micro-batch
    deterministic: bool = False
    # checkpointing
    ckpt_dir: str = ""  # empty: no checkpoints
    ckpt_every: int = 0  # 0: only when stopping
    keep_ckpts: int = 2
    resume: bool = True
    async_ckpt: bool = False  # DCP async_save: block only to stage the state, publish when written
    stop_on_term: bool = False  # treat SIGTERM like the wall-time USR1: checkpoint at the next step, exit
    # run control and test hooks
    run_name: str = ""
    max_seconds: float = 0.0  # stop (checkpoint, exit) after this much wall time; 0 = off
    stop_at_step: int = -1  # behave as if SIGUSR1 arrived before this step
    crash_at_step: int = -1  # SIGKILL every rank before this step (simulated node failure)
    crash_in_save: bool = False  # SIGKILL after the checkpoint is written but before it is published
    crash_in_eval: bool = False  # SIGKILL at the start of the end-of-run evaluation
    term_at_step: int = -1  # send SIGTERM to this process before this step (tests stop_on_term)
    oom_at_step: int = -1  # raise OutOfMemoryError on rank 0 before this step (tests the OOM record)
    write_record: bool = True
    # measurement
    timing_warmup: int = 3  # leading steps excluded from timing
    profile_steps: int = 0  # >0: profile this many steps for compute / exposed communication
    profile_start: int = 5  # first profiled step, counted from the run's first step
    profile_dir: str = ""  # keep the chrome traces here (default: analyzed, then deleted)
    profiler: str = "torch"  # torch: PyTorch Profiler, analyzed in-process | nsys: cudaProfilerApi range
    nvtx: bool = False  # NVTX ranges around forward, backward and optimizer (for Nsight Systems)
    alloc_conf: str = ""  # PYTORCH_CUDA_ALLOC_CONF for this run, e.g. expandable_segments:True

    def resolved_revision(self) -> str | None:
        return self.revision or PINNED.get(self.model)

    def validate(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if self.init not in ("pretrained", "random", "tiny"):
            raise ValueError("init must be pretrained, random or tiny")
        if not self.data:
            raise ValueError("data (path to train.bin) is required")
        if self.accum_sync not in ("last", "every"):
            raise ValueError("accum_sync must be last or every")
        if self.sdpa_backend not in ("auto", "flash", "efficient", "cudnn", "math"):
            raise ValueError("sdpa_backend must be auto, flash, efficient, cudnn or math")
        if self.profiler not in ("torch", "nsys"):
            raise ValueError("profiler must be torch or nsys")
        if self.ac_every < 1 or self.loss_chunk < 0 or self.fsdp_prefetch < 0:
            raise ValueError("ac_every >= 1, loss_chunk >= 0 and fsdp_prefetch >= 0")
        if self.task not in ("lm", "sft") or (self.task == "sft" and not self.schemas):
            raise ValueError("task must be lm or sft; sft needs schemas")
        if self.quant not in ("", "nf4") or self.peft not in ("", "lora"):
            raise ValueError("quant must be '' or nf4; peft must be '' or lora")
        if self.quant and not self.peft:
            raise ValueError("a quantized base can only be trained through adapters (peft=lora)")
        if self.peft and (self.backend == "fsdp2" or self.loss_chunk or self.compile or self.ac_every > 1):
            raise ValueError("peft runs support single/ddp/fsdp1 without loss_chunk, compile or ac_every")
        if self.backend == "fsdp1" and not self.peft:
            raise ValueError("backend fsdp1 is only for adapter runs (full-parameter runs use fsdp2)")
        if self.eval_samples < 0 or (self.eval_samples and not self.val_data):
            raise ValueError("eval_samples needs val_data and must be >= 0")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["revision"] = self.resolved_revision()
        return d


def coerce(key: str, value):
    """Convert a YAML value or a --set string to the type of the field's default."""
    defaults = {f.name: getattr(TrainConfig(), f.name) for f in dataclasses.fields(TrainConfig)}
    if key not in defaults:
        raise ValueError(f"unknown config key {key!r}")
    default = defaults[key]
    if isinstance(value, str):
        text = value.strip()
        if isinstance(default, bool):
            if text.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"{key}: expected a boolean, got {value!r}")
            return text.lower() in ("true", "1", "yes")
        if isinstance(default, int):
            return int(text)
        if isinstance(default, float):
            return float(text)
        if isinstance(default, list):
            return yaml.safe_load(text)
        return None if text.lower() in ("none", "null") else text
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def parse(argv: list[str] | None = None) -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train a causal LM (see dlt.train.engine)")
    ap.add_argument("--config", type=Path, default=None, help="YAML file with TrainConfig fields")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a field")
    args = ap.parse_args(argv)
    cfg = TrainConfig()
    if args.config:
        for k, v in (yaml.safe_load(args.config.read_text()) or {}).items():
            setattr(cfg, k, coerce(k, v))
    for item in args.set:
        k, _, v = item.partition("=")
        setattr(cfg, k, coerce(k, v))
    for k in ("data", "val_data", "ckpt_dir", "profile_dir", "schemas", "adapter_out", "model"):
        # '${DLT_HOME}/...' expands inside the job
        setattr(cfg, k, os.path.expandvars(getattr(cfg, k)))
    cfg.validate()
    return cfg
