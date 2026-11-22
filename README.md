# distributed-llm-training

Distributed LLM training on Northeastern Explorer H200s: a tested DDP/FSDP2 training engine, a
full-parameter scaling and memory study on the Qwen2.5 size ladder, and a Qwen2.5-72B QLoRA fine-tune
evaluated against its base model.

Status: Phase 0 done (hardware manifest and decisions); Phase 1 next. See [PLAN.md](PLAN.md) for the phases and
[docs/phase0.md](docs/phase0.md) for what has been measured so far.

Every number in this README will be generated from files in `results/`.
