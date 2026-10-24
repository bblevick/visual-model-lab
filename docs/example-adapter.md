# Preserved example adapter

The learned weights in artifacts/example-adapter/adapters.safetensors are the actual 60-step MLX output, with SHA-256 recorded in public/training-run.json. They are not mock weights. The accompanying adapter_config.json replaces machine-specific model/data/output paths with portable locations; optimizer and LoRA parameters are unchanged. The original local configuration remains in the ignored .local/latest run.

The model source/revision, synthetic split hashes, sampled loss metrics, four held-out before/after generations and limitations are preserved in the report. Training used Apple M4 / 24 GB RAM. Base weights are not committed. See trainer/setup_model.py to download the pinned base model.
