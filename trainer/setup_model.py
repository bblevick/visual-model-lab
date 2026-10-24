from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
snapshot_download(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
    local_dir=str(root / ".local/base-model"),
    allow_patterns=[
        "*.safetensors",
        "config.json",
        "tokenizer*",
        "special_tokens_map.json",
        "generation_config.json",
        "README.md",
        "LICENSE",
    ],
)
