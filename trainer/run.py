"""Real local MLX LoRA training and source-backed report generation."""

import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(steps=60, learning_rate=0.0003, data=None, out=None):
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    data = Path(data or ROOT / "trainer/data")
    out = Path(out or ROOT / ".local/latest")
    out.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / ".local/base-model"
    if not (model_path / "model.safetensors").exists():
        raise RuntimeError(
            "Download the base model first: python trainer/setup_model.py"
        )
    rows = [
        json.loads(s)
        for s in (data / "test.jsonl").read_text().splitlines()
        if s.strip()
    ][:4]
    prompts = [row["messages"][0]["content"] for row in rows]
    started = time.time()
    model, tokenizer = load(str(model_path))

    def samples():
        result = []
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            result.append(
                generate(
                    model,
                    tokenizer,
                    prompt=text,
                    max_tokens=45,
                    sampler=make_sampler(temp=0),
                    verbose=False,
                )
            )
        return result

    before = samples()
    del model
    mx.clear_cache()
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        str(model_path),
        "--train",
        "--data",
        str(data),
        "--iters",
        str(steps),
        "--num-layers",
        "4",
        "--batch-size",
        "2",
        "--max-seq-length",
        "128",
        "--learning-rate",
        str(learning_rate),
        "--steps-per-report",
        "10",
        "--steps-per-eval",
        "20",
        "--val-batches",
        "2",
        "--save-every",
        str(steps),
        "--adapter-path",
        str(out / "adapter"),
        "--seed",
        "42",
        "--mask-prompt",
    ]
    log = []
    with (out / "training.log").open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
            log.append(line)
        if process.wait() != 0:
            raise RuntimeError("MLX training failed; inspect training.log")
    model, tokenizer = load(str(model_path), adapter_path=str(out / "adapter"))
    after = samples()
    del model
    mx.clear_cache()
    text = "".join(log)
    metrics = []
    for line in log:
        cleaned = re.sub(r"\x1b\[[0-9;]*m", "", line)
        match = re.search(r"^\s*(\d+)\s+(?:(val)\s+)?([0-9]+\.[0-9]+)", cleaned)
        if match:
            metrics.append(
                {
                    "step": int(match[1]),
                    "split": "val" if match[2] else "train",
                    "loss": float(match[3]),
                }
            )
    adapter = out / "adapter/adapters.safetensors"
    if not adapter.exists():
        raise RuntimeError("No trained adapter was saved")
    report = {
        "version": 1,
        "kind": "actual-local-lora-run",
        "model": MODEL_ID,
        "modelRevision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "device": str(mx.default_device()),
        "hardware": "Apple Silicon; locally executed MLX",
        "steps": steps,
        "learningRate": learning_rate,
        "durationSeconds": round(time.time() - started, 2),
        "completedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": 42,
        "trainingRows": len((data / "train.jsonl").read_text().splitlines()),
        "validationRows": len((data / "valid.jsonl").read_text().splitlines()),
        "testRows": len((data / "test.jsonl").read_text().splitlines()),
        "datasetSHA256": {
            name: digest(data / (name + ".jsonl"))
            for name in ["train", "valid", "test"]
        },
        "baseWeightsSHA256": digest(model_path / "model.safetensors"),
        "adapterSHA256": digest(adapter),
        "adapterBytes": adapter.stat().st_size,
        "metrics": metrics,
        "comparisons": [
            {
                "prompt": prompt,
                "before": b,
                "after": a,
                "reference": row["messages"][-1]["content"],
            }
            for prompt, b, a, row in zip(prompts, before, after, rows)
        ],
        "limitations": "Small synthetic formatting task. Held-out streets differ from training streets; templates are shared. Lower loss and changed outputs do not establish broad model improvement. LoRA updates adapters, not all base weights.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT_SAVED", out / "report.json", flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, choices=[20, 60, 120], default=60)
    p.add_argument(
        "--learning-rate", type=float, choices=[0.0001, 0.0003, 0.001], default=0.0003
    )
    p.add_argument("--data")
    p.add_argument("--out")
    args = p.parse_args()
    run(args.steps, args.learning_rate, args.data, args.out)
