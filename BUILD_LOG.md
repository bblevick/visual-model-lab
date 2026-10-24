# Build log

## October 24, 2026 — user-requested expansion to five projects

- User explicitly requested visual-model-lab alongside the original three, and specifically requested real LLM fine-tuning in Visual Model Lab.
- Implemented Visual Model Lab — The Almost-Right Department with the original free/local constraints and its own repository, dependency lock and Pages workflow.
- TypeScript, TensorFlow.js MobileNet embeddings and a logistic head; optional Python/MLX trainer on Apple Silicon.
- Synthetic examples and a small formatting task. The model may already classify biased examples correctly. Intentionally reversed labels illustrate incorrect supervision explicitly. No pretrained weights are updated in vision mode. LLM mode trains real adapters on Apple Silicon; public hosting shows evidence, not a GPU training service. No broad quality, accuracy, calibrated probability or production-readiness claim.

Preserved actual MobileNet embeddings computed on WebGL. The biased preset initially produced zero held-out errors; kept that result rather than altering the metric. Added an explicitly deliberate label reversal so incorrect supervision can be inspected.

Executed MLX LoRA on the M4: 60 real steps, saved 1,308,381-byte adapter. Parsed the actual CLI’s ANSI-formatted logging, preserving the original raw log and correcting the report parser without rerunning that experiment. Sampled validation loss changed from 3.893 to 0.020. Four held-out generations were preserved before and after training. Two follow the reference format; the flooded-road output substitutes the wrong road and the clear-road output degenerates into repetition. Lower sampled loss did not eliminate these errors. This remains a synthetic formatting demonstration, not a general improvement claim.

Also executed a real 20-step job through the local HTTP controller and another through the actual local UI, verifying adapter output and the returned report. See docs/local-training-check.json. Base weights and private/local runs remain ignored; the example adapter and public report are intentionally included.

The explicit misfile intervention produced 12/12 held-out errors; correcting with all training labels produced 0/12 on this fixed synthetic set. These are measured fixture outcomes, not an external benchmark.

Linux WebKit CI stalled clicking a tab immediately after a download. Moved download verification to the end of the interaction sequence, preserving every assertion. Native Safari separately verified 12 errors after mislabeling and zero after correction.
