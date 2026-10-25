# Publishing

Expected repository: https://github.com/bblevick/visual-model-lab
Expected Pages demo: https://bblevick.github.io/visual-model-lab/

The workflow verifies code and builds with BASE_PATH=/visual-model-lab/ before deploying through GitHub Actions Pages. Enable GitHub Actions as the repository Pages source. Local preview: `BASE_PATH=/visual-model-lab/ npm run build && BASE_PATH=/visual-model-lab/ npm run preview` and visit the repository path at the preview origin.

Only the static application and intended public example assets are deployed. Git ignores dependency directories, local datasets/runs, base LLM weights, virtual environments and raw captures. Profile pins are a separate UI action.

Published October 24, 2026. [First successful verification and deployment](https://github.com/bblevick/visual-model-lab/actions/runs/40144799425). Public browser interactions and real frozen-backbone recomputation passed without page errors, failed requests or external application requests; see `public-release-check.json`. The repository is pinned on the profile. Public hosting displays an actual recorded LoRA run; fresh adapter training runs locally on Apple Silicon.
