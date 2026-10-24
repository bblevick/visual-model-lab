import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
if (process.platform !== "darwin" || process.arch !== "arm64")
  throw Error(
    "This local MLX trainer requires an Apple Silicon Mac. The browser classifier and recorded run work on other supported desktop browsers.",
  );
let python = process.env.PYTHON;
for (const candidate of [
  python,
  "python3.13",
  "python3.12",
  "python3.11",
  "/opt/homebrew/opt/python@3.13/bin/python3.13",
  "python3",
].filter(Boolean)) {
  const r = spawnSync(
    candidate,
    ["-c", "import sys; assert (3,11)<=sys.version_info<(3,14)"],
    { stdio: "ignore" },
  );
  if (r.status === 0) {
    python = candidate;
    break;
  }
}
if (!python)
  throw Error("Install Python 3.11–3.13 or set PYTHON to its executable.");
function run(binary, args) {
  const result = spawnSync(binary, args, { stdio: "inherit" });
  if (result.status !== 0) throw Error(`${binary} failed. See output above.`);
}
if (!existsSync(".venv/bin/python")) run(python, ["-m", "venv", ".venv"]);
run(".venv/bin/python", [
  "-m",
  "pip",
  "install",
  "-r",
  "trainer/requirements.txt",
]);
run(".venv/bin/python", ["trainer/setup_model.py"]);
