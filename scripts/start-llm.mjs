import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
const python =
  process.platform === "win32"
    ? ".venv/Scripts/python.exe"
    : ".venv/bin/python";
if (!existsSync(python))
  throw Error(
    "Run npm run llm:setup first. Live training requires Apple Silicon.",
  );
const children = [
  spawn(python, ["trainer/server.py"], { stdio: "inherit" }),
  spawn(
    process.execPath,
    [
      "node_modules/vite/bin/vite.js",
      "--host",
      "127.0.0.1",
      "--port",
      "4197",
      "--strictPort",
    ],
    { stdio: "inherit" },
  ),
];
let closing = false;
function close() {
  if (closing) return;
  closing = true;
  for (const child of children) child.kill("SIGTERM");
}
process.on("SIGINT", close);
process.on("SIGTERM", close);
for (const child of children) child.on("exit", () => close());
