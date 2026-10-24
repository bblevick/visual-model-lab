import { probability, projection, type LinearModel } from "./classifier";
import { stamps } from "./stamps";
export function plot(
  canvas: HTMLCanvasElement,
  features: number[][],
  model: LinearModel | null,
) {
  canvas.width = 1000;
  canvas.height = 610;
  const ctx = canvas.getContext("2d")!,
    W = 1000,
    H = 610;
  ctx.fillStyle = "#f0e8d8";
  ctx.fillRect(0, 0, W, H);
  const p = projection(features),
    r = p.range;
  const point = (x: number, y: number) => ({
    x: 70 + (x / r + 1) * 430,
    y: 530 - (y / r + 1) * 225,
  });
  if (model)
    for (let x = 0; x < 860; x += 14)
      for (let y = 0; y < 450; y += 14) {
        const u = ((x / 860) * 2 - 1) * r,
          v = (1 - (y / 450) * 2) * r,
          vector = p.mean.map((m, i) => m + p.u[i] * u + p.v[i] * v),
          pr = probability(model, vector);
        ctx.fillStyle =
          pr >= 0.5
            ? `rgba(76,104,143,${0.08 + Math.abs(pr - 0.5) * 0.3})`
            : `rgba(187,111,61,${0.08 + Math.abs(pr - 0.5) * 0.3})`;
        ctx.fillRect(70 + x, 80 + y, 14, 14);
      }
  ctx.strokeStyle = "#c2bcae";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const x = 70 + i * 215,
      y = 80 + i * 112.5;
    ctx.beginPath();
    ctx.moveTo(x, 80);
    ctx.lineTo(x, 530);
    ctx.moveTo(70, y);
    ctx.lineTo(930, y);
    ctx.stroke();
  }
  stamps.forEach((s, i) => {
    const pt = point(p.points[i][0], p.points[i][1]),
      pred = model ? Number(probability(model, features[i]) >= 0.5) : null;
    ctx.fillStyle = s.kind === 0 ? "#bd703e" : "#4d688e";
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, s.split === "test" ? 8 : 5, 0, Math.PI * 2);
    ctx.fill();
    if (s.split === "test") {
      ctx.strokeStyle = model && pred !== s.kind ? "#a3373c" : "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 12, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = "#394a45";
    ctx.font = "8px monospace";
    ctx.fillText(s.id.slice(-3), pt.x + 13, pt.y + 3);
  });
  ctx.fillStyle = "#41514b";
  ctx.font = "11px monospace";
  ctx.fillText("FROZEN IMAGE FEATURES / PCA PROJECTION", 35, 35);
  ctx.fillText(
    "SMALL DOT: TRAINING STAMP  /  LARGE RING: HELD OUT  /  RED RING: MISTAKE",
    35,
    581,
  );
  ctx.font = "10px monospace";
  ctx.fillText(
    "Background is a 2D slice through the full classifier; scores use all embedding dimensions.",
    35,
    601,
  );
}
