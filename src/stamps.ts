export interface Stamp {
  id: string;
  kind: 0 | 1;
  ink: "red" | "blue";
  split: "train" | "test";
  variant: number;
}
export const stamps: Stamp[] = Array.from({ length: 36 }, (_, i) => ({
  id: `post-${String(i + 1).padStart(3, "0")}`,
  kind: (i % 2) as 0 | 1,
  ink: Math.floor(i / 2) % 2 ? "blue" : "red",
  split: i < 24 ? "train" : "test",
  variant: i,
}));
export function drawStamp(canvas: HTMLCanvasElement, s: Stamp) {
  canvas.width = 160;
  canvas.height = 160;
  const c = canvas.getContext("2d")!,
    paper = "#efe8d5",
    ink = s.ink === "red" ? "#a84f4a" : "#426c87";
  c.fillStyle = paper;
  c.fillRect(0, 0, 160, 160);
  c.strokeStyle = ink;
  c.lineWidth = 2;
  c.strokeRect(11, 11, 138, 138);
  for (let i = 0; i < 12; i++) {
    for (const [x, y] of [
      [i * 14, 0],
      [i * 14, 160],
      [0, i * 14],
      [160, i * 14],
    ]) {
      c.fillStyle = "#faf7ef";
      c.beginPath();
      c.arc(x, y, 4, 0, Math.PI * 2);
      c.fill();
    }
  }
  c.font = "8px monospace";
  c.fillStyle = ink;
  c.fillText("POSTE / LASTLIGHT", 20, 27);
  c.fillText(`${String(s.variant + 1).padStart(3, "0")}     05¢`, 20, 139);
  c.save();
  c.translate(80 + ((s.variant % 3) - 1) * 3, 82);
  c.rotate(((s.variant % 5) - 2) * 0.1);
  c.fillStyle = ink;
  c.beginPath();
  c.arc(0, 0, 28 + (s.variant % 3) * 2, 0, Math.PI * 2);
  c.fill();
  if (s.kind === 0) {
    c.strokeStyle = ink;
    c.lineWidth = 4;
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * Math.PI * 2;
      c.beginPath();
      c.moveTo(Math.cos(a) * 36, Math.sin(a) * 36);
      c.lineTo(Math.cos(a) * 46, Math.sin(a) * 46);
      c.stroke();
    }
  } else {
    c.fillStyle = paper;
    c.beginPath();
    c.arc(16, -11, 27, 0, Math.PI * 2);
    c.fill();
    c.fillStyle = ink;
    for (let i = 0; i < 4; i++) {
      c.beginPath();
      c.arc(29 + (i % 2) * 10, -25 + i * 14, 2, 0, Math.PI * 2);
      c.fill();
    }
  }
  c.restore();
  c.globalAlpha = 0.13;
  for (let i = 0; i < 55; i++) {
    c.fillStyle = ink;
    c.fillRect(
      (i * 37 + s.variant * 13) % 160,
      (i * 67 + s.variant * 19) % 160,
      1,
      1,
    );
  }
  c.globalAlpha = 1;
}
