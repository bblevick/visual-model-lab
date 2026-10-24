import * as tf from "@tensorflow/tfjs";
import { drawStamp, type Stamp } from "./stamps";
import { normalize } from "./classifier";
let model: tf.LayersModel | null = null;
export async function embed(stamp: Stamp) {
  if (!model) {
    await tf.ready();
    const base = await tf.loadLayersModel(
      import.meta.env.BASE_URL + "model/model.json",
    );
    model = tf.model({
      inputs: base.inputs,
      outputs: base.getLayer("global_average_pooling2d_1").output,
    });
  }
  const canvas = document.createElement("canvas");
  drawStamp(canvas, stamp);
  const tensor = tf.tidy(() => {
    const pixels = tf.browser.fromPixels(canvas).toFloat();
    const input = tf.image
      .resizeBilinear(pixels, [224, 224])
      .div(127.5)
      .sub(1)
      .expandDims(0);
    return model!.predict(input) as tf.Tensor;
  });
  const values = Array.from(await tensor.data());
  tensor.dispose();
  await tf.nextFrame();
  return normalize(values);
}
export function backend() {
  return tf.getBackend();
}
