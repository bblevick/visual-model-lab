# Models, code and data notices

Original application code, synthetic stamp illustration generator and synthetic dispatch examples are MIT licensed (LICENSE).

## Frozen vision backbone

TensorFlow.js is Apache-2.0 licensed; its license is retained in docs/licenses/tensorflow-LICENSE.txt. The MobileNet V1 0.25 224 weights and manifest are preserved from Google's TensorFlow.js model distribution: https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/ . Model source, preprocessing, embedding layer and every file SHA-256 are in public/model-provenance.json. Original MobileNet work: https://arxiv.org/abs/1704.04861 . The repository does not imply Google endorsement.

The 36 stamp illustrations are original procedural assets. Their actual image embeddings are preserved with extraction backend and timestamp in public/embeddings.json. Held-out truth describes the generated shape, not an external image-labeling dataset.

## Language model and training runtime

The base model is HuggingFaceTB/SmolLM2-135M-Instruct, Apache-2.0 according to its official model card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct . Revision is pinned to 12fd25f77366fa6b3b4b768ec3050bf629380bac. Base weights are downloaded locally and not redistributed here. The intentionally published example adapter is Apache-2.0 licensed and includes its license in artifacts/example-adapter/LICENSE. Its parent model and training data are identified in public/training-run.json.

MLX-LM is MIT licensed; upstream license retained in docs/licenses/mlx-lm-LICENSE.txt. MLX and other Python packages retain their own licenses. Exact local training dependencies are pinned in trainer/requirements.txt. Model setup/training make no paid service calls; initial downloads use official public model/package hosts.

The public report records actual synthetic-task measurements, original source/adapter hashes, and generated before/after outputs. It is not a third-party quality certification. Imported examples and local subsequent runs remain on the user's machine and are ignored by Git.
