# BoostTrack setup notes (this checkout)

The upstream repo ships `boost-track-env.yml` (conda, py3.8 + torch 2.1 + CUDA 12.1 + lap==0.4.0). This tree converts it to a `uv`/pip flow so it fits alongside the other venvs.

## What differs from upstream

- `requirements-pip.txt` — pip-only equivalent of the conda env's pip section, without `onnx-simplifier`/`onnxoptimizer` (they need `cmake` and aren't needed for tracker inference). `lap` is bumped to `>=0.5` because `lap==0.4.0` no longer builds on modern setuptools.
- Torch is installed separately against the cu121 wheel index to match host CUDA 12.4 driver:
  ```
  uv pip install torch==2.1.0 torchvision==0.16.0 \
      --index-url https://download.pytorch.org/whl/cu121
  ```
- numpy/cython/setuptools must be present **before** `pip install -r requirements-pip.txt` so `lap` and `cython_bbox` can build.

## Known upstream issues worth flagging

From the upstream README: there's a shape-similarity bug where the denominator is wrong. Pass `--s_sim_corr` when running to use the corrected form. The authors keep the buggy version as the default so old hyperparameters reproduce.

## Weights

`external/weights/` holds the Deep OC-SORT weights from the [Google Drive folder](https://drive.google.com/drive/folders/15hZcR4bW_Z9hEaXXjeWhQl_jwRKllauG). These are not committed.
