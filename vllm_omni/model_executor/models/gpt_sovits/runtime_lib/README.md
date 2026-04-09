# GPT-SoVITS Native Runtime Lib

This directory is the GPT-SoVITS inference runtime vendored directly into
`vllm_omni.model_executor.models.gpt_sovits`.

Kept for runtime:

- `GPT_SoVITS/AR`
- `GPT_SoVITS/BigVGAN`
- `GPT_SoVITS/TTS_infer_pack`
- `GPT_SoVITS/configs`
- `GPT_SoVITS/eres2net`
- `GPT_SoVITS/f5_tts`
- `GPT_SoVITS/feature_extractor`
- `GPT_SoVITS/module`
- `GPT_SoVITS/text`
- `GPT_SoVITS/process_ckpt.py`
- `GPT_SoVITS/sv.py`
- `GPT_SoVITS/utils.py`
- `runtime_preload.py`
- `tools/audio_sr.py`
- `tools/AP_BWE_main`
- `tools/i18n`

Linked instead of copied:

- `GPT_SoVITS/pretrained_models`
- `GPT_SoVITS/text/G2PWModel`
- `GPT_SoVITS/text/ja_userdic`

Git submodules:

- `third_party/g2pw-cu`
- `third_party/split-lang`

These local development links must not be committed to Git. Recreate them on a
developer machine with:

```bash
bash tools/dev/setup_gptsovits_runtime_links.sh
```

Only the three GPT-SoVITS asset paths above are local development links. The
two `third_party` repositories are real Git submodules of this repository and
should be initialized with:

```bash
git submodule update --init --recursive -- \
  vllm_omni/model_executor/models/gpt_sovits/runtime_lib/third_party/g2pw-cu \
  vllm_omni/model_executor/models/gpt_sovits/runtime_lib/third_party/split-lang
```

Override the source GPT-SoVITS checkout with `GPT_SOVITS_LINK_SOURCE_ROOT=/abs/path`
when it is not located at `/root/GPT-SoVITS`.

Removed from the native runtime copy:

- WebUI / API entry scripts
- training scripts
- dataset preparation scripts
- ONNX / TorchScript export scripts
- legacy `unified_engine*.py` scheduler stack that is not part of the current vLLM-native runtime path
- duplicated vendored root under `/root/vllm-omni/third_party/GPT-SoVITS`
- obvious non-runtime BigVGAN files such as `train.py`, `tests/`, `README.md`
