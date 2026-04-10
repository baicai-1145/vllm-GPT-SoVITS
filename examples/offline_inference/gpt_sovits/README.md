# GPT-SoVITS Offline Inference

This directory contains a minimal offline GPT-SoVITS example that calls the native runtime directly and writes audio through the bounded-memory file path.

## What This Example Does

- Uses `GPTSoVITSRuntime.synthesize_to_file(...)`
- Writes audio fragments directly to the target WAV file
- Avoids accumulating the full synthesized waveform in Python memory before export
- Supports long-text synthesis via `--text-file`

## Quick Start

Use inline text:

```bash
python end2end.py \
  --text "Hello, this is a GPT-SoVITS offline synthesis test." \
  --text-lang en
```

Use a UTF-8 text file:

```bash
python end2end.py \
  --text-file /path/to/long_text.txt \
  --text-lang zh \
  --ref-audio /path/to/reference.wav \
  --ref-text "参考音频对应的文本"
```

Batch export from a directory:

```bash
python end2end.py \
  --input-dir /path/to/text_dir \
  --glob "*.txt" \
  --output-dir /path/to/output_audio \
  --output-name-template "{index:03d}_{stem}"
```

By default the script writes:

- audio: `output_audio/gpt_sovits_<timestamp>.wav`
- summary: `output_audio/gpt_sovits_<timestamp>.json`

In batch mode the script writes:

- one WAV per matched text file
- one sidecar JSON per WAV
- one aggregate batch JSON summary

## Cache Root

If the default Triton or TorchInductor cache path has quota issues, set a repo-local cache root:

```bash
python end2end.py \
  --text-file /path/to/long_text.txt \
  --runtime-cache-root /path/to/runtime_cache
```

This will configure:

- `TORCHINDUCTOR_CACHE_DIR=<runtime_cache_root>/torchinductor`
- `TRITON_CACHE_DIR=<runtime_cache_root>/triton`

## Common Arguments

| Argument | Description |
|---|---|
| `--text` | Inline text to synthesize |
| `--text-file` | UTF-8 text file to synthesize |
| `--input-dir` | Directory of text files for batch export |
| `--glob` | Glob pattern used with `--input-dir` |
| `--text-lang` | Target text language |
| `--ref-audio` | Reference audio path |
| `--ref-text` | Transcript of the reference audio |
| `--prompt-lang` | Language of the reference transcript |
| `--text-split-method` | GPT-SoVITS segmentation method, default `cut5` |
| `--output` | Output WAV path |
| `--output-json` | Output summary JSON path |
| `--output-dir` | Output directory for batch mode and default single export |
| `--output-name-template` | Batch output name template, fields: `{index}`, `{id}`, `{stem}` |
| `--runtime-cache-root` | Optional cache root for Triton/TorchInductor |

## Notes

- This example targets offline file export, not streaming.
- The bounded-memory path currently matches the main long-text production route with `super_sampling=False`.
- If you enable `--super-sampling`, grouped long-text outputs are still constrained by the current runtime limitation documented in the runtime layer.
