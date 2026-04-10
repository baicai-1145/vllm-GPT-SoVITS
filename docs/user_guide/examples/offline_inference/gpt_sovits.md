# GPT-SoVITS Offline Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/gpt_sovits>.

`end2end.py` runs GPT-SoVITS offline inference through the native runtime and exports audio with the bounded-memory file path. Unlike the old “materialize full audio then `sf.write`” pattern, this route writes fragments directly to the output WAV file and is suitable for very long text export.

## Usage Examples

Inline text:

```bash
python end2end.py \
  --text "Hello, this is a GPT-SoVITS offline synthesis test." \
  --text-lang en
```

Long text from file:

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

Use a local compile cache directory when the default cache path has quota issues:

```bash
python end2end.py \
  --text-file /path/to/long_text.txt \
  --runtime-cache-root /path/to/runtime_cache
```

## Output

By default the example writes:

- audio: `output_audio/gpt_sovits_<timestamp>.wav`
- summary: `output_audio/gpt_sovits_<timestamp>.json`

In batch mode it writes:

- one WAV per matched text file
- one sidecar JSON summary per WAV
- one aggregate batch JSON summary

The summary JSON includes:

- sample rate
- frame count
- output duration
- wall time
- RTF
- runtime cache root

## Common Arguments

| Argument | Description |
|---|---|
| `--text` | Inline text to synthesize |
| `--text-file` | UTF-8 text file to synthesize |
| `--input-dir` | Directory of text files for batch export |
| `--glob` | Glob pattern used with `--input-dir` |
| `--text-lang` | Target text language |
| `--ref-audio` | Reference audio path |
| `--ref-text` | Reference transcript |
| `--prompt-lang` | Reference transcript language |
| `--text-split-method` | GPT-SoVITS segmentation method |
| `--output` | Output WAV path |
| `--output-json` | Output summary JSON path |
| `--output-dir` | Output directory for batch mode and default single export |
| `--output-name-template` | Batch output name template, fields: `{index}`, `{id}`, `{stem}` |
| `--runtime-cache-root` | Cache root for TorchInductor/Triton |

## Notes

- This example is for offline export, not streaming.
- It uses the bounded-memory runtime path intended to avoid holding the full synthesized waveform in Python memory before writing to disk.
- For grouped long-text output, keep `super_sampling=False` unless the runtime gains a bounded-memory super-sampling implementation.
