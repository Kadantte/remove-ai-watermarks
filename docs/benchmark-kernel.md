# Watermark benchmark kernel

`scripts/watermark_benchmark.py` is a development-only runner for reproducible
image-watermark experiments. It calls the existing local DWT-DCT and TrustMark
detectors through thin adapters. It does not add a runtime command, download a
corpus, or contact a provenance oracle or provider API. The optional TrustMark
package can fetch its official Adobe model weights when its local package cache
is incomplete; prepare that dependency before an offline run.

The kernel answers three different questions and never merges their answers:

- `detection` records whether one named adapter recognized its own signal.
- `removal` records that a removed-state artifact no longer produced a signal,
  or that the signal remained. It always sets `certifies_erasure` to `false`.
- `fidelity` compares decoded pixels with an explicit reference. It says
  nothing about whether an invisible watermark remains.

In particular, `not_detected` means only that the selected detector did not
recognize its signal. It is not a general `clean` verdict.

## Manifest

Input is strict JSONL, one case per line. Relative paths resolve from the
manifest directory. The runner verifies every named file against its lowercase
SHA-256 digest before invoking any detector, rejects duplicate `case_id` values,
and rejects missing or extra fields.

```json
{"schema_version":1,"case_id":"sample-removed","pair_id":"sample","media_type":"image","adapter":"dwt-dct","arm":"positive","state":"removed","path":"removed.png","sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","reference_path":"clean.png","reference_sha256":"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789","source_revision":"local-corpus@2026-09-05","transform":{"name":"remove","revision":"remove-ai-watermarks@abc123","parameters":{"method":"auto"}},"seed":7,"expected":"not_detected"}
```

The fixed fields are:

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `1`. |
| `case_id` | Unique case identity. |
| `pair_id` | Groups related clean, marked, attacked, and removed cases. |
| `media_type` | `image` in schema v1. |
| `adapter` | `dwt-dct` or `trustmark`. |
| `arm` | `positive`, `matched_negative`, `wrong_key`, or `hard_negative`. |
| `state` | `clean`, `marked`, `attacked`, or `removed`. |
| `path`, `sha256` | Artifact path and pinned content digest. |
| `reference_path`, `reference_sha256` | Both strings or both `null`; the fidelity reference. |
| `source_revision` | Corpus, generator, or acquisition revision. |
| `transform` | Exact `name`, `revision`, and JSON `parameters`. |
| `seed` | Integer or `null`; never an implicit random seed. |
| `expected` | `detected`, `not_detected`, or `unresolved`. |

The arms prevent a positive-only benchmark from being mistaken for a detector
evaluation. A useful detector run needs known positives plus matched negatives,
wrong-key cases where the method supports keys, and difficult watermark-free
images. Transformations such as resize, crop, recompression, or removal should
be separate rows with exact parameters, not edits made outside the manifest.

## Run and output

Keep the report outside the repository, and choose a new output path for every
run:

```bash
uv run python scripts/watermark_benchmark.py \
  /path/to/manifest.jsonl \
  --output /path/to/results/run-2026-09-05.jsonl
```

The runner refuses to overwrite an existing report. Every output row repeats
the case identity and records the repository commit, whether tracked files were
dirty, and SHA-256 digests of the benchmark kernel and adapter source. Detector
dependency absence is `unavailable`; an undecodable artifact or adapter
exception is `error`. Neither is counted as `not_detected`.

Fidelity is calculated only when a reference is present, both decoded images
are 8-bit, and their dimensions match. Other bit depths remain explicitly
incomparable until their sample-range contract is defined. The result records
8-bit MAE, the fraction of changed pixel positions, and PSNR. Identical decoded
pixels produce `psnr_db: null` with
`psnr_status: unbounded_identical`, never a misleading numeric zero. Missing or
incomparable references remain explicit states rather than fabricated metrics.

Schema v1 deliberately has no aggregate score, confidence interval, video
adapter, automatic attack generator, removal runner, or provider oracle. Add
those only after the case-level evidence is large enough to define and test the
corresponding statistical or media contract.
