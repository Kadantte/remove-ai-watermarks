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

`detection.adapter_elapsed_ms` measures the wall time of the named detector
adapter call only. Image decoding and fidelity calculation are outside that
interval. The value is `null` when the adapter was unavailable or the artifact
could not be decoded, because no detector call occurred. Preserve the first
call separately when aggregating: it can include lazy imports and model loading,
while subsequent rows describe warm validation cost.

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

## Synthetic cohort v1

`scripts/watermark_benchmark_cohort.py` builds a reproducible local smoke cohort
without a provider API or provenance oracle. Its default 65 cases cover three
deterministic 512-pixel carriers, SDXL, FLUX, and Stable Diffusion 1.x DWT-DCT
patterns, TrustMark Variant P with schema 1, and JPEG quality 90, 75% resize
round-trip, and 5% crop-per-edge round-trip attacks. Each adapter also receives
a high-frequency hard negative. TrustMark cases include the package remover's
output as a separate `removed` state.

```bash
uv run python scripts/watermark_benchmark_cohort.py \
  --output-dir .local-eval/watermark-benchmark-cohort-v1
uv run python scripts/watermark_benchmark.py \
  .local-eval/watermark-benchmark-cohort-v1/manifest.jsonl \
  --output .local-eval/watermark-benchmark-cohort-v1-results.jsonl
```

The builder refuses an existing destination, hashes every generated file, pins
its own recipe digest and dependency versions, and validates the finished
manifest through the benchmark's strict loader. It reads the DWT-DCT patterns
from the production detector so the generated positive cannot silently drift
from the matcher. TrustMark's optional package and model weights must already be
available.

Attacked and removed rows intentionally use `expected: unresolved`: they are
measurements, not assertions smuggled into fixtures. The cohort is too small
and too synthetic for prevalence, confidence, or release-quality claims. It is
a deterministic first baseline that proves the harness and exposes gross
detector behavior before a licensed, stratified real-image cohort is added.

### Development baseline, 2026-09-05

The first local v1 run used cohort recipe
`sha256:d98fd6402bcc149833005ec03aab77bd0074ee105696f9e1ac5da01ef9c666f7`
and kernel
`sha256:37e552ba53c3d2aff2c5f9e160a33ad47b0d32c877b0a8fbba39c90ca0ab0179`.
Dependencies were invisible-watermark 0.2.0, TrustMark 0.9.1, NumPy 1.26.4,
and Pillow 12.3.0. The tree was based on commit `ad53790` with the cohort and
timing changes still uncommitted, so the content hashes, not that commit alone,
identify the executed code. A second build in a different directory produced a
byte-identical manifest and artifact-hash inventory.

| Adapter and state | Unique artifacts | Detected | Not detected |
| --- | ---: | ---: | ---: |
| DWT-DCT clean | 4 | 0 | 4 |
| DWT-DCT marked | 9 | 6 | 3 |
| DWT-DCT attacked | 27 | 13 | 14 |
| TrustMark clean | 4 | 0 | 4 |
| TrustMark marked | 3 | 2 | 1 |
| TrustMark attacked | 9 | 4 | 5 |
| TrustMark removed | 3 | 0 | 3 |

All three plain-gradient DWT-DCT positives were missed, while all texture and
low-detail positives were recognized. DWT-DCT retained 7/9 positives through
JPEG, 6/9 through resize, and 0/9 through crop. TrustMark retained 2/3 through
JPEG and 1/3 through each resize and crop. Its three remover outputs produced no
recognized signal and measured 46.17-49.24 dB PSNR against their clean
carriers. Those three negatives are a post-removal observation, not proof that
the signal is absent beyond this decoder.

Ten independent sequential benchmark processes reproduced every case status.
Nearest-rank adapter timings were:

| Adapter call | n | p50 | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| DWT-DCT cold, first call per process | 10 | 14.96 ms | 19.86 ms | 27.88 ms |
| DWT-DCT warm | 450 | 0.65 ms | 1.64 ms | 13.53 ms |
| TrustMark cold, first call per process | 10 | 1,989.47 ms | 2,085.62 ms | 2,086.02 ms |
| TrustMark warm | 180 | 17.96 ms | 34.20 ms | 39.95 ms |
| TrustMark warm, removed state only | 30 | 16.80 ms | 17.69 ms | 18.56 ms |

These intervals include only the detector adapter call. They exclude image
decoding, fidelity metrics, removal, and process startup. The roughly two-second
TrustMark cold start matters only when validation is the first TrustMark use in
the process; a workflow that already identified the source pays the warm cost.
The sample is deliberately diagnostic rather than statistically representative.

## Aggregate repeated runs

`scripts/watermark_benchmark_report.py` turns one or more result JSONL files
into a standalone Markdown report. Treat each input file as one independent
process run: the first measured call for each adapter in that file is cold, and
the remaining measured calls are warm.

```bash
uv run python scripts/watermark_benchmark_report.py \
  .local-eval/results/run-01.jsonl \
  .local-eval/results/run-02.jsonl \
  --output .local-eval/results/report.md
```

The report retains raw observation counts but deduplicates detector summaries
by artifact SHA-256 within each adapter and state. It reports state and
transform breakdowns, post-removal observations, nearest-rank timing, fidelity,
expected-result mismatches, unstable repeated verdicts, errors, and unavailable
adapters as separate evidence. A repeated `case_id` whose artifact, transform,
or other case identity changes is rejected rather than merged. Input file
digests in the final report bind every aggregate to its exact case-level source.
