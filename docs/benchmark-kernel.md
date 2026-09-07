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
  --output-dir .local-eval/watermark-benchmark-cohort-v1 \
  --size 512
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

`--size` accepts square carrier sides of at least 256 pixels, divisible by 16.
The size is part of every case identity and transform record, so resource sweeps
at multiple boundaries can be aggregated without merging unlike artifacts.

Attacked and removed rows intentionally use `expected: unresolved`: they are
measurements, not assertions smuggled into fixtures. The cohort is too small
and too synthetic for prevalence, confidence, or release-quality claims. It is
a deterministic first baseline that proves the harness and exposes gross
detector behavior before the real-image cohort below is run.

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

## Real-image cohort v1

`scripts/watermark_benchmark_real_cohort.py` replaces procedural carriers with
the 38 committed OpenAI/Meta images in the engine-selection content matrix. The
matrix has 19 prompt-matched content strata, one image per provider in every
stratum, and no duplicate file hash. Its images are project-collected model
outputs rather than camera originals; "real-image" here means real, decoded
content carriers rather than generated gradients or checkerboards.

Every source row carries
`reuse_basis: project-maintainer-public-test-clearance`. This records permanent
public test and benchmark permission for those exact committed bytes. It does
not claim that provider terms grant a general dataset license for arbitrary
generations. The builder refuses a source without that explicit basis, a hash
or dimension mismatch, repeated pixels, an incomplete provider pair, or a
duplicate provider/content stratum.

```bash
uv run python scripts/watermark_benchmark_real_cohort.py \
  --output-dir .local-eval/watermark-benchmark-real-v1 \
  --size 512
uv run python scripts/watermark_benchmark.py \
  .local-eval/watermark-benchmark-real-v1/manifest.jsonl \
  --output .local-eval/watermark-benchmark-real-v1-results.jsonl
```

Each source is deterministically center-cropped only if necessary, resized to
the selected square geometry with LANCZOS, converted to RGB, and written as a
PNG reference. At the default size all current sources are already square, so
only the downscale and RGB conversion apply. The source manifest hash, original
file and payload hashes, provider, model, content stratum, native dimensions,
reuse basis, prompt hash, and standardization recipe travel with every case.

The default build has 798 cases: all three production DWT-DCT patterns plus
TrustMark Variant P, each with matched negatives, marked positives, the same
three fixed attacks as the synthetic cohort, and TrustMark remover outputs.
Clean sources are expected negative only for the selected DWT-DCT or TrustMark
adapter; the result must not be generalized into a claim that no other
provenance signal exists. The aggregate report adds provider and content-stratum
tables when this source metadata is present.

The 12 existing SynthID and Content Seal signal carriers are deliberately not
mixed into this matched-negative baseline. They remain suitable for a separate
cross-watermark interference layer, where their original signal and oracle
provenance can be named instead of being mislabeled as globally clean.

### Development baseline, 2026-09-06

Three sequential local runs reproduced all 798 case verdicts with no unstable
case, adapter error, or unavailable result. Both adapters rejected all 38
matched negatives. The carrier set therefore removed the synthetic cohort's
false impression that TrustMark was broadly carrier-fragile, while confirming
and sharpening the DWT-DCT carrier-dependence finding.

| Adapter and state | Unique artifacts | Detected | Not detected |
| --- | ---: | ---: | ---: |
| DWT-DCT marked | 114 | 74 | 40 |
| DWT-DCT JPEG q90 | 114 | 76 | 38 |
| DWT-DCT resize 75% round-trip | 114 | 77 | 37 |
| DWT-DCT crop 5% per edge | 114 | 0 | 114 |
| TrustMark marked | 38 | 38 | 0 |
| TrustMark, all three attacked arms | 114 | 114 | 0 |
| TrustMark removed | 38 | 0 | 38 |

DWT-DCT marked recall was 26/38 for FLUX, 23/38 for Stable Diffusion
1.x, and 25/38 for SDXL. JPEG and resize retained every one of the 74 base
detections, then moved two and three previously missed cases above the decoder
threshold respectively. They did not create new ground-truth marks; this is why
the report presents pairwise transitions rather than describing 76/114 and
77/114 as ordinary attack retention. The crop destroyed every recognized
DWT-DCT signal.

The content strata expose a strong dependence that three procedural carriers
could not measure. Interior low light, person in context, and single portrait
were 0/6 on the un-attacked DWT-DCT embeds, while nine strata were 6/6. Meta
carriers were detected in 42/57 marked cases and their prompt-matched OpenAI
partners in 32/57. Those are descriptive counts over one image per provider
and stratum, not a provider-population estimate.

TrustMark was detected before and after every fixed attack in every stratum.
All 38 remover outputs produced no recognized TrustMark signal, with 41.26 to
49.80 dB PSNR and 45.27 dB median PSNR against the standardized carriers. This
remains a decoder-scoped post-removal observation, not proof of erasure against
another implementation.

The run was based on commit `f61a7ec` with tracked benchmark changes still
uncommitted. The exact inputs and implementations are therefore bound by hash:

- source manifest: `sha256:aa37aa97ed3b32a59242fd506fa3cee545df615f98690d373735929a9d7d0089`;
- content-manifest validator: `sha256:72ddb2e489d5a7f92cd948b880a0fd308d4083ed86d68fc49897d646aceb7ec5`;
- real-cohort recipe: `sha256:f3851dae5d4eee69510c3dc5b5b15b5116720d944cfeed631894e635c2a5e9c9`;
- shared embed/attack helper: `sha256:c78342f0d125d6636259b641e67e71b72b577455057e568052d2a0a4df00c9dd`;
- benchmark kernel: `sha256:a96874a41fa0a38cd29d957c6a69683de887e942b480ce7668090b0f1a9e3329`;
- generated manifest: `sha256:77dc6094fcfbbaa6fdb9a15d96dee6fb3d4349829b2b1d7cf5f006aa7fb0ae8a`;
- case-level results: `sha256:bb483be015c012fd28d0979510244c834475a23bbc215226c998fada73485ed3`,
  `sha256:35d735de9654a36d6d2a6e096f7a2c3f8a70b1b22ead4f6f414d6c1494e53b2a`, and
  `sha256:52a9a2bb78dcc9f40609b5ca13812470d21477e5e553892c2bbef4b803fee433`;
- aggregate report: `sha256:4496edf231a20cc7977f003c30313af5acbe8c99075e22d2cb2eeec5e104f3eb`.

The local timing rows were host-contended and are not used for a new latency
claim. The controlled resource profile below remains the performance evidence.

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
The runner decodes an artifact used by multiple cases only once per process and
passes the decoded pixels to both local adapters; the adapter timing therefore
does not include file decoding.

## Profile process cost

`scripts/watermark_benchmark_resources.py` runs each manifest in fresh Python
processes. It records parent-observed wall time and the child's absolute peak
RSS, while preserving the ordinary case-level JSONL outputs for the detector
report above.

```bash
uv run python scripts/watermark_benchmark_resources.py \
  .local-eval/cohort-256/manifest.jsonl \
  .local-eval/cohort-512/manifest.jsonl \
  .local-eval/cohort-1024/manifest.jsonl \
  --repeat 3 \
  --output-dir .local-eval/resource-profile
```

Wall time includes interpreter startup, imports, image decoding, detector calls,
fidelity metrics, and result writing. Peak RSS is the process's absolute
high-water mark, not incremental memory caused only by validation. Run DWT-DCT
and TrustMark manifests separately to avoid attributing one adapter's model load
to the other. The profiler uses only local adapters and never calls an oracle or
provider API.

### Development resource profile, 2026-09-06

Ten sequential fresh-process runs per size were measured on macOS arm64 with
Python 3.12.13. DWT-DCT manifests contained 46 cases; TrustMark manifests
contained 19. Every status count was stable across all ten repetitions.

| Adapter | Max geometry | Cases | Process wall p50 | p90 | Absolute peak RSS p50 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DWT-DCT | 256x256 | 46 | 263.07 ms | 322.61 ms | 74.66 MiB | 75.31 MiB |
| DWT-DCT | 512x512 | 46 | 482.67 ms | 498.07 ms | 90.56 MiB | 91.02 MiB |
| DWT-DCT | 1024x1024 | 46 | 1,360.10 ms | 1,389.40 ms | 159.58 MiB | 160.00 MiB |
| TrustMark | 256x256 | 19 | 3,191.18 ms | 6,074.68 ms | 741.11 MiB | 743.66 MiB |
| TrustMark | 512x512 | 19 | 9,172.26 ms | 15,430.20 ms | 752.92 MiB | 758.17 MiB |
| TrustMark | 1024x1024 | 19 | 7,301.26 ms | 9,797.40 ms | 848.91 MiB | 852.92 MiB |

The TrustMark wall-time variance, including one 18.34-second 512 px maximum,
did not coincide with comparable memory movement. The non-monotonic 512/1024
times prove that this host was contended, so the TrustMark wall rows are a
diagnostic range and cannot establish size scaling or throughput. Repeat them
on an idle host before making a performance claim. Peak RSS stayed much more
stable: its nearly flat 256/512 values establish the fixed model cost, while
1024 px adds about 100 MiB of high-water memory.

These whole-process rows are deliberately not divided by the case count. A
fresh-process average would hide the fixed model load and is not the cost of one
post-removal check. The current three-run real-image cohort measured warm p50
adapter calls of 0.57 ms for DWT-DCT and 29.11 ms for TrustMark at 512 px,
excluding file decoding. A workflow that identified the source before removal
already holds TrustMark's singleton decoder, so validation reuses that fixed
model cost. Starting a new process or loading TrustMark only after removal is
the expensive design and should be avoided.

The exact profiler source was
`sha256:caf9674dffceddebb8cbf26067581250334fbc11e0fbcb567e8dd45555357b54`.
The local DWT-DCT and TrustMark resource JSONL files were respectively
`sha256:b7ceb012db5c2f549720c37d6234526396c7120fb68e040705322be35dfd6743`
and
`sha256:f3639f46a9257a3161ed0197016c089ca5c084b2de2e07cb4b65fc774e20544d`.
They remain untracked because they contain host-specific paths and timings;
these hashes bind the summarized numbers to the exact local evidence.
