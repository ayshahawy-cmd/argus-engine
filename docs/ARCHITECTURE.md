# Architecture

The engine was extracted and adapted from the original ARGUS / OpnReality
Python tracker. It preserves the central sequence: local person tracking,
largest eligible crop per track, time-and-distance candidate selection,
color filtering, visual pair review, track-root resolution, periodic scene
review, and a summary grounded in the results.

The public package separates that sequence from product-specific retail
dashboards, customer configuration, cloud databases, evidence storage services,
and operational reporting workflows. Its output contract is local JSON and
selected JPEG evidence.

| Module | Responsibility |
| --- | --- |
| `types.py` | Detection, frame, provider, and result contracts |
| `detectors.py` | Optional Ultralytics tracker and OpenCV capture |
| `images.py` | Bounds checks, crop redaction, JPEGs, color similarity |
| `engine.py` | Capture, track filtering, candidate review, merges, results |
| `providers.py` | Optional Anthropic and compatible API adapters |
| `prompts.py` | Replaceable visual review and scene prompts |
| `config.py` | Public thresholds and budgets; no credentials |
| `demo.py` | Generated frames with explicitly scripted responses |

## Frame contract

`Engine.run()` consumes `Iterable[FramePacket]`. Timestamps are non-negative,
finite, and monotonically non-decreasing. Images are PIL images; bounding boxes
are `(x1, y1, x2, y2)` in original-frame pixels. Track IDs are non-negative
integers, stable within one run. Reused IDs can corrupt continuity estimates;
start a new engine run when a tracker resets. The built-in adapter uses person
class `0` from a COCO-style model.

The adapter processes every `stride`th decoded frame with a persistent tracker.
Large strides make tracking harder. For variable-frame-rate videos, reported
FPS is an approximation; supply timestamped packets for precise timing.
Streams use elapsed capture time, not camera-provided presentation timestamps.
The final duration is the last processed packet's timestamp.

## Decisions and merges

Only non-overlapping, nearby tracks with retained crops are candidates.
Candidates are sorted by gap and pixel distance, independent of ID numbering.
RGB histogram correlation is a cost-saving heuristic and can reject true matches.
Only an exact first-line `SAME` applies a merge. Empty, malformed, uncertain,
different, or failed answers keep tracks separate. Track-root checks prevent
cycles; successor constraints prevent one old track branching into two new
tracks. Merged groups cannot contain overlapping observed intervals.

This greedy policy does not solve global assignment or calibrate model confidence.
A false early merge can affect subsequent decisions. Every considered pair's
verdict or skip reason appears in `result.json` for inspection.

## Processing and resource limits

Capture completes before provider calls start. Crop files are overwritten only
when a larger eligible view of that track appears. Scenes are sampled at bounded
intervals and capped per run. A run's output directory is created with a UUID;
evidence is never shared across runs.

The engine retains a record and at most one crop for every raw track, including
short tracks later filtered from results. Candidate search is quadratic in kept
track count. Use bounded recording windows for long videos or busy streams.
`max_reid_calls` bounds provider comparisons, not candidate-search CPU time or
the number of raw crops on disk. Capture errors can leave partial local evidence
without a completed result file.

## Provider contract

`complete(prompt, images=(), *, max_tokens=300)` returns `VisionResponse`.
Images are local JPEG paths. The same method handles two-image track review,
one-image scene descriptions, and a text-only summary. Token counts are optional
and default to zero. Custom providers can execute locally or remotely.

Built-in adapters read SDK credentials from the environment and disable retries.
Failures are recorded without copying raw exception payloads into logs. An API
failure leaves track counts unchanged for that comparison. Built-in request
shapes have mock tests; live model availability and quality need separate
verification with the user's endpoint.
