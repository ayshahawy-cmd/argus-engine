# Evaluation: earn the improvement claim

The hypothesis is that selective vision-model review can reduce fragmented
track counts and add useful scene context. The initial release demonstrates
the mechanism and tests its software behavior. It does not establish that the
AI improves accuracy on real footage.

## A useful first benchmark

1. Collect same-camera clips you have permission to analyze and redistribute,
   or keep the footage private and publish only aggregate results.
2. Annotate unique visits, track breaks, occlusions, and true track pair matches.
   Label ambiguity explicitly. Keep evaluation clips separate from prompt tuning.
3. Run the exact same detector weights, tracker settings, frames, and thresholds
   with AI disabled and enabled.
4. Report count error, pair precision/recall, false-merge rate, missed matches,
   requests, tokens, capture time, and review latency. Add IDF1 / HOTA when
   ground-truth trajectories are available.
5. Break results down by crowding, uniforms, lighting, occlusion, and camera
   position. Repeat with and without the color prefilter and head redaction.
6. Include failed requests, malformed answers, and `UNSURE` decisions in the
   report. Publish provider, model version, prompts, settings, and sample size.

Track stitching can lower a count while making it less accurate. A lower number
of tracks alone is not evidence of improvement. Scene summaries need their own
timestamp-grounded human review for correctness and unsupported claims.

The synthetic demo and mocked tests are engineering checks, not benchmark data.
