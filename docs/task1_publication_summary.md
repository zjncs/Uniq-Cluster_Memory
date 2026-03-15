# Task 1 Publication Summary

This note consolidates the current Task 1 evidence chain for the Uniq-Cluster Memory project: controlled evaluation on `Med-LongMem`, module ablations, and real-world validation on official `MedDialog`.

## Scope

Task 1 focuses on patient-level unique memory construction from long medical dialogues:

- extract structured medical events from dialogue
- group repeated mentions into information bundles
- ground relative time expressions
- maintain unique memories with conflict history

The current repository should be presented as a research prototype with quantitative support, not as a finished product system.

## Main Quantitative Results

### M1 event extraction on Med-LongMem

Source: `results/main_results/extraction_eval_med_longmem.json`

- Strict Event-F1: `0.7695`
- Relaxed Event-F1: `0.7742`
- Precision: `0.6410`
- Recall: `0.9839`
- Field F1:
  - `attribute`: `0.9626`
  - `value`: `0.8383`
  - `unit`: `1.0000`
  - `time_scope`: `0.8833`
  - `speaker`: `1.0000`

Interpretation:

- `M1` already captures most events (`high recall`).
- the main remaining weakness is over-extraction and value normalization, not missing the entire fact.

### UCM main Task 1 results on Med-LongMem

Source: `results/main_results/our_method_eval.json`

- Unique-F1 strict: `0.8503`
- Unique-F1 relaxed: `0.8580`
- Conflict-F1: `0.9562`
- Attribute coverage: `0.9929`
- Redundancy: `0.0000`
- Average latency: `40.63s/sample`

Interpretation:

- the core Task 1 objective is already met at prototype level
- the system maintains very high coverage while keeping redundancy at zero

## Ablation Results

Source: `results/ablation/ablation_summary.json`

| Variant | U-F1(S) | U-F1(R) | C-F1 | T-F1 | IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0.8370 | 0.8461 | 0.9662 | 0.7174 | 0.8138 |
| w/o_time | 0.1310 | 0.8740 | 0.2559 | 0.1310 | 0.1348 |
| w/o_conflict | 0.8279 | 0.8439 | 0.0000 | 0.7099 | 0.8044 |
| w/o_m4 | 0.8484 | 0.8561 | 0.9662 | 0.7283 | 0.8144 |
| w/o_m2 | 0.5764 | 0.5835 | 0.0000 | 0.4852 | 0.6370 |

Key conclusions:

1. `Time grounding` is a primary contributor.
   Removing it collapses strict uniqueness and temporal correctness:
   `U-F1(S) 0.8370 -> 0.1310`, `C-F1 0.9662 -> 0.2559`.

2. `M2 / information bundling` is the second major contributor.
   Removing clustering degrades both unique memory quality and conflict handling:
   `U-F1(S) 0.8370 -> 0.5764`, `C-F1 0.9662 -> 0.0000`.

3. `Conflict detection` is a specialized but necessary module.
   Removing it leaves strict uniqueness mostly intact but destroys conflict tracking:
   `C-F1 0.9662 -> 0.0000`.

4. `M4 compression` is not yet a strong contribution.
   In the current implementation it does not outperform the no-M4 variant.
   For paper writing, `M4` should be framed as an auxiliary post-processing layer, not a primary novelty claim.

## Real-World Validation on Official MedDialog

Source directory:
`results/real_world_validation/meddialog_official_zh_test_long_r20_s42_ref2024`

Protocol:

- official Chinese `MedDialog`
- `20` long dialogues
- fixed `reference_date=2024-01-01` for reproducible relative-time grounding
- output includes `silver_gt.jsonl`, `audit_subset.csv`, `case_studies.md`, and `summary.json`

Observed results:

- completed dialogues: `20/20`
- failed dialogues: `0`
- mean turns: `12.5`
- mean predicted memories: `13.25`
- mean conflicts: `1.35`
- dialogues with conflicts: `15/20`

Latency observations from `silver_gt.jsonl`:

- mean latency: `168.26s`
- median latency: `46.71s`
- max latency: `2136.41s`
- samples over `120s`: `4`

Interpretation:

- the pipeline is usable on real dialogues and does not fail systematically
- real-world output is rich enough to support case studies and audit-based validation
- latency has a heavy tail because `M1` window extraction may hit retry-heavy JSON failures on long or noisy dialogues

This real-world package should be used for:

- case studies
- audit-based factual inspection
- external validity claims

It should not be used as the same-version headline quantitative benchmark because the set does not contain human gold labels for unique memories and conflicts.

## Paper Positioning

At the current stage, the strongest paper narrative is:

`Long medical dialogues need patient-level unique memory construction rather than raw retrieval or plain summarization. Information bundling plus time grounding is the central mechanism; conflict tracking is a necessary complementary capability.`

Recommended emphasis order:

1. M2.5/M2 information bundling
2. M3 time grounding and unique memory policies
3. explicit conflict history
4. real-world MedDialog validation
5. M4 as a supporting optimization only

## What Is Still Missing for a Stronger Submission

The project is already strong enough for:

- an internal research milestone
- a lab report
- a workshop-style submission draft
- a solid pre-submission prototype

The main remaining gaps for a stronger paper are:

1. stronger real-world evaluation beyond silver-only evidence
2. a cleaned latency story for long noisy dialogues
3. clearer bad-case categorization for `M1` over-extraction and time normalization
4. optionally, one additional external benchmark or downstream task for transfer evidence

## Practical Recommendation

If time is limited, the best publication-oriented packaging is:

1. use `Med-LongMem` as the main benchmark with the current quantitative results
2. use the 20-sample `MedDialog` package as real-world qualitative validation
3. explicitly state that gold-labeled real-world unique-memory evaluation is future work

This keeps the project honest, technically coherent, and defensible.
