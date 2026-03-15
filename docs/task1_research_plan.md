# Task 1 Research Plan: Memory Bundle Construction

## Goal

Task 1 targets patient-level memory construction rather than downstream QA.

Input:
- multi-turn medical dialogue

Output:
- `BundleGraph`: event/entity bundles for analysis and version tracking
- `CanonicalMemory`: unique, time-aware, conflict-aware patient memory units

Core requirement:
- convert repeated, scattered, time-ambiguous medical facts into structured memory that is unique, traceable, and evaluable

## Method

The project follows a four-stage route:

1. `M1 Event Extraction`
   - extract `attribute/value/unit/time_expr/provenance/speaker`
   - keep raw time expression for later grounding

2. `M2 + M2.5 Bundle Construction`
   - normalize attributes
   - build medication entity bundles
   - build event bundles around shared time anchors

3. `M3 Uniqueness Management`
   - ground relative/implicit time expressions
   - apply policy-aware merge rules:
     - `unique` for measurements/diagnoses
     - `latest` for medications
     - `append` for symptoms
   - preserve conflict history instead of overwriting earlier values

4. `M4 / M5 Auxiliary Validation`
   - compress memory for storage
   - validate usefulness via retrieval and temporal reasoning

## Primary Metrics

Task 1 should be judged primarily by:

- `Unique-F1 (strict)`
- `Unique-F1 (relaxed)`
- `Conflict-F1`
- `Attribute Coverage`
- `Redundancy`

`M1` also needs its own extraction metrics:

- Event-F1 (strict)
- Event-F1 (relaxed)
- field-level F1 for `attribute/value/unit/time_scope/speaker`

## Dataset Strategy

Main benchmark:
- `Med-LongMem`

Supporting evidence:
- real MedDialog audit set for manual inspection
- LongMemEval or similar long-context benchmark for auxiliary validation

Recommended evidence chain:

1. synthetic benchmark with gold labels
2. ablation study for each module
3. small real-data manual audit

## Completion Criteria

Task 1 should be considered complete only when all of the following are true:

1. `dialogue -> BundleGraph -> CanonicalMemory` is stable and reproducible
2. M1 extraction is evaluated independently
3. Med-LongMem results are generated with a consistent pipeline configuration
4. ablation results match the same evaluation setup
5. a small real-data audit package exists for qualitative validation

## Current Priority Gaps

The highest-priority remaining gaps are:

1. maintain one consistent Med-LongMem evaluation configuration across scripts
2. keep M1 extraction evaluation as a first-class experiment
3. expand beyond the current small synthetic release when resources allow
4. add stronger real-world gold or audited evidence instead of relying only on silver labels
