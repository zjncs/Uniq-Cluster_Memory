# Med-LongMem v1: Design Document

**Version**: 1.0
**Date**: 2026-02-28
**Author**: Manus AI

## 1. Introduction

**Med-LongMem** is a synthetic, adversarial benchmark dataset designed to evaluate the long-term memory capabilities of Large Language Models (LLMs) in the clinical domain. Unlike traditional QA datasets, Med-LongMem focuses on a model's ability to maintain a **unique**, **conflict-free**, and **temporally consistent** knowledge base from conversational data.

The generation of this dataset is guided by a **Ground-Truth-First** principle. We first programmatically construct a timeline of structured medical events with controlled adversarial challenges, and only then use an LLM to render this timeline into a naturalistic patient-doctor dialogue. This ensures the dataset is both challenging and precisely evaluable against our core metrics: **Unique-F1** and **Conflict Detection F1**.

## 2. Generation Pipeline

The generation process follows four distinct steps:

1.  **Stochastic Event Generation**: For each sample, we generate a timeline of `RawEvent` objects based on predefined statistical distributions. This ensures a controlled variety of medical scenarios.

2.  **Adversarial Injection**: We programmatically inject specific, evaluable challenges into the event timeline. This includes creating value conflicts, inserting long-span coreferences, and scheduling entity updates.

3.  **Ground Truth Derivation**: From the final, adversarially-modified event timeline, we derive the three layers of ground truth data: `raw_events.jsonl`, `canonical_gt.jsonl`, and `conflict_gt.jsonl`.

4.  **Dialogue Surfacing**: The complete event timeline is provided to a powerful generator LLM (e.g., GPT-4-Turbo) with a specific prompt to "surface" these structured events into a coherent, multi-turn patient-doctor dialogue. The LLM acts as a rendering engine, not a source of truth.

## 3. Ground Truth Schema

Each sample in the Med-LongMem dataset, identified by a `dialogue_id`, will contain three ground truth files:

1.  **`raw_events.jsonl`**: A complete log of all medical events as they appear chronologically. This represents the unprocessed, raw information stream.
    *   **Schema**: `RawEvent(event_id, turn_id, attribute, value, unit, time_scope, adversarial_tag, coref_target)`

2.  **`canonical_gt.jsonl`**: The final, deduplicated, and conflict-aware state of the patient's record, represented as a list of `CanonicalMemory` objects. This is the primary target for our model to reconstruct and serves as the ground truth for the **Unique-F1** metric.

3.  **`conflict_gt.jsonl`**: An explicit list of all `CanonicalMemory` objects from the GT that have their `conflict_flag` set to `True`. This serves as the direct ground truth for the **Conflict Detection F1** metric.

## 4. Adversarial Injection & Difficulty Grading

To ensure the benchmark is challenging and allows for robustness analysis, we control the distribution of adversarial events and classify each sample into a difficulty tier.

### Statistical Controls

| Parameter                       | Distribution / Value          | Purpose                                      |
| :------------------------------ | :---------------------------- | :------------------------------------------- |
| `events_per_dialogue`           | `Uniform(5, 12)`              | Controls the information density of a dialogue.  |
| `conflict_probability`          | `0.3`                         | The probability that an event is a conflict.     |
| `long_coref_probability`        | `0.4`                         | The probability of a long-span coreference.    |
| `update_event_probability`      | `0.2`                         | The probability of a `latest` policy update.   |
| `coref_span_distribution`       | `Geom(p=0.1)`                 | Controls the turn distance for coreferences.   |

### Difficulty Levels

| Level  | `conflict_events` | `coref_span` (turns) | `update_events` | Description                                      |
| :----- | :---------------- | :------------------- | :-------------- | :----------------------------------------------- |
| **Easy**   | 0                 | `≤ 5`                | 0               | Basic information retention and deduplication.   |
| **Medium** | 1                 | `5-15`               | 0-1             | Requires handling a single conflict and mid-range coreference. |
| **Hard**   | `≥ 2`             | `≥ 15`               | `≥ 1`             | Involves multiple conflicts, long-range reasoning, and entity updates. |

## 5. Medical Plausibility

To prevent the dataset from being toy-like, all generated numerical values for clinical attributes (e.g., vital signs, lab results) will be constrained within medically plausible ranges. Conflicts will be generated as variations *within* these reasonable boundaries (e.g., a blood glucose value changing from `6.5 mmol/L` to `8.0 mmol/L`, not to `50 mmol/L`). This ensures the model is tested on realistic clinical fluctuations.

## 6. Scale & Scope

### v0.1 (Validation Release)

Before generating the full dataset, we first produce a small **v0.1 validation release** to verify that the generation pipeline, GT derivation logic, and evaluation metrics are correctly aligned.

*   **Total Samples**: **20** dialogues.
*   **Dialogue Length**: **20** turns per dialogue.
*   **Difficulty**: **Hard only** (≥ 2 conflicts, coref span ≥ 15 turns, ≥ 1 update event).
*   **Language**: **English** (with architecture designed to extend to Chinese).
*   **Coref Span Distribution**: `Geom(p=0.08)` — heavier tail than default, producing more long-range coreferences.
*   **Validation Goal**: Confirm that our method yields a meaningfully higher Unique-F1 and Conflict-F1 than the Raw-RAG baseline on adversarial samples. If no clear advantage is observed, the generation logic must be revised before scaling up.

### v1.0 (Full Release)

The full version (v1) of the Med-LongMem dataset will target the following scale:

*   **Total Samples**: **200-300** dialogues.
*   **Dialogue Length**: **15-30** turns per dialogue.
*   **Difficulty Mix**: Easy (20%), Medium (40%), Hard (40%).
*   **Challenge Guarantee**: Each generated dialogue is guaranteed to contain at least **one conflict**, **one long-span coreference**, and **one update event** to ensure a baseline level of difficulty across the entire dataset.

The initial version (v1) of the Med-LongMem dataset will target the following scale:

*   **Total Samples**: **200-300** dialogues.
*   **Dialogue Length**: **15-30** turns per dialogue.
*   **Challenge Guarantee**: Each generated dialogue is guaranteed to contain at least **one conflict**, **one long-span coreference**, and **one update event** to ensure a baseline level of difficulty across the entire dataset.
