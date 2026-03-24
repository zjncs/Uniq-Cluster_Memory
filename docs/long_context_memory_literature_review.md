# Long-Context Memory Systems for LLMs: Literature Review (2024-2026)

*Compiled: 2026-03-22*

---

## 1. Key Papers and Methods

### 1.1 MemoryBank (AAAI 2024)

- **Paper**: [arXiv:2305.10250](https://arxiv.org/abs/2305.10250)
- **Core contribution**: A long-term memory mechanism for LLMs with three components: (1) storage of daily chats, event summaries, and user personality assessments; (2) vector-encoded retrieval for dialogues and summaries; (3) memory intensity update using an exponential decay model inspired by the **Ebbinghaus Forgetting Curve**.
- **Memory updates/conflicts**: Uses forgetting-curve-based decay to selectively prune stale memories. No explicit contradiction detection — relies on recency weighting.
- **Benchmarks**: Custom evaluation via "SiliconFriend" AI companion scenario (long-term dialogue).
- **Temporal reasoning**: Limited — timestamps are tracked but no structured temporal inference.
- **Limitations**: No explicit conflict detection between old and new facts. Forgetting curve is heuristic, not learned. Evaluation is narrow (companion chatbot only). No multi-hop reasoning over memory.

### 1.2 MemWalker (arXiv 2023, evaluated against 2024 baselines)

- **Paper**: [arXiv:2310.05029](https://arxiv.org/abs/2310.05029)
- **Core contribution**: Treats the LLM as an interactive agent that navigates a **hierarchical memory tree** built from text segments. Upon receiving a query, the model walks the tree to locate relevant segments.
- **Memory updates/conflicts**: Static — the tree is built once from input text. No mechanism for updating or handling contradictions in evolving documents.
- **Benchmarks**: Three long-context QA tasks; outperforms recurrence, retrieval, and vanilla LLM baselines.
- **Temporal reasoning**: None — purely spatial navigation of a document tree.
- **Limitations**: Highly dependent on the underlying LLM's reasoning capability for navigation decisions. Static tree structure cannot accommodate evolving information. Each query requires tree traversal (latency). Struggles with multi-hop questions across distant tree branches.

### 1.3 ReadAgent (arXiv 2024)

- **Paper**: [arXiv:2402.09727](https://arxiv.org/abs/2402.09727)
- **Core contribution**: A human-inspired reading agent with three steps: (1) **episode pagination** — LLM decides reading pause points; (2) **memory gisting** — compresses each page into a shorter gist; (3) **interactive look-up** — LLM examines gists and selectively retrieves original pages.
- **Memory updates/conflicts**: Static gist memory — no update mechanism.
- **Benchmarks**: Long-context QA tasks; also adapted to web navigation.
- **Temporal reasoning**: None.
- **Limitations**: Gist quality depends heavily on the LLM. Information loss during gisting is inevitable. No mechanism for evolving documents or contradictory information.

### 1.4 LongMem (NeurIPS 2023)

- **Paper**: [arXiv:2306.07174](https://arxiv.org/abs/2306.07174)
- **Core contribution**: A **decoupled network architecture** with: (1) frozen backbone LLM as memory encoder; (2) adaptive residual **SideNet** as memory retriever/reader; (3) **Cache Memory Bank** storing attention key-value pairs from previous segments. The frozen backbone avoids catastrophic forgetting while SideNet learns to retrieve from the memory bank.
- **Memory updates/conflicts**: Memory bank is append-only with cached KV pairs. No explicit conflict resolution — retrieval is based on attention similarity.
- **Benchmarks**: ChapterBreak (40.5% accuracy, SOTA), memory-augmented in-context learning.
- **Temporal reasoning**: None.
- **Limitations**: Memory bank grows linearly; no consolidation or forgetting. KV-cache memory is opaque (not human-interpretable). No structured knowledge representation.

### 1.5 SCM — Self-Controlled Memory (arXiv 2023, updated 2024)

- **Paper**: [arXiv:2304.13343](https://arxiv.org/abs/2304.13343)
- **Core contribution**: Three components: (1) LLM-based agent backbone; (2) **memory stream** storing agent memories; (3) **memory controller** that determines when/how to update and utilize memories. Plug-and-play with any instruction-following LLM, no fine-tuning required.
- **Memory updates/conflicts**: The memory controller manages updates but primarily through summarization — no explicit contradiction detection.
- **Benchmarks**: Custom ultra-long text evaluation (20K to 2M tokens) covering long-term dialogues, book summarization, meeting summarization.
- **Temporal reasoning**: Not explicitly addressed.
- **Limitations**: Memory controller decisions depend on LLM quality. Summarization-based memory loses detail. No structured temporal or relational modeling.

### 1.6 MemoRAG (TheWebConf 2025)

- **Paper**: [arXiv:2409.05591](https://arxiv.org/abs/2409.05591)
- **Core contribution**: A **dual-system RAG architecture**: (1) a lightweight, super-long-range model creates a **global memory** of the entire context via KV compression; (2) an expensive, expressive model generates answers from retrieved evidence. The memory model is trained with **RLGF** (Reinforcement Learning from Generation Feedback) to improve its clue-generation ability.
- **Memory updates/conflicts**: The global memory is rebuilt when context changes. No incremental update or conflict resolution mechanism.
- **Benchmarks**: Various long-context evaluation tasks; handles contexts up to 400K tokens (Qwen2-based) or 128K (Mistral-based).
- **Temporal reasoning**: Not explicitly addressed.
- **Limitations**: Requires a dedicated memory model (additional training). Global memory is a lossy compression — detail can be lost. No temporal awareness or conflict handling.

### 1.7 HippoRAG (NeurIPS 2024) and HippoRAG 2

- **Paper**: [arXiv:2405.14831](https://arxiv.org/abs/2405.14831)
- **Core contribution**: Inspired by **hippocampal indexing theory** of human long-term memory. Synergistically orchestrates: (1) an LLM for open information extraction (neocortex analog); (2) a **schemaless knowledge graph** as artificial hippocampal index; (3) **Personalized PageRank** for retrieval across the KG. Enables knowledge integration across passage boundaries.
- **Memory updates/conflicts**: New passages are integrated into the KG via open IE. However, conflict detection between existing and new triples is not explicitly handled — the KG grows but doesn't reconcile contradictions.
- **Benchmarks**: Multi-hop QA (up to 20% improvement over SOTA). Single-step retrieval is 10-30x cheaper and 6-13x faster than iterative methods like IRCoT.
- **Temporal reasoning**: Not explicitly modeled in the KG structure.
- **HippoRAG 2**: Adds deeper passage integration and more effective online LLM use. 7% improvement on associative memory tasks. Improves across both simple and complex tasks.
- **Limitations**: KG construction requires LLM calls (cost). No temporal metadata on triples. No explicit contradiction detection when facts evolve. Open IE quality affects downstream retrieval.

### 1.8 MemGPT / Letta (NeurIPS 2024)

- **Paper**: [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- **Core contribution**: Draws analogy with **virtual memory in operating systems**. Two-tier memory: (1) **main context** (in-context, analogous to RAM); (2) **external context** (archival + recall storage, analogous to disk). The LLM uses function calls to page data between tiers. Supports self-editing memory through tool use.
- **Memory updates/conflicts**: The LLM can explicitly edit/overwrite memories via function calls. Conflict resolution is delegated to the LLM's judgment during editing — no formal mechanism.
- **Benchmarks**: Document analysis (large docs exceeding context window), multi-session chat (DMR benchmark: 93.4% accuracy with GPT-4-turbo).
- **Temporal reasoning**: No explicit temporal model.
- **Limitations**: Quality of memory management depends entirely on the LLM's ability to make good paging decisions. No structured knowledge representation. Memory editing is ad hoc (no formal consistency guarantees).

### 1.9 RAPTOR (ICLR 2024)

- **Paper**: [arXiv:2401.18059](https://arxiv.org/abs/2401.18059)
- **Core contribution**: Recursively embeds, clusters (using GMMs with soft clustering), and summarizes text chunks to build a **tree with multiple levels of abstraction**. Retrieval can operate at any granularity level. Uses UMAP for dimensionality reduction and BIC for optimal cluster count.
- **Memory updates/conflicts**: Static tree — built once from a corpus. No update or conflict mechanism.
- **Benchmarks**: NarrativeQA, QASPER, QuALITY. 20% absolute accuracy improvement on QuALITY with GPT-4.
- **Temporal reasoning**: None.
- **Limitations**: Purely static. Recursive summarization can lose fine-grained details. No mechanism for evolving documents.

### 1.10 A-Mem — Agentic Memory (NeurIPS 2025)

- **Paper**: [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)
- **Core contribution**: Inspired by the **Zettelkasten method** — creates interconnected knowledge networks through dynamic indexing and linking. Each memory becomes a comprehensive note with structured attributes (contextual descriptions, keywords, tags) and embedding vectors. Enables autonomous, flexible memory management without predefined operations.
- **Memory updates/conflicts**: Memories are dynamically linked and organized. However, conflict detection between contradictory memories is not a primary focus — the system treats all memories uniformly.
- **Benchmarks**: Evaluated on 6 foundation models; outperforms existing baselines.
- **Temporal reasoning**: Not explicitly addressed.
- **Limitations**: Quality depends on the underlying LLM. Treats all memory uniformly (no distinction between stable facts and evolving information). Text-only.

### 1.11 MemoryLLM (ICML 2024)

- **Paper**: [arXiv:2402.04624](https://arxiv.org/abs/2402.04624)
- **Core contribution**: Augments a transformer with a **fixed-size memory pool** in the latent space of every transformer layer. The model can **self-update** by processing new text and writing to the memory pool. No catastrophic forgetting observed even after ~1M memory updates.
- **Memory updates/conflicts**: Self-update mechanism overwrites portions of the memory pool. Implicit conflict handling through the fixed-size constraint (new information displaces old). No explicit contradiction detection.
- **Benchmarks**: Model editing benchmarks, custom long-term retention evaluations, long-context benchmarks.
- **Temporal reasoning**: Not explicitly modeled.
- **Limitations**: Fixed memory size limits total knowledge capacity. Memory is in latent space — not interpretable. Cannot selectively retain or forget specific facts.

### 1.12 GraphReader (EMNLP 2024 Findings)

- **Paper**: [arXiv:2406.14550](https://arxiv.org/abs/2406.14550)
- **Core contribution**: Structures long texts into a **graph** of (key element, atomic facts) nodes linked by shared elements. An agent explores the graph via coarse-to-fine navigation with predefined functions, maintaining a plan and recording insights.
- **Memory updates/conflicts**: Static graph — built once from input text.
- **Benchmarks**: LV-Eval (16K-256K contexts), HotpotWikiQA-mixup; outperforms GPT-4-128k using only a 4K context window.
- **Temporal reasoning**: None.
- **Limitations**: Static construction. Graph quality depends on extraction accuracy. Navigation can be slow for very large graphs.

### 1.13 Zep / Graphiti (January 2025)

- **Paper**: [arXiv:2501.13956](https://arxiv.org/abs/2501.13956)
- **Core contribution**: A **temporal knowledge graph** engine with **bi-temporal modeling**: Event Time (when a fact occurred) and Ingestion Time (when it was recorded). Four timestamps per fact: t'_created, t'_expired, t_valid, t_invalid. Hybrid search combining semantic embeddings, BM25, and graph traversal — no LLM calls during retrieval (P95 latency 300ms).
- **Memory updates/conflicts**: **Explicit conflict resolution via temporal metadata** — when conflicts arise, Graphiti uses temporal metadata to update or invalidate (but not discard) outdated information, preserving historical accuracy. This is one of the few systems with principled conflict handling.
- **Benchmarks**: DMR benchmark (94.8% vs MemGPT's 93.4%), LongMemEval (up to 18.5% improvement with 90% latency reduction).
- **Temporal reasoning**: **Yes** — bi-temporal modeling is a core design principle.
- **Limitations**: Requires Neo4j or similar graph database infrastructure. KG construction still requires LLM calls. Commercial product (open-source core but enterprise features are paid).

### 1.14 LIGHT Framework (ICLR 2026)

- **Paper**: [arXiv:2510.27246](https://arxiv.org/abs/2510.27246)
- **Core contribution**: Cognition-inspired three-component memory: (1) **episodic memory** — long-term index of full conversation for retrieval; (2) **working memory** — most recent turns; (3) **scratchpad** — after each turn, the model reasons and records salient facts. Designed for ultra-long conversations (up to 10M tokens).
- **Memory updates/conflicts**: The scratchpad is continuously updated. However, contradiction detection between scratchpad entries is not formally addressed.
- **Benchmarks**: BEAM benchmark (100 conversations, 2000 questions, up to 10M tokens). 3.5-12.69% improvement over strongest baselines.
- **Temporal reasoning**: Partially — event ordering is one of BEAM's evaluation dimensions.
- **Limitations**: Scratchpad growth must be managed. No formal conflict resolution. Performance still degrades significantly at extreme lengths.

### 1.15 Additional Notable Methods (2024-2026)

| Method | Venue | Key Idea |
|--------|-------|----------|
| **TSM** (Temporal Semantic Memory) | Jan 2026 | Duration-aware memory consolidation with semantic-time grounding. 74.8% on LongMemEval_S (vs 62.6% for A-Mem) |
| **REMem** | Feb 2026 | Explicit episodic memory with spatiotemporal context and situational grounding |
| **TReMu** | ACL Findings 2025 | Neuro-symbolic temporal reasoning using Python-based symbolic execution |
| **MemoTime** | Oct 2025 | Temporal KG + Tree-of-Time hierarchical reasoning + experience memory |
| **MemOS** | 2025 | Memory operating system treating memory as a system resource with MemCubes (content + metadata + versioning) |
| **M+** | 2025 | Extension of MemoryLLM with scalable long-term memory and co-trained retriever |
| **LightMem** | ICLR 2026 | Lightweight, modular memory-augmented generation |
| **MemoryAgentBench** | ICLR 2026 | Benchmark converting long-context datasets to multi-turn incremental format |
| **LOCCO** | ACL Findings 2025 | Long-term dialogue memory evaluation with 100 users, 3080 interactions |
| **Reflective Memory Mgmt** | ACL 2025 | Reflective memory management for long-term conversations |
| **EvoReasoner + EvoKG** | Sep 2025 | Temporal-aware multi-hop reasoning with KG that resolves factual contradictions |
| **AriGraph** | IJCAI 2025 | Episodic + semantic KG for LLM agents in partially observable environments |

---

## 2. Evaluation Benchmarks

### Tier 1: Widely Accepted Community Standards

| Benchmark | Venue | Focus | Scale | Key Feature |
|-----------|-------|-------|-------|-------------|
| **LongBench** | ACL 2024 | General long-context understanding | 21 tasks, ~6.7K avg words | Bilingual (EN+ZH), multi-task |
| **LongBench v2** | ACL 2025 | Deep understanding + reasoning | 503 questions, 8K-2M words | Human experts achieve only 53.7% |
| **InfiniteBench** | ACL 2024 | Ultra-long context (100K+) | 12 tasks, 100K+ tokens avg | First benchmark >100K tokens |
| **LongMemEval** | ICLR 2025 | Long-term interactive memory | 500 questions, 115K-1.5M tokens | 5 core abilities: IE, MR, TR, KU, ABS |
| **SCROLLS** | EMNLP 2022 | Long-text synthesis | 7 datasets, up to 100K tokens | Foundational; inspired later benchmarks |

### Tier 2: Specialized / Emerging Benchmarks

| Benchmark | Focus | Scale | Notes |
|-----------|-------|-------|-------|
| **BEAM** (ICLR 2026) | Long-term conversational memory | 100 convos, 2000 Qs, up to 10M tokens | Tests 10 memory abilities including contradiction resolution |
| **RULER** (COLM 2024) | True context utilization | Flexible length, 13 tasks | Extends NIAH with multi-hop and aggregation |
| **LooGLE** (ACL 2024) | Long context with up-to-date docs | 24K+ tokens/doc | Post-2022 documents to avoid contamination |
| **DMR** (MemGPT) | Multi-session chat memory | Conversation-based | Used by MemGPT and Zep |
| **MemoryAgentBench** (ICLR 2026) | Incremental multi-turn memory | Converted long-context datasets | Multi-turn format for agent evaluation |
| **LOCCO** (ACL Findings 2025) | Long-term dialogue memory | 100 users, 3080 interactions | Supervised fine-tuning based |
| **MemBench** | Continual learning from feedback | Various | Tests feedback utilization |

### Community Consensus
- **LongBench (v1/v2)** and **InfiniteBench** are the most widely used for general long-context evaluation.
- **LongMemEval** is becoming the standard for long-term interactive memory (ICLR 2025 acceptance signals strong community endorsement).
- **BEAM** is the newest major benchmark (ICLR 2026), notable for testing contradiction resolution and event ordering at extreme scale (10M tokens).
- **SCROLLS** is foundational but somewhat superseded by newer benchmarks.
- **RULER** has largely replaced vanilla NIAH as the standard for probing true context utilization.

---

## 3. Taxonomy of Approaches

### By Memory Architecture

| Category | Methods | Strengths | Weaknesses |
|----------|---------|-----------|------------|
| **Hierarchical Tree** | MemWalker, RAPTOR, ReadAgent | Multi-granularity retrieval | Static; no updates; info loss in summarization |
| **Knowledge Graph** | HippoRAG, Zep/Graphiti, GraphReader, EvoKG | Structured relations; multi-hop reasoning | Construction cost; extraction errors |
| **Latent Memory Pool** | LongMem, MemoryLLM, M+ | Integrated into model; fast access | Opaque; fixed capacity; not interpretable |
| **External DB + Paging** | MemGPT/Letta, SCM | Flexible; works with any LLM | Quality depends on LLM's paging decisions |
| **Dual-System (Global + Local)** | MemoRAG, LIGHT | Global context awareness | Requires dedicated memory model |
| **Agentic / Dynamic** | A-Mem, MemOS | Self-organizing; flexible operations | Quality depends on LLM; overhead |
| **Forgetting-Curve Based** | MemoryBank | Biologically inspired decay | Heuristic; no learned policy |

### By Conflict Handling Capability

| Level | Methods | Mechanism |
|-------|---------|-----------|
| **Explicit temporal conflict resolution** | Zep/Graphiti, EvoKG | Bi-temporal modeling; fact invalidation with history preservation |
| **Implicit overwrite** | MemoryLLM, MemGPT | New information displaces old via fixed capacity or LLM editing |
| **Forgetting-based** | MemoryBank | Ebbinghaus decay removes stale memories |
| **None** | MemWalker, ReadAgent, RAPTOR, HippoRAG, MemoRAG, LongMem | Static construction or append-only |

---

## 4. Open Gaps and Unsolved Problems

### 4.1 Conflict Detection in Evolving Information

**Status: Largely unsolved.** This is the most critical gap.

- The EMNLP 2024 survey on Knowledge Conflicts identifies three conflict types (context-memory, inter-context, intra-memory) but notes that existing approaches rely on simple prior assumptions for resolution.
- DYNAMICQA and MULAN report **contradictory findings** on whether external context can resolve temporal fact conflicts — the question remains open.
- Only **Zep/Graphiti** and **EvoKG** attempt principled conflict resolution via temporal metadata, but neither handles subjective belief evolution (e.g., "I used to like X but now prefer Y").
- Most memory systems are **append-only or overwrite-only** — they cannot detect when a new fact contradicts an existing one and reason about which is correct.
- The BEAM benchmark (ICLR 2026) introduces contradiction resolution as an evaluation dimension, signaling growing community awareness, but no method yet excels at this.

**Key unsolved sub-problems:**
1. Detecting contradictions between stored memories and new input without exhaustive pairwise comparison
2. Distinguishing genuine updates (facts that changed) from errors/hallucinations
3. Maintaining a "belief state" that reflects the most current understanding while preserving history
4. Handling partial contradictions (some aspects changed, others didn't)

### 4.2 Temporal Grounding of Facts

**Status: Early-stage solutions emerging, but far from solved.**

- Most memory systems (MemWalker, ReadAgent, RAPTOR, HippoRAG, MemoRAG, LongMem, A-Mem) have **zero temporal awareness** — they treat all stored information as equally current.
- **Zep/Graphiti** is the most advanced with bi-temporal modeling (event time + ingestion time), but it is a commercial system and its temporal reasoning is limited to fact validity windows.
- **TSM** (Temporal Semantic Memory, Jan 2026) introduces duration-aware consolidation and achieves the best LongMemEval results, but is very recent.
- **MemoTime** (Oct 2025) combines temporal KGs with hierarchical reasoning but focuses on temporal KG QA, not general conversation memory.
- **TReMu** (ACL 2025) proposes neuro-symbolic temporal reasoning but uses a separate symbolic engine rather than integrating temporality into the memory itself.

**Key unsolved sub-problems:**
1. Representing time natively in memory structures (not just as metadata)
2. Handling implicit temporal references ("last time", "before my trip", "recently")
3. Temporal scoping of retrieved memories (knowing when facts were valid)
4. Combining temporal and causal reasoning over memory
5. Accuracy drops 23-35% when shifting from absolute dates to relative temporal references

### 4.3 Structured Memory vs. Free-Form Summary

**Status: Strong evidence favoring structured approaches, but practical challenges remain.**

- Empirical evidence strongly favors structured memory: Zep/Graphiti (94.8%) and AriGraph significantly outperform recursive summarization baselines (~35%) on memory benchmarks.
- However, structured approaches (KG-based) require expensive construction (LLM calls for extraction), are brittle to extraction errors, and scale poorly.
- Free-form summaries are cheap and easy but suffer from information loss, lack of provenance, and inability to support multi-hop reasoning.
- **The emerging consensus is hybrid approaches**: use structured representations for factual/relational information and free-form summaries for context/background.
- The dual episodic + semantic memory pattern (inspired by human cognition) is gaining traction: AriGraph, Zep, LIGHT, and TSM all incorporate elements of this.

**Key unsolved sub-problems:**
1. Automatic schema discovery for memory structures (most systems require predefined schemas or use schemaless triples)
2. Graceful degradation when extraction fails (structured systems can be fragile)
3. Balancing granularity: too fine-grained = noise; too coarse = info loss
4. Integrating structured and unstructured memory in a unified retrieval framework
5. Memory consolidation: when and how to merge redundant memories

### 4.4 Additional Open Challenges

1. **Memory inflation and self-degradation**: Naive "add-everything" strategies cause performance decay over time. Intelligent forgetting/consolidation policies are needed but not well-studied.
2. **Evaluation**: No single benchmark covers all dimensions (recall, reasoning, temporal awareness, conflict resolution, abstention). The field needs unified evaluation.
3. **Multi-agent memory**: Memory sharing, synchronization, and access control across agents is almost entirely unaddressed.
4. **Provenance and explainability**: Few systems track why a memory was stored or how it was derived, making debugging and trust difficult.
5. **Scalability**: Most methods are demonstrated at <1M tokens. Behavior at 10M+ tokens (BEAM-scale) is largely unknown.
6. **Modality**: Nearly all work is text-only. Multi-modal memory (images, audio, structured data) is unexplored.

---

## 5. Summary Table: Method Comparison

| Method | Venue | Memory Type | Updates | Conflict | Temporal | Max Scale |
|--------|-------|-------------|---------|----------|----------|-----------|
| MemoryBank | AAAI 2024 | Vector store + summaries | Forgetting curve | None | Timestamps only | Unbounded |
| MemWalker | arXiv 2023 | Hierarchical tree | None (static) | None | None | ~100K |
| ReadAgent | arXiv 2024 | Gist directory | None (static) | None | None | ~500K |
| LongMem | NeurIPS 2023 | KV cache bank | Append-only | None | None | 65K |
| SCM | arXiv 2023/2024 | Memory stream | Controller-managed | None | None | 2M |
| MemoRAG | TheWebConf 2025 | Global KV compression | Rebuild | None | None | 400K |
| HippoRAG | NeurIPS 2024 | Knowledge graph + PPR | Append via open IE | None | None | Corpus-level |
| MemGPT | NeurIPS 2024 | Two-tier (main + archival) | LLM-driven editing | LLM judgment | None | Unbounded |
| RAPTOR | ICLR 2024 | Hierarchical tree | None (static) | None | None | Corpus-level |
| A-Mem | NeurIPS 2025 | Zettelkasten notes | Dynamic linking | None | None | Unbounded |
| MemoryLLM | ICML 2024 | Latent memory pool | Self-update | Implicit overwrite | None | Fixed pool |
| GraphReader | EMNLP 2024 | Element-fact graph | None (static) | None | None | 256K |
| Zep/Graphiti | arXiv 2025 | Temporal KG | Bi-temporal update | **Explicit** | **Yes** | Unbounded |
| LIGHT | ICLR 2026 | Episodic + working + scratchpad | Continuous scratchpad | None | Partial | 10M |
| TSM | Jan 2026 | Temporal semantic memory | Duration-aware | Partial | **Yes** | 1.5M |

---

## 6. Key References

### Surveys
- [A Comprehensive Survey on Long Context Language Modeling](https://arxiv.org/abs/2503.17407) (March 2025)
- [A Survey on the Memory Mechanism of LLM-based Agents](https://dl.acm.org/doi/10.1145/3748302) (ACM TOIS, 2024)
- [Knowledge Conflicts for LLMs: A Survey](https://arxiv.org/abs/2403.08319) (EMNLP 2024)
- [From Human Memory to AI Memory](https://arxiv.org/abs/2504.15965) (April 2025)
- [Memory in LLM-based Multi-Agent Systems](https://www.techrxiv.org/users/1007269/articles/1367390) (2025)

### Curated Paper Lists
- [Agent Memory Paper List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List) — "Memory in the Age of AI Agents"
- [Awesome LLM Long Context Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling)

### Benchmarks
- [LongBench / LongBench v2](https://github.com/THUDM/LongBench) (ACL 2024/2025)
- [InfiniteBench](https://github.com/OpenBMB/InfiniteBench) (ACL 2024)
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025)
- [BEAM](https://github.com/mohammadtavakoli78/BEAM) (ICLR 2026)
- [RULER](https://github.com/NVIDIA/RULER) (COLM 2024)
- [SCROLLS](https://www.scrolls-benchmark.com/) (EMNLP 2022)
