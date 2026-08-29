# Optimization Course — Use Case Brief & Full Technique Catalog

## Use case brief

**Context:** Data + AI Academy, run by Reshma Upadhyaya (Technical Lead & Instructor,
Platformatory) — a program preparing cohorts for Databricks certifications. This
material is one module within a longer curriculum; specifically, a standalone **2-day
optimization course** covering storage-layer and query-execution optimization on
Databricks, aimed at cohorts preparing for the **Databricks Certified Data Engineer
Professional** exam.

**Reference baseline:** Databricks' *"Comprehensive Guide to Optimize Databricks, Spark
and Delta Lake Workloads"* — every default value across the course notebooks and field
guide is sourced from that guide plus current Delta Lake / Unity Catalog documentation.

**Teaching domain — QuickBite QSR:** a fictional multi-store food & beverage chain, used
throughout both days so every technique fixes something real and visible in one running
dataset rather than a disconnected toy example:
- **`orders`** — a fact table built as ~1,000,000 rows across 200 small unoptimized
  append batches (simulating a drip-fed ingestion pattern with no file-size tuning), with
  a deliberate skew baked in: `store_id = 101` ("Flagship — MG Road") carries ~40% of all
  rows, used later for the data-skew and salting lab
- **`stores`** — a small (60-row) dimension table across 8 Indian cities, used for the
  broadcast-join and DPP labs

**Course structure:**
- **Day 1 — Storage Layer Optimization:** the small-files problem, `OPTIMIZE`
  (compaction), `ZORDER BY`, Liquid Clustering, `VACUUM`, broadcast joins
- **Day 2 — Query Execution & Distributed Processing:** predicate pushdown & column
  pruning, partition pruning (static, DPP, DFP), Predictive Optimization, shuffle
  mechanics & AQE, data skew & salting

**Delivery constraints worked through while building this material:**
- Databricks Free Edition is serverless-only and exposes no Spark UI (Query Profile is
  the closest substitute, but doesn't cover Jobs/Stages/Executors)
- Databricks Community Edition — the historical free option with real classic clusters —
  was retired January 1, 2026
- The 14-day Free Trial gives genuine classic compute but its signup flow asked for card
  details, ruling it out for this course
- **Resolution:** run the full course notebooks (`Day1`/`Day2`) inside Databricks on
  whatever classic compute is available for a given cohort session; use a standalone
  local PySpark script (`spark_ui_concepts_demo.py`, no Delta/Databricks dependency) as a
  guaranteed-available fallback specifically for shuffle, skew, broadcast join, and AQE
  demos, run from VS Code with a real `localhost:4040` Spark UI

**Course materials produced:**
- `Day1_Storage_Optimization_Notebook.py`, `Day2_Query_Execution_Optimization_Notebook.py`
  — production-style Databricks notebooks (cleaned, commented, Spark UI checkpoints
  throughout)
- `Spark_UI_Field_Guide.md` — tab-by-tab reference, section-by-section map, AQE deep
  dive, defaults cheat sheet, professor delivery script
- `VS_Code_Spark_UI_Setup.md` — Ubuntu/macOS/Windows local environment setup
- `spark_ui_concepts_demo.py` — standalone shuffle/skew/broadcast/AQE demo, verified
  working end-to-end

---

## Full technique catalog

Everything usable for performance optimization on Databricks that's either in the
current 2-day course, or a short, deliberately-scoped list of near-term candidates for
extending it.

### Storage & file layout

| Technique | What it does | In this course? |
|---|---|---|
| `OPTIMIZE` (bin-packing/compaction) | Merges small files toward a target size (1GB manual default) | ✅ Day 1 §B |
| `OPTIMIZE ... ZORDER BY` | Co-locates similar values so min/max stats skip more files | ✅ Day 1 §C |
| Liquid Clustering (`CLUSTER BY`) | Same mechanism as Z-Order, as an incremental table property | ✅ Day 1 §D |
| `VACUUM` | Physically deletes files no longer referenced past the retention window | ✅ Day 1 §E |
| Data skipping (min/max stats) | Automatic per-file column stats, first 32 columns by default | ✅ Day 1 §A/§C |
| Predictive Optimization | Databricks auto-runs OPTIMIZE/VACUUM/ANALYZE on UC managed tables | ✅ Day 2 §C (conceptual) |
| Partitioning strategy | Physical directory-level data layout by column | ✅ Day 2 §B (as a teaching aid; guidance to avoid on tables <1TB) |
| **Deletion Vectors** | Marks deleted/updated rows via a bitmap instead of rewriting whole files | Added to Notebook |
| **Predictive I/O** | Photon-exclusive; uses deletion vectors + ML-predicted access patterns to accelerate reads and DELETE/UPDATE/MERGE | Added to Notebook |
| **Table statistics** (`ANALYZE TABLE` / `COMPUTE STATISTICS`) | Cost-based optimizer input for join ordering, etc. | Added to Notebook |
| **Shallow & Deep Clones** | Shallow: metadata-only copy referencing original data files, cheap, for dev/test. Deep: full independent copy including data files, for backup/migration | Added to Notebook |

### Query execution & distributed processing

| Technique | What it does | In this course? |
|---|---|---|
| Predicate pushdown | Pushes filters to the storage engine (source-dependent: Parquet/Delta yes, JSON/XML no) | ✅ Day 2 §A |
| Column pruning | Reads only selected columns | ✅ Day 2 §A |
| Static partition pruning | Skips whole partition directories on a literal filter | ✅ Day 2 §B.1 |
| Dynamic Partition Pruning (DPP) | Prunes partitions using a runtime filter from a joined table | ✅ Day 2 §B.2 |
| Dynamic File Pruning (DFP) | File-level sibling of DPP, via Delta min/max stats | ✅ Day 2 §B.3 |
| Broadcast joins | Copies a small table to every executor to avoid a shuffle | ✅ Day 1 §F |
| Shuffle partition tuning | Manually sizing `spark.sql.shuffle.partitions` for a workload | ✅ Day 2 §D |
| Adaptive Query Execution (AQE) | Runtime partition coalescing, join-strategy switching, skew splitting | ✅ Day 2 §D.4 |
| Data skew handling / salting | Splitting a hot key into synthetic sub-keys to spread shuffle load | ✅ Day 2 §E |
| Join strategy selection | Broadcast hash / sort-merge / shuffle hash join trade-offs | ✅ Day 1 §F.3 (guardrails) |
| **Repartition vs. coalesce** | Increasing (repartition, full shuffle) vs. decreasing (coalesce, no shuffle) partition count | Added to Notebook |

---

## When to use which technique — decision guide

### Z-ORDER vs. Liquid Clustering (`CLUSTER BY`)

| Use Z-ORDER when... | Use Liquid Clustering when... |
|---|---|
| The table already exists and this is a one-off or occasional reorganization | You're creating a new table (or willing to migrate one) |
| You have 1–4 known, stable, high-cardinality filter/join columns | The clustering key might change over time |
| You're comfortable re-running `OPTIMIZE ... ZORDER BY` manually as data grows | You want incremental clustering upkeep without full rewrites when the key changes |
| — | You want it to work smoothly with Predictive Optimization / Automatic Liquid Clustering |

Databricks' own guidance leans toward Liquid Clustering as the default choice for new
Unity Catalog managed tables going forward, with Z-Order remaining relevant for existing
tables or one-off cleanups.

### Plain `OPTIMIZE` vs. `OPTIMIZE ... ZORDER BY` vs. `CLUSTER BY`

- **Plain `OPTIMIZE`** — the table just needs file-size compaction; there's no specific
  filter pattern worth optimizing data layout around
- **`OPTIMIZE ... ZORDER BY`** — there's a known, stable, high-cardinality column (or up
  to 4) that queries filter or join on repeatedly
- **`CLUSTER BY`** — same goal as Z-Order, but for a new table, or one where the ideal
  clustering key may change as query patterns evolve

### When to partition a table at all

Only for tables **over 1TB**, or when there's a natural, low-cardinality,
frequently-filtered column with roughly balanced partition sizes (e.g., a date column
with a sensible grain). Under 1TB, prefer Z-Order or Liquid Clustering instead — this is
Databricks' own stated guidance, and why `orders` stays unpartitioned through Day 1 and
only gets partitioned as a Day 2 teaching aid for the DPP/DFP lab specifically.

### When to broadcast a join

The small side is confidently under ~30MB (the AQE runtime threshold) or you're
comfortable forcing it below the 10MB static default with an explicit hint. Never
broadcast a table over 1GB on disk, and stay well clear of the 8GB in-memory hard cap —
watch out for heavily-compressed formats where on-disk size understates in-memory size
significantly.

### When to run `VACUUM`, and how often

As its own regularly scheduled job — never bundled into the ingestion job, and never run
immediately after `OPTIMIZE`/`ZORDER` in production, since replaced files might still be
in use by an in-flight query or a time-travel read. Never drop retention below 7 days in
production. Where eligible (UC managed tables, Premium plan), Predictive Optimization can
take this off your schedule entirely.

---

## How to avoid a shuffle

1. **Filter and select columns as early as possible** — predicate pushdown and column
   pruning both reduce what a later shuffle has to move
2. **Broadcast the small side of a join** instead of shuffling both sides across the
   network
3. **Lean on partition pruning** (static filters, or DPP for joined dimension filters) so
   a shuffle stage that does happen touches less data
4. **Avoid unnecessary wide transformations** — a `repartition()`, `distinct()`, or
   `orderBy()` that isn't actually needed for correctness is a shuffle you're paying for
   with nothing to show for it
5. **Cache a DataFrame that's reused across multiple actions** instead of letting Spark
   recompute (and reshuffle) it from scratch each time
6. **Let AQE coalesce partitions automatically** rather than manually over-provisioning
   `spark.sql.shuffle.partitions` "just in case" — a static, too-high partition count
   creates its own overhead

## How to avoid or fix data skew

Escalation order — try each before reaching for the next:

1. **Filter out the skewed value first**, if business logic allows it — the cheapest fix
   by far, when it's applicable at all
2. **Try an explicit skew hint** before reaching for a code-level workaround
3. **Let AQE's automatic skew-join handling catch it** — it activates when a partition
   clears both the 5x-median-size factor and the 256MB minimum threshold. Check the
   Stages tab for a "skewed partitions handled" annotation before assuming you need to
   intervene manually
4. **Isolate and broadcast the hot key** — split the join into two: broadcast-join just
   the rows matching the skewed key, run a normal shuffle join on everything else, then
   union the two results. Often cheaper to implement than full salting when only one or
   two keys are the problem
5. **Salt the hot key as a last resort** — spread one key's rows across N synthetic
   sub-keys so they land on multiple shuffle partitions instead of one. Requires code
   changes on both sides of the join (exploding the small side to match every salt value),
   which is exactly why it's last on this list, not first

---

## Candidates for future addition to the notebooks

The four items kept above as "not yet covered" are deliberately scoped as near-term,
addable extensions to the *existing* two days, not a new module:

1. **Deletion Vectors + Predictive I/O** — natural extension of the existing Day 1
   storage narrative; ties directly into the existing `VACUUM`/file-management story,
   since deletion vectors change what "a file needs rewriting" even means
2. **Table statistics (`ANALYZE TABLE`)** — ties into the existing Day 1 broadcast-join
   and Day 2 join-strategy content, since the cost-based optimizer uses these stats for
   exactly those decisions
3. **Shallow & Deep Clones** — fits naturally alongside the existing Day 1 `VACUUM`/time
   travel material, since clones share the same underlying "the log, not the files, is
   the source of truth" idea
4. **Repartition vs. coalesce** — a short addition to the existing Day 2 shuffle section,
   directly next to the manual shuffle-partition-tuning material already there
