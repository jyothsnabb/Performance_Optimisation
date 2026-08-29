# Spark UI Field Guide — Data + AI Academy Optimization Course

Companion reference for the QuickBite QSR optimization course. Covers what each Spark UI
tab shows, where to see it for this course's two demo paths, and a read-aloud script for
each technique. `SPARK UI CHECKPOINT` comments in the course notebooks and demo script
point back to the section numbers below.

**Two demo paths, one field guide:**
- **`Day1_Storage_Optimization_Notebook.py` / `Day2_Query_Execution_Optimization_Notebook.py`**
  — the full storage + query optimization course, run inside Databricks on classic
  compute. Covers everything including Delta-specific commands (`OPTIMIZE`, `ZORDER`,
  `VACUUM`, Liquid Clustering) and Databricks-managed features (Predictive Optimization).
- **`spark_ui_concepts_demo.py`** — a standalone, plain-PySpark script run locally in VS
  Code (see `VS_Code_Spark_UI_Setup.md`). Covers shuffle, skew, broadcast joins, and AQE
  with a real local Spark UI at `localhost:4040` — no Databricks dependency at all.
  This is the reliable fallback for these four topics whenever classic Databricks compute
  isn't available.

---

## §1 — Getting a Spark UI, for real

This took some working out over the course of building this material, so it's worth
being explicit about what actually works:

| Environment | Has Spark UI? | Notes |
|---|---|---|
| **Databricks Free Edition** | No | Serverless-only compute; no classic cluster to expose a UI. Free Edition gives **Query Profile** instead — useful for query-level plan inspection, but not a substitute for Jobs/Stages/Executors |
| **Databricks Community Edition** | No | **Retired January 1, 2026** — no longer an option |
| **Databricks 14-day Free Trial** | Yes (classic clusters) | Genuine classic compute with full Spark UI, same Unity Catalog setup as the course notebooks. In practice this route often asks for card details during signup depending on region/cloud-provider flow, which ruled it out for this course |
| **Databricks classic/interactive compute (paid workspace)** | Yes | The real target for running the Day 1/Day 2 notebooks as-is |
| **Local PySpark (`local[*]`)** | Yes, genuinely | Full Jobs/Stages/Executors/SQL tabs, real AQE, real shuffle — this is what `spark_ui_concepts_demo.py` uses |

**Bottom line for this course:** run `Day1`/`Day2` inside Databricks whenever classic
compute is available. For a guaranteed, always-available Spark UI — especially for
shuffle/skew/broadcast/AQE, which need no Delta or Unity Catalog features at all — use
the local `spark_ui_concepts_demo.py` path instead.

---

## §2 — Tab-by-tab reference

### Jobs tab
One row per Spark **job**. An action (`.collect()`, `.count()`, a write, `display()`)
triggers a job; a single cell can trigger more than one, and AQE re-planning can add
extra ones. Click a job to see its stage DAG.

**Cert relevance:** one wide-transformation chain = one job, split into stages at each
shuffle boundary.

### Stages tab
One row per **stage** — a set of tasks that can run without a shuffle in between. Click
into a stage for:
- **Summary Metrics** at the top — min/25th/median/75th/max for task duration and
  shuffle read size. **The fastest place to spot skew** — a big gap between median and
  max is the signature, not total runtime.
- A **sortable task table** — sort by Duration or Shuffle Read Size to find the exact
  outlier task(s).

### Executors tab
One row per executor (+ driver): cores, memory, active tasks, GC time, shuffle
read/write, **disk spill**. Nonzero spill signals partitions too large for available
memory.

**Cert relevance:** unified memory model, spill-to-disk as a performance/OOM precursor.

### Storage tab
Lists cached RDDs/DataFrames (`.cache()`/`.persist()`).

### Environment tab
Every active Spark/Databricks conf value, live — the tab to check instead of printing
`spark.conf.get(...)` in code.

### SQL / DataFrame tab
**The most important tab for this course.** One row per SQL/DataFrame **query** —
distinct from a job, since a query can span multiple jobs (especially with AQE
re-optimization). Click a query for a visual DAG of the physical plan: `Scan`, `Filter`,
`Exchange` (shuffle), `BroadcastExchange`, `Sort`, `HashAggregate`, join nodes — the same
names `.explain()` prints as text. This is also where AQE becomes visible — see §4.

---

## §3 — Section-by-section map

### Databricks course notebooks (Day 1 / Day 2)

| Section | What to open | What you're looking for |
|---|---|---|
| Day 1 §A.5 baseline query | SQL/DataFrame tab → Delta scan node | "number of files read" / "size of files read" (before) |
| Day 1 §B OPTIMIZE | Jobs / Stages tab | Mostly file I/O, almost no shuffle |
| Day 1 §C.1 ZCube tags | *(none — delta log only)* | No Spark UI equivalent |
| Day 1 §C.2 data skipping | SQL/DataFrame tab → scan node | Sharp drop in "files read" vs. §A.5, plus `PushedFilters` |
| Day 1 §D Liquid Clustering | Jobs / Stages tab | Same shape as §B |
| Day 1 §E VACUUM | Jobs tab only | Maintenance command, never on the SQL/DataFrame tab |
| Day 1 §F Broadcast joins | SQL/DataFrame tab | `BroadcastExchange` vs. plain `Exchange` |
| Day 2 §A pushdown/pruning | SQL/DataFrame tab → scan node | `PushedFilters`, `ReadSchema` column count |
| Day 2 §B.1–B.2 pruning/DPP | SQL/DataFrame tab | `PartitionFilters` populated; DPP subquery on the dimension side |
| Day 2 §C Predictive Optimization | *(none — Catalog Explorer instead)* | Background serverless jobs, not visible on any cluster's Spark UI |
| Day 2 §D shuffle | Stages tab + SQL tab `Exchange` node | Max vs. median task duration/shuffle size |
| Day 2 §D.4 AQE | SQL/DataFrame tab | `AQEShuffleRead` node; join type before vs. after execution |
| Day 2 §E skew | Stages tab, task table sorted by duration | One or two dramatically longer tasks |
| Day 2 §E.4 salting | Stages tab, same job as §E.2 | Tighter spread after salting |

### Local concept demo (`spark_ui_concepts_demo.py`)

| Demo section | What to open | What you're looking for |
|---|---|---|
| 1 — Shuffle | SQL tab `Exchange` node + Stages Summary Metrics | Shuffle bytes written; duration/shuffle-size spread |
| 2 — Skew | Stages tab, task table sorted by Duration | One dramatic outlier task (store 101) |
| 2b — Salting fix | Stages tab, compare to step 2's job | Tighter spread, comparable total runtime |
| 3 — Broadcast | SQL/DataFrame tab | `BroadcastExchange` vs. `Exchange`; unhinted plan may already differ from hinted |
| 4 — AQE | SQL/DataFrame tab (after execution) | `AQEShuffleRead` node; **confirmed via smoke test** — the unhinted join in this exact script showed a runtime `BroadcastHashJoin` conversion in its post-execution plan |

---

## §4 — AQE deep dive

**What AQE does, three optimizations:**
1. **Coalesces post-shuffle partitions** at runtime instead of leaving
   `spark.sql.shuffle.partitions` (default 200) fixed
2. **Dynamically switches join strategy** — sort-merge → broadcast — after seeing actual
   runtime shuffle statistics
3. **Dynamically splits skewed partitions**

**Two ways to see it — code and Spark UI, and you want both:**

**In code:** call `.explain()` on a DataFrame *before* it has executed — shows
`AdaptiveSparkPlan isFinalPlan=false`, the initial estimate. Trigger execution
(`.collect()`, `.count()`), then call `.explain()` again on the **same** DataFrame
variable — shows `isFinalPlan=true` with the plan Spark **actually ran**. This is
`spark_ui_concepts_demo.py` §4, and the smoke-test run of that exact script confirmed
this works: the "after" output showed both an `AQEShuffleRead` node (with
`Arguments: coalesced`) and the unhinted join converted to `BroadcastHashJoin`.

**In the Spark UI:** open the query on the SQL/DataFrame tab after it's run and look for:
- **`AQEShuffleRead`** in place of a plain `Exchange` — partition coalescing/splitting
- **A join node reading `BroadcastHashJoin`** with no explicit `broadcast()` hint in the
  code — AQE's runtime conversion
- **Skewed-partition split annotations** on a shuffle read node
- **Per-node runtime metrics** (actual rows/size/duration) — measured, not estimated

---

## §5 — Master defaults cheat sheet

| Area | Setting | Default |
|---|---|---|
| File layout | Healthy Parquet file size | 16MB – 1GB |
| `OPTIMIZE` | Manual compaction target | 1GB (`delta.targetFileSize`) |
| Auto Optimize | Optimize Write / Auto Compact target | 128MB (Databricks-managed only) |
| Z-ORDER | Max columns | 4 |
| Data skipping | Columns with min/max stats | first 32 (`delta.dataSkippingNumIndexedCols`) |
| `VACUUM` | Data file retention | 7 days (`delta.deletedFileRetentionDuration`) |
| `VACUUM` | Log retention | 30 days (`delta.logRetentionDuration`) |
| Broadcast join | Static threshold | 10MB (`spark.sql.autoBroadcastJoinThreshold`) |
| Broadcast join | AQE runtime threshold | 30MB |
| Broadcast join | Hard limits | never > 1GB disk; 8GB in-memory cap |
| Predictive Optimization | Auto-enabled since | accounts created on/after Nov 11, 2024 |
| Predictive Optimization | Runs ZORDER? | No |
| Shuffle | Default partitions | 200 (`spark.sql.shuffle.partitions`) |
| Shuffle | Target size/task | 128–200MB |
| Skew (AQE) | Partition factor / size threshold | 5x median / 256MB |
| DPP | Enabled since | Spark 3.0+ |
| DFP | Enabled since | DBR 6.1+ |

---

## §6 — Delivery script for professors

**OPTIMIZE:** "This is compaction. Before: hundreds of tiny files from small append
batches. After: a handful of large ones, up to 1GB each. This job is almost pure disk
I/O, no shuffle — OPTIMIZE rewrites files, it never moves rows between machines by key."

**Z-ORDER:** "OPTIMIZE fixed file size. Z-ORDER fixes file contents — it sorts rows so
similar `customer_id` values sit near each other, so Delta's per-file min/max stats let
it skip files that can't contain the value we're looking for, without opening them."

**Liquid Clustering:** "Same ZCube mechanism as Z-ORDER, but as a table property you set
once. If you change the clustering column later, only new data picks up the new layout —
no full rewrite."

**VACUUM:** "OPTIMIZE and Z-ORDER don't delete the old files they replace — they just
stop pointing the current version at them, so time travel keeps working. VACUUM is the
only step that physically deletes anything, only past the retention window."

**Broadcast joins:** "A join usually shuffles both tables across the network. If one side
is small enough, Spark can just copy it to every machine instead. Watch the SQL tab: the
hinted version has no Exchange node on the big table's side at all."

**Predicate pushdown / column pruning:** "The filter and column list both push down to
the file format itself for Parquet and Delta — Spark never reads what it doesn't need.
JSON can't do this — there's no cheap way to know what's inside without opening it."

**Partition pruning / DPP:** "Static pruning: filter directly on the partition column,
Spark skips whole folders. DPP is the same idea when the partitions to skip only become
known after filtering a joined table — Spark works that out automatically at runtime."

**Predictive Optimization:** "Databricks running OPTIMIZE, VACUUM, and ANALYZE for you
automatically, on its own serverless compute, based on how the table's actually used —
you won't see it in any cluster's Spark UI because it isn't running on your cluster."

**Shuffle & AQE:** "Every join or groupBy moves data across the network so matching keys
land together — usually the most expensive part of a job. AQE watches actual data sizes
as the query runs and adjusts on the fly — merging shuffle partitions that turned out too
small, or switching to a broadcast join if a table turned out smaller than expected."

**Skew & salting:** "One store has 40% of all the orders. In a shuffle, all its rows land
on one task — everyone else finishes, that one keeps running. Salting splits that one key
into several fake sub-keys so its rows spread across multiple tasks — we explode the
small side too, so every salted row still finds its match."
