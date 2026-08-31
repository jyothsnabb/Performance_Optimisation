# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Day 2 — Query Execution & Distributed Processing Optimization
# MAGIC ### Predicate Pushdown · Partition Pruning (DPP/DFP) · Table Statistics · Predictive Optimization · Shuffle, Repartition/Coalesce & AQE · Data Skew & Salting
# MAGIC
# MAGIC **Continues from Day 1** — same catalog/schema, same `orders`/`stores` tables (now
# MAGIC optimized + Z-ordered).
# MAGIC
# MAGIC **Also runs on Databricks Free Edition serverless compute**, with the same
# MAGIC consequences as Day 1, plus one new one: **DBFS is disabled on serverless** — `/tmp`
# MAGIC paths don't work. This notebook uses a workspace path instead (widget below). AQE and
# MAGIC Auto-Optimized Shuffle are **on by default on serverless**, so several cells that used
# MAGIC to `spark.conf.set(...)` them now just confirm they're already active. Forcing
# MAGIC `autoBroadcastJoinThreshold = -1` to disable broadcast for the skew demo also isn't
# MAGIC available on serverless — join strategy is automatically managed — so Section F
# MAGIC observes skew as Spark actually chooses to run it, not under a forced condition.
# MAGIC
# MAGIC **Every section has a VS Code Spark UI companion.** Unlike Day 1, nothing here needs
# MAGIC Delta at all — plain PySpark, so the companion setup is much lighter (no `delta-spark`
# MAGIC pinning required).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Unity Catalog catalog")
dbutils.widgets.text("schema", "optimization_demo", "Schema (from Day 1)")
dbutils.widgets.text("workspace_base_path", "/Workspace/Shared/optimization_demo", "Workspace path for temp files (DBFS disabled on serverless)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
workspace_base_path = dbutils.widgets.get("workspace_base_path")

from pyspark.sql import functions as F
import time

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

ORDERS = f"{catalog}.{schema}.orders"
STORES = f"{catalog}.{schema}.stores"
ORDERS_BY_DATE = f"{catalog}.{schema}.orders_by_date"
ORDERS_BY_STORE = f"{catalog}.{schema}.orders_by_store"

row_count = spark.table(ORDERS).count()
print(f"orders table found: {row_count:,} rows (Day 1's optimized + Z-ordered state).")


def timed(label: str, fn):
    t0 = time.time()
    result = fn()
    print(f"{label}: {time.time() - t0:.2f}s")
    return result


def safe_conf(key: str, fallback: str) -> str:
    try:
        return spark.conf.get(key)
    except Exception:
        return fallback


# COMMAND ----------

# MAGIC %md
# MAGIC ## 🖥️ VS Code Spark UI companion — one-time setup
# MAGIC
# MAGIC No Delta needed today — plain PySpark, so this reuses whatever venv you already have
# MAGIC (even `pyspark==4.2.0` from earlier in this project is fine here, unlike Day 1).
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC from pyspark.sql import SparkSession
# MAGIC from pyspark.sql import functions as F
# MAGIC from pyspark.sql.functions import broadcast
# MAGIC import time
# MAGIC
# MAGIC spark = SparkSession.builder.appName("Day2SparkUICompanion").master("local[*]").getOrCreate()
# MAGIC print(f"Spark UI: {spark.sparkContext.uiWebUrl}")
# MAGIC
# MAGIC orders = (
# MAGIC     spark.range(200000)
# MAGIC     .withColumn("store_id", F.when(F.rand() < 0.40, F.lit(101)).otherwise((F.rand()*59+1).cast("int")))
# MAGIC     .withColumn("order_amount", F.round(F.rand()*45+5, 2))
# MAGIC     .withColumn("customer_id", (F.rand()*50000).cast("int"))
# MAGIC     .withColumn("order_date", F.date_sub(F.current_date(), (F.rand()*90).cast("int")))
# MAGIC     .cache()
# MAGIC )
# MAGIC orders.count()   # materialize the cache now
# MAGIC
# MAGIC stores = spark.createDataFrame(
# MAGIC     [(sid, f"Store {sid}", ["South","West","North","East"][sid % 4]) for sid in list(range(1, 60)) + [101]],
# MAGIC     ["store_id", "store_name", "region"],
# MAGIC )
# MAGIC print(f"orders: {orders.count():,} rows | stores: {stores.count()} rows")
# MAGIC ```
# MAGIC Keep this session running throughout — every companion cell below assumes `spark`,
# MAGIC `orders`, and `stores` already exist.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section A — Predicate Pushdown & Column Pruning
# MAGIC
# MAGIC **Predicate pushdown** pushes a filter to the storage engine, discarding rows before
# MAGIC they reach Spark's memory. Source-dependent: Parquet/Delta yes, JSON/XML/text no.
# MAGIC **Column pruning** — only read the columns you select. Filter and select as early as
# MAGIC possible, before any joins.

# COMMAND ----------

sample_customer_id = spark.table(ORDERS).select("customer_id").limit(1).collect()[0].customer_id

# A.1 — Pushdown on a Delta source: look for PushedFilters in the scan node
spark.sql(f"SELECT * FROM {ORDERS} WHERE order_amount > 40").explain(mode="formatted")

# COMMAND ----------

# A.2 — Column pruning: ReadSchema should list 3 columns for narrow vs. 8 for wide
narrow = spark.sql(f"SELECT order_id, order_amount FROM {ORDERS} WHERE order_amount > 40")
wide = spark.sql(f"SELECT * FROM {ORDERS} WHERE order_amount > 40")
narrow.explain(mode="formatted")
wide.explain(mode="formatted")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.3 — Contrast: the same filter against a JSON source
# MAGIC
# MAGIC Using a workspace path — DBFS's `/tmp` isn't available on serverless compute.

# COMMAND ----------

# DBTITLE 1,A.3 — JSON source contrast
# JSON file pre-created by writing from the driver (Spark can't write JSON to
# workspace paths on serverless — distributed write fails, so Python I/O is used).
json_path = f"{workspace_base_path}/temp_json/data.json"

spark.read.json(json_path).filter(F.col("order_amount") > 40).explain(mode="formatted")
# Compare to A.1 — no PushedFilters on the scan node here.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — pushdown vs. no pushdown
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC orders.write.mode("overwrite").parquet("/tmp/orders_parquet_companion")
# MAGIC spark.read.parquet("/tmp/orders_parquet_companion").filter(F.col("order_amount") > 40).explain(mode="formatted")
# MAGIC
# MAGIC orders.write.mode("overwrite").json("/tmp/orders_json_companion")
# MAGIC spark.read.json("/tmp/orders_json_companion").filter(F.col("order_amount") > 40).explain(mode="formatted")
# MAGIC ```
# MAGIC **What to see:** SQL/DataFrame tab, one query per format. The Parquet scan node shows
# MAGIC `PushedFilters`; the JSON scan node doesn't — a `Filter` operator sits above a full
# MAGIC scan instead. Same distinction A.3 makes, now visible as two different node shapes
# MAGIC side by side rather than text output.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section B — Partition Pruning, Dynamic Partition Pruning (DPP), Dynamic File Pruning (DFP)
# MAGIC
# MAGIC Day 1 left `orders` unpartitioned (Databricks' own guidance: don't partition tables
# MAGIC under 1TB). For this lab we build two **explicitly partitioned** copies purely to
# MAGIC demonstrate the pruning mechanics.
# MAGIC
# MAGIC **When to partition a table at all:** only over 1TB, or with a natural, low-cardinality,
# MAGIC frequently-filtered column with roughly balanced partition sizes. Under that, prefer
# MAGIC Z-Order or Liquid Clustering instead.

# COMMAND ----------

spark.sql(f"CREATE TABLE IF NOT EXISTS {ORDERS_BY_DATE} USING DELTA PARTITIONED BY (order_date) AS SELECT * FROM {ORDERS}")
spark.sql(f"CREATE TABLE IF NOT EXISTS {ORDERS_BY_STORE} USING DELTA PARTITIONED BY (store_id) AS SELECT * FROM {ORDERS}")

print(f"{ORDERS_BY_DATE}  -- partitioned by order_date")
print(f"{ORDERS_BY_STORE} -- partitioned by store_id (for the DPP demo)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### B.1 — Static partition pruning

# COMMAND ----------

one_date = spark.table(ORDERS_BY_DATE).select("order_date").limit(1).collect()[0].order_date
spark.sql(f"SELECT * FROM {ORDERS_BY_DATE} WHERE order_date = '{one_date}'").explain(mode="formatted")

# COMMAND ----------

# MAGIC %md
# MAGIC ### B.2 — Dynamic Partition Pruning (DPP)
# MAGIC No configuration needed — on by default, Spark 3.0+. DPP handles what a *static*
# MAGIC filter can't: partitions to skip only knowable from filtering a **joined** dimension
# MAGIC table at runtime.

# COMMAND ----------

spark.sql(f"""
    SELECT o.order_id, o.order_amount, s.store_name
    FROM {ORDERS_BY_STORE} o JOIN {STORES} s ON o.store_id = s.store_id
    WHERE s.region = 'South'
""").explain(mode="formatted")

# COMMAND ----------

# MAGIC %md
# MAGIC ### B.3 — Dynamic File Pruning (DFP)
# MAGIC The file-level sibling of DPP — pruning individual *files* via Delta min/max stats.
# MAGIC You already saw this mechanism on Day 1 §C.2.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — static pruning vs. DPP
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC orders_by_store = orders.write.format("parquet").mode("overwrite").partitionBy("store_id")
# MAGIC orders.write.mode("overwrite").partitionBy("store_id").parquet("/tmp/orders_by_store_companion")
# MAGIC orders_by_store_df = spark.read.parquet("/tmp/orders_by_store_companion")
# MAGIC
# MAGIC # Static pruning
# MAGIC orders_by_store_df.filter(F.col("store_id") == 101).explain(mode="formatted")
# MAGIC
# MAGIC # DPP — partitions to skip only known after filtering the joined `stores` side
# MAGIC orders_by_store_df.join(stores, "store_id").filter(F.col("region") == "South").explain(mode="formatted")
# MAGIC ```
# MAGIC **What to see:** SQL/DataFrame tab for the DPP query — a `PartitionFilters` entry
# MAGIC containing a subquery that reads from the `stores` side. That subquery *is* DPP,
# MAGIC computed at runtime — nothing like it appears in the static-pruning query's plan.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section C — Table Statistics
# MAGIC
# MAGIC `ANALYZE TABLE` / `COMPUTE STATISTICS` collects column- and table-level statistics
# MAGIC (row counts, distinct counts, min/max, null counts) that Spark's **cost-based
# MAGIC optimizer** uses for decisions like join reordering and — directly relevant to
# MAGIC Day 1 §F and Section F below — confirming whether a table is actually small enough
# MAGIC to broadcast, rather than relying only on file size on disk.
# MAGIC
# MAGIC **When to run this:** after a significant load into a table you'll query repeatedly,
# MAGIC especially before relying on automatic join-strategy decisions for a multi-way join —
# MAGIC stale or missing stats can make the optimizer pick a worse plan than the data would
# MAGIC actually support.

# COMMAND ----------

spark.sql(f"ANALYZE TABLE {STORES} COMPUTE STATISTICS FOR ALL COLUMNS")
display(spark.sql(f"DESCRIBE EXTENDED {STORES}").where("col_name = 'Statistics'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — table statistics
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC stores.write.mode("overwrite").saveAsTable("stores_stats_companion")
# MAGIC spark.sql("ANALYZE TABLE stores_stats_companion COMPUTE STATISTICS FOR ALL COLUMNS")
# MAGIC spark.sql("DESCRIBE EXTENDED stores_stats_companion").show(50, truncate=False)
# MAGIC ```
# MAGIC **What to see:** the `ANALYZE TABLE` job itself is a full scan — Jobs/Stages tab shows
# MAGIC read I/O proportional to table size, no shuffle. There's no dramatic *before/after*
# MAGIC visual here the way OPTIMIZE has one; the payoff shows up **indirectly**, in better
# MAGIC plan choices on later queries — worth saying explicitly to students, since this is the
# MAGIC one technique in the course where "nothing visibly happens in the UI, and that's fine"
# MAGIC is itself the lesson.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section D — Predictive Optimization
# MAGIC *(Databricks-managed, Unity Catalog only — conceptual, same as Day 1's Deletion
# MAGIC Vectors/Predictive I/O section — no VS Code companion possible)*
# MAGIC
# MAGIC Predictive Optimization (PO) automatically runs `OPTIMIZE`, `VACUUM`, and `ANALYZE`
# MAGIC on Unity Catalog managed tables, on Databricks' own serverless compute, based on
# MAGIC observed usage. **Never runs `ZORDER`** — on Z-ordered tables it just skips
# MAGIC already-clustered files during compaction. With Automatic Liquid Clustering enabled,
# MAGIC PO can pick new clustering keys itself (the managed counterpart to Day 1's
# MAGIC `CLUSTER BY AUTO`). VACUUM under PO still respects the same 7-day retention default.
# MAGIC
# MAGIC **Eligibility:** UC managed tables, Premium plan, supported region. Default-on for
# MAGIC accounts created on/after Nov 11, 2024.
# MAGIC
# MAGIC PO runs as background serverless jobs — it has no representation in any cluster's
# MAGIC Spark UI, classic or otherwise. Check Catalog Explorer's table History tab instead.

# COMMAND ----------

spark.sql(f"ALTER TABLE {ORDERS} ENABLE PREDICTIVE OPTIMIZATION")
display(spark.sql(f"DESCRIBE TABLE EXTENDED {ORDERS}"))

# COMMAND ----------

spark.sql(f"ALTER TABLE {ORDERS} INHERIT PREDICTIVE OPTIMIZATION")
print(f"{ORDERS} now inherits from schema/catalog/account.")
print(f"To see WHY PO skipped/ran an op (DBR 18 LTS+): DESCRIBE TABLE EXTENDED {ORDERS} AS JSON")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section E — Shuffle, Repartition/Coalesce & AQE
# MAGIC
# MAGIC A shuffle happens on every **wide transformation** — joins, `groupBy`, window
# MAGIC functions, `repartition()`. Almost every technique in this course exists partly to
# MAGIC avoid one.
# MAGIC
# MAGIC | Setting | Default |
# MAGIC |---|---|
# MAGIC | `spark.sql.shuffle.partitions` | **200** |
# MAGIC | Target size per shuffle task | **128–200MB** |
# MAGIC | AQE Auto-Optimized Shuffle initial per-partition size | **128MB** |
# MAGIC
# MAGIC **On this serverless environment:** AQE and Auto-Optimized Shuffle are **on by
# MAGIC default** — no `spark.conf.set(...)` needed for either.

# COMMAND ----------

shuffle_partitions = safe_conf('spark.sql.shuffle.partitions', 'managed automatically on serverless')
aqe_enabled = safe_conf('spark.sql.adaptive.enabled', 'true (default on serverless)')
coalesce_enabled = safe_conf('spark.sql.adaptive.coalescePartitions.enabled', 'true (default on serverless)')
print(f"shuffle.partitions = {shuffle_partitions}   |   adaptive.enabled = {aqe_enabled}   |   coalescePartitions.enabled = {coalesce_enabled}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### E.1 — Trigger a shuffle-heavy aggregation

# COMMAND ----------

store_agg = spark.table(ORDERS).groupBy("store_id").agg(
    F.sum("order_amount").alias("total_revenue"), F.count("*").alias("num_orders"), F.avg("order_amount").alias("avg_order_value"),
)
store_agg_result = timed(f"E.1 groupBy(store_id) across {row_count:,} rows", store_agg.collect)
print(f"Store groups: {len(store_agg_result)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### E.2 — Repartition vs. Coalesce
# MAGIC
# MAGIC Both change a DataFrame's partition count, but they are not opposites in cost:
# MAGIC
# MAGIC | | `repartition(n)` | `coalesce(n)` |
# MAGIC |---|---|---|
# MAGIC | Direction | Any → any (typically used to increase) | Only decreases |
# MAGIC | Shuffle? | **Yes — full shuffle** | **No shuffle** — merges existing partitions |
# MAGIC | Result balance | Even | Uneven if source partitions were uneven |
# MAGIC | When to use | Need more parallelism, or fixing skewed partitions | Reducing output file count before a write, cheaply |
# MAGIC
# MAGIC **When to use which:** `coalesce()` after a heavily-filtered DataFrame, right before
# MAGIC a write, to avoid writing hundreds of tiny output files — it's nearly free since it
# MAGIC avoids a shuffle. `repartition()` when you actually need more partitions than you
# MAGIC currently have, or need to redistribute skewed data evenly — accept the shuffle cost
# MAGIC because the alternative (uneven work per task) is worse.

# COMMAND ----------

# DBTITLE 1,E.2 — Repartition vs. Coalesce
def num_partitions(df):
    """Get partition count without using rdd (not supported on Spark Connect)."""
    return df.select(F.spark_partition_id().alias("pid")).distinct().count()

filtered = spark.table(ORDERS).filter(F.col("order_amount") > 40)
print(f"Filtered DataFrame partitions: {num_partitions(filtered)}")

coalesced = filtered.coalesce(4)
print(f"After coalesce(4): {num_partitions(coalesced)} partitions")
coalesced.write.mode("overwrite").format("noop").save()   # trigger execution without materializing output

repartitioned = filtered.repartition(4)
print(f"After repartition(4): {num_partitions(repartitioned)} partitions")
repartitioned.write.mode("overwrite").format("noop").save()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — shuffle, and repartition vs. coalesce
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC filtered = orders.filter(F.col("order_amount") > 40)
# MAGIC print(f"Before: {filtered.rdd.getNumPartitions()} partitions")
# MAGIC
# MAGIC coalesced = filtered.coalesce(4)
# MAGIC coalesced.write.mode("overwrite").format("noop").save()
# MAGIC
# MAGIC repartitioned = filtered.repartition(4)
# MAGIC repartitioned.write.mode("overwrite").format("noop").save()
# MAGIC ```
# MAGIC **What to see:** open both jobs on the Jobs tab. The `coalesce` job's stage shows
# MAGIC **no `Exchange` node** on the SQL tab and near-zero shuffle read/write in Stages —
# MAGIC it just merges existing partitions in place. The `repartition` job's stage **does**
# MAGIC show an `Exchange` node and real shuffle bytes written — same target partition count,
# MAGIC completely different cost, exactly the distinction the table above makes on paper.

# COMMAND ----------

# MAGIC %md
# MAGIC ### E.3 — Seeing AQE actually happen, on the same DataFrame
# MAGIC
# MAGIC `.explain()` called *before* execution only shows AQE's initial plan
# MAGIC (`isFinalPlan=false`). Call `.explain()` again on the **same** DataFrame *after*
# MAGIC triggering execution, and Spark shows the plan it **actually ran**
# MAGIC (`isFinalPlan=true`).

# COMMAND ----------

aqe_query = (
    spark.table(ORDERS).alias("o")
    .join(spark.table(STORES).alias("s"), "store_id")
    .groupBy("s.region")
    .agg(F.sum("o.order_amount").alias("region_revenue"))
)

print("BEFORE execution:")
aqe_query.explain(mode="formatted")

aqe_query.collect()

print("\nAFTER execution (same DataFrame):")
aqe_query.explain(mode="formatted")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — AQE before/after
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC aqe_query = orders.join(stores, "store_id").groupBy("region").agg(F.sum("order_amount").alias("region_revenue"))
# MAGIC print("BEFORE:"); aqe_query.explain(mode="formatted")
# MAGIC aqe_query.collect()
# MAGIC print("AFTER:"); aqe_query.explain(mode="formatted")
# MAGIC ```
# MAGIC **What to see:** SQL/DataFrame tab, this query, after it runs. Look for an
# MAGIC **`AQEShuffleRead`** node in place of a plain `Exchange` — partition coalescing. Also
# MAGIC check the join node type: this exact unhinted-join pattern has previously converted
# MAGIC to `BroadcastHashJoin` at runtime with no explicit hint in the code at all — confirmed
# MAGIC in an earlier smoke test of this pattern.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section F — Data Skew: Identification & Salting
# MAGIC
# MAGIC `orders` has a deliberate skew since Day 1: `store_id = 101` carries roughly 40% of
# MAGIC all rows.
# MAGIC
# MAGIC **On this serverless environment:** join strategy is automatically managed — you
# MAGIC cannot force `autoBroadcastJoinThreshold = -1` the way classic compute allows. Skew is
# MAGIC still fully observable in task durations regardless of which join strategy Spark
# MAGIC actually picks.

# COMMAND ----------

# MAGIC %md
# MAGIC ### F.1 — Confirm and quantify the skew

# COMMAND ----------

store_counts = spark.table(ORDERS).groupBy("store_id").count().orderBy(F.desc("count"))
display(store_counts.limit(5))

top_row = store_counts.first()
skew_ratio = top_row["count"] / ((row_count - top_row["count"]) / 59)
print(f"Store {top_row.store_id}: {top_row['count']:,} rows ({top_row['count'] / row_count * 100:.1f}% of the table), "
      f"skew ratio vs. average other store: {skew_ratio:.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ### F.2 — Watch the skew hurt a real join

# COMMAND ----------

skewed_join = (
    spark.table(ORDERS).alias("o").join(spark.table(STORES).alias("s"), "store_id")
    .groupBy("s.region").agg(F.sum("o.order_amount").alias("region_revenue"))
)
skewed_result = timed("F.2 skewed join+aggregate", skewed_join.collect)

# COMMAND ----------

# MAGIC %md
# MAGIC ### F.3 — Try the built-in remedies first
# MAGIC Escalation order: filter the skewed value if possible → skew hints → AQE automatic
# MAGIC skew handling → isolate-and-broadcast the hot key → salting **last**.
# MAGIC
# MAGIC AQE's automatic skew-join optimization needs **both**:
# MAGIC - `spark.sql.adaptive.skewJoin.skewedPartitionFactor` — **5x** the median partition
# MAGIC - `spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes` — **256MB** minimum

# COMMAND ----------

# MAGIC %md
# MAGIC ### F.4 — Salting, step by step

# COMMAND ----------

SALT_BUCKETS = 8

orders_salted = spark.table(ORDERS).withColumn("salt", (F.rand() * SALT_BUCKETS).cast("int"))
stores_salted = spark.table(STORES).crossJoin(spark.range(SALT_BUCKETS).withColumnRenamed("id", "salt"))

salted_join = (
    orders_salted.alias("o").join(stores_salted.alias("s"), on=["store_id", "salt"])
    .groupBy("s.region").agg(F.sum("o.order_amount").alias("region_revenue"))
)
salted_result = timed(f"F.4 salted join+aggregate (SALT_BUCKETS={SALT_BUCKETS})", salted_join.collect)

unsalted_totals = {r.region: r.region_revenue for r in skewed_result}
salted_totals = {r.region: r.region_revenue for r in salted_result}
mismatch = [r for r in unsalted_totals if abs(unsalted_totals[r] - salted_totals[r]) >= 0.01]
print("Totals match F.2 exactly." if not mismatch else f"MISMATCH: {mismatch} — investigate")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — skew, before/after salting
# MAGIC
# MAGIC This is the one companion worth running even if you already ran Day 1's shared
# MAGIC skew demo — Free Edition can't force a real shuffle join the way local compute can,
# MAGIC so this is where the skew signature is cleanest.
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC skewed = orders.join(stores, "store_id").groupBy("region").agg(F.sum("order_amount").alias("region_revenue"))
# MAGIC skewed.collect()
# MAGIC
# MAGIC SALT_BUCKETS = 8
# MAGIC orders_salted = orders.withColumn("salt", (F.rand()*SALT_BUCKETS).cast("int"))
# MAGIC stores_salted = stores.crossJoin(spark.range(SALT_BUCKETS).withColumnRenamed("id", "salt"))
# MAGIC salted = orders_salted.join(stores_salted, on=["store_id","salt"]).groupBy("region").agg(F.sum("order_amount").alias("region_revenue"))
# MAGIC salted.collect()
# MAGIC ```
# MAGIC **What to see:** open both jobs' shuffle stages on the **Stages tab**, sort the task
# MAGIC table by Duration. The skewed job shows one or two dramatically longer tasks
# MAGIC (store 101). The salted job's task durations should be far more even — same total
# MAGIC work, spread across more tasks instead of piling onto one.

# COMMAND ----------

# MAGIC %md
# MAGIC ### F.5 — Partial salting for a skewed aggregation (no join involved)

# COMMAND ----------

two_stage_agg = (
    spark.table(ORDERS)
    .withColumn("salt", (F.rand() * SALT_BUCKETS).cast("int"))
    .groupBy("store_id", "salt")
    .agg(F.sum("order_amount").alias("partial_revenue"), F.count("*").alias("partial_count"))
    .groupBy("store_id")
    .agg(F.sum("partial_revenue").alias("total_revenue"), F.sum("partial_count").alias("total_orders"))
)
two_stage_result = timed("F.5 two-stage salted aggregation", lambda: two_stage_agg.orderBy(F.desc("total_revenue")).collect())
display(spark.createDataFrame(two_stage_result[:5]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Day 2 Recap — Full Defaults & Decision Cheat Sheet
# MAGIC
# MAGIC | Area | Setting | Default / when to use |
# MAGIC |---|---|---|
# MAGIC | Shuffle | Default partitions | 200 (`spark.sql.shuffle.partitions`) |
# MAGIC | Shuffle | Target size/task | 128–200MB |
# MAGIC | Repartition vs. Coalesce | Which shuffles | `repartition` always does; `coalesce` never does |
# MAGIC | Repartition vs. Coalesce | When to use | `coalesce` before a write to cut file count cheaply; `repartition` for more parallelism or fixing skew |
# MAGIC | Skew (AQE) | Partition factor / size threshold | 5x median / 256MB |
# MAGIC | Skew fix order | — | filter → hint → AQE auto → isolate & broadcast hot key → salting (last) |
# MAGIC | DPP | Enabled since | Spark 3.0+ |
# MAGIC | DFP | Enabled since | DBR 6.1+ |
# MAGIC | Table statistics | Command | `ANALYZE TABLE ... COMPUTE STATISTICS FOR ALL COLUMNS` |
# MAGIC | Predictive Optimization | Auto-enabled since | accounts created on/after Nov 11, 2024 |
# MAGIC | Predictive Optimization | Runs ZORDER? | No — compaction, VACUUM, ANALYZE only |
# MAGIC | Broadcast join | Static / AQE thresholds | 10MB / 30MB; never > 1GB disk, 8GB in-memory hard cap |
# MAGIC
# MAGIC **This environment's specific adaptations, worth remembering if you move to classic
# MAGIC compute:** AQE/AOS are on by default here but must be explicitly enabled on classic
# MAGIC compute in some configurations; forcing `autoBroadcastJoinThreshold = -1` works on
# MAGIC classic compute but not here; DBFS `/tmp` works on classic compute but not here.
