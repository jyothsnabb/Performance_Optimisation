# Databricks notebook source
# MAGIC %md
# MAGIC # Day 1 — Storage Layer Optimization
# MAGIC ### OPTIMIZE · Z-ORDER · Liquid Clustering (incl. AUTO) · VACUUM · Broadcast Joins · Deletion Vectors & Predictive I/O · Clones
# MAGIC
# MAGIC **Domain:** QuickBite QSR — a multi-store food & beverage chain. We build one
# MAGIC `orders` table deliberately as a pile of tiny files, and one small `stores`
# MAGIC dimension table. Every technique fixes something real and visible in this dataset.
# MAGIC
# MAGIC **Runs on Databricks Free Edition serverless compute.** That has real consequences
# MAGIC for how a few sections work, confirmed by actually running this notebook:
# MAGIC - Managed Unity Catalog tables don't expose a physical file path — `dbutils.fs.ls`
# MAGIC   and reading raw `_delta_log/*.json` files **don't work**. Every inspection in this
# MAGIC   notebook goes through `DESCRIBE DETAIL` / `DESCRIBE HISTORY` instead — which,
# MAGIC   usefully, also works identically on classic compute, so nothing here is a
# MAGIC   downgrade if you do have a classic cluster.
# MAGIC - `VACUUM` enforces a **hard 168-hour (7-day) minimum retention with no override** —
# MAGIC   stronger than classic compute's configurable safety check. You cannot force a
# MAGIC   visible deletion in this environment. Section E's VS Code companion is the only
# MAGIC   place in this course where you actually watch VACUUM delete a file.
# MAGIC - **Free Edition has no Spark UI at all** (serverless-only compute). Every section
# MAGIC   below has a **VS Code Spark UI companion** — small, fast, self-contained code you
# MAGIC   copy into a local session to see the real Jobs/Stages/SQL tabs for that exact
# MAGIC   technique, with what to expect **before and after**.
# MAGIC
# MAGIC **Reference:** Databricks, *"Comprehensive Guide to Optimize Databricks, Spark and
# MAGIC Delta Lake Workloads"*, plus current Delta Lake / Unity Catalog documentation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Unity Catalog catalog")
dbutils.widgets.text("schema", "optimization_demo", "Schema (will be created)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from functools import reduce
import time

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

ORDERS = f"{catalog}.{schema}.orders"
STORES = f"{catalog}.{schema}.stores"
ORDERS_LC = f"{catalog}.{schema}.orders_lc"
ORDERS_LC_AUTO = f"{catalog}.{schema}.orders_lc_auto"
ORDERS_CLONE = f"{catalog}.{schema}.orders_shallow_clone"

print(f"catalog.schema = {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Two small helpers we reuse all day

# COMMAND ----------

snapshots = []
timings = []


def snapshot(table_name: str, stage_label: str):
    """Capture numFiles / size for table_name via DESCRIBE DETAIL and label it."""
    row = (
        spark.sql(f"DESCRIBE DETAIL {table_name}")
        .select("numFiles", "sizeInBytes")
        .withColumn("avg_file_size_mb", F.round(F.col("sizeInBytes") / F.col("numFiles") / 1024 / 1024, 3))
        .withColumn("stage", F.lit(stage_label))
        .select("stage", "numFiles", "sizeInBytes", "avg_file_size_mb")
    )
    snapshots.append(row)
    return row


def show_snapshots():
    display(reduce(lambda a, b: a.unionByName(b), snapshots))


def timed(label: str, fn):
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    timings.append((label, elapsed))
    print(f"{label}: {elapsed:.2f}s")
    return result


def show_timings():
    display(spark.createDataFrame(timings, ["stage", "elapsed_seconds"]))


def safe_conf(key: str, fallback: str) -> str:
    """spark.conf.get() isn't accessible for every key on serverless — fall back cleanly."""
    try:
        return spark.conf.get(key)
    except Exception:
        return fallback


# COMMAND ----------

# MAGIC %md
# MAGIC ## 🖥️ VS Code Spark UI companion — one-time setup
# MAGIC
# MAGIC Every section below has its own companion cell, but they all share **one local
# MAGIC session** you set up once. Full install steps: `VS_Code_Spark_UI_Setup.md`.
# MAGIC
# MAGIC **In a fresh venv (do NOT reuse a venv that already has `pyspark==4.2.0` — see the
# MAGIC note below):**
# MAGIC ```bash
# MAGIC python3 -m venv venv && source venv/bin/activate
# MAGIC pip install pyspark==3.5.3 delta-spark==3.2.1 pandas ipython
# MAGIC ```
# MAGIC This exact pair is confirmed to install and import cleanly together. The newest
# MAGIC `pyspark`/`delta-spark` releases have an active, unresolved version-compatibility
# MAGIC bug between them — this older pinned pair sidesteps it entirely, and comfortably
# MAGIC supports every command used today (`ZORDER` since Delta 2.0, `CLUSTER BY` since
# MAGIC Delta 3.1, `SHALLOW CLONE` is OSS-supported; `DEEP CLONE`, Deletion Vectors, and
# MAGIC Predictive I/O are Databricks-managed only — flagged where they come up).
# MAGIC
# MAGIC **Paste this once into a new `.py` file in VS Code** (`# %%` cells — see the setup
# MAGIC guide for how to run them one at a time):
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC import os, shutil, time
# MAGIC from pyspark.sql import SparkSession
# MAGIC from pyspark.sql import functions as F
# MAGIC from pyspark.sql.functions import broadcast
# MAGIC from delta import configure_spark_with_delta_pip
# MAGIC
# MAGIC BASE_DIR = os.path.expanduser("~/spark-ui-test/day1_companion")
# MAGIC os.makedirs(BASE_DIR, exist_ok=True)
# MAGIC # Uncomment for a completely fresh start:
# MAGIC # shutil.rmtree(BASE_DIR, ignore_errors=True); os.makedirs(BASE_DIR, exist_ok=True)
# MAGIC
# MAGIC builder = (
# MAGIC     SparkSession.builder.appName("Day1SparkUICompanion")
# MAGIC     .master("local[*]")
# MAGIC     .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
# MAGIC     .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
# MAGIC )
# MAGIC spark = configure_spark_with_delta_pip(builder).getOrCreate()
# MAGIC print(f"Spark UI: {spark.sparkContext.uiWebUrl}")
# MAGIC
# MAGIC ORDERS_PATH = f"{BASE_DIR}/orders"
# MAGIC STORES_PATH = f"{BASE_DIR}/stores"
# MAGIC ORDERS = f"delta.`{ORDERS_PATH}`"
# MAGIC STORES = f"delta.`{STORES_PATH}`"
# MAGIC
# MAGIC # A SMALL dataset on purpose — this companion is about seeing the Spark UI signal
# MAGIC # for each technique quickly, not replicating Databricks' full-scale numbers.
# MAGIC store_rows = [(sid, f"Store {sid}", ["South","West","North","East"][sid % 4]) for sid in list(range(1, 60)) + [101]]
# MAGIC spark.createDataFrame(store_rows, ["store_id", "store_name", "region"]).write.format("delta").mode("overwrite").save(STORES_PATH)
# MAGIC
# MAGIC def make_batch(batch_id, n=2000):
# MAGIC     return (
# MAGIC         spark.range(n)
# MAGIC         .withColumn("order_id", F.concat(F.lit(f"B{batch_id:03d}-"), F.col("id").cast("string")))
# MAGIC         .withColumn("store_id", F.when(F.rand() < 0.40, F.lit(101)).otherwise((F.rand()*59+1).cast("int")))
# MAGIC         .withColumn("customer_id", (F.rand()*50000).cast("int"))
# MAGIC         .withColumn("order_amount", F.round(F.rand()*45+5, 2))
# MAGIC         .drop("id")
# MAGIC     )
# MAGIC
# MAGIC t0 = time.time()
# MAGIC for b in range(25):   # 25 small batches ~ 50,000 rows, seconds not minutes
# MAGIC     make_batch(b).write.format("delta").mode("append").save(ORDERS_PATH)
# MAGIC print(f"Built local orders table: {time.time()-t0:.1f}s, {spark.read.format('delta').load(ORDERS_PATH).count():,} rows")
# MAGIC ```
# MAGIC
# MAGIC Keep this session running (same Interactive Window / kernel) as you work through the
# MAGIC companion cells below — each one assumes `spark`, `ORDERS`, `ORDERS_PATH`, `STORES`,
# MAGIC `STORES_PATH`, and `make_batch` already exist.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section A — The Small Files Problem
# MAGIC
# MAGIC Underneath every Delta table are plain Parquet files plus a `_delta_log` of
# MAGIC JSON/checkpoint transaction logs. Query performance is very sensitive to Parquet
# MAGIC file size: too many tiny files means Spark spends more time *opening and closing*
# MAGIC files than reading data. Healthy range: **16MB–1GB per file**.
# MAGIC
# MAGIC We simulate the classic cause: frequent small appends, no file-size tuning. We also
# MAGIC bake in a deliberate skew — `store_id = 101` ("Flagship — MG Road") gets ~40% of all
# MAGIC orders — used later for the skew lab in Day 2.

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.1 — Build the `stores` dimension table (small, 60 stores)

# COMMAND ----------

CITIES = ["Bengaluru", "Mumbai", "Delhi", "Pune", "Chennai", "Hyderabad", "Kolkata", "Ahmedabad"]
REGIONS = ["South", "West", "North", "East"]
STORE_TYPES = ["Dine-in", "Drive-thru", "Kiosk", "Delivery-only"]

store_rows = [
    (sid, f"Store {sid}" if sid != 101 else "Flagship - MG Road",
     CITIES[sid % len(CITIES)], REGIONS[sid % len(REGIONS)], STORE_TYPES[sid % len(STORE_TYPES)])
    for sid in list(range(1, 60)) + [101]
]

(
    spark.createDataFrame(store_rows, ["store_id", "store_name", "city", "region", "store_type"])
    .write.format("delta").mode("overwrite").saveAsTable(STORES)
)
display(spark.table(STORES).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.2 — Build the `orders` fact table as many small, unoptimized appends
# MAGIC
# MAGIC 200 append batches x ~5,000 rows = ~1,000,000 rows, each its own commit. **Takes a
# MAGIC few minutes.**

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {ORDERS} (
  order_id STRING, store_id INT, customer_id INT, order_date DATE,
  order_amount DOUBLE, item_count INT, product_category STRING, payment_type STRING
)
USING DELTA
""")

CATEGORIES = ["Burgers", "Fried Chicken", "Beverages", "Desserts", "Sides", "Breakfast"]
PAYMENT_TYPES = ["Card", "UPI", "Cash", "Wallet"]
NUM_BATCHES = 200
ROWS_PER_BATCH = 5000
FLAGSHIP_STORE_ID = 101
FLAGSHIP_WEIGHT = 0.40
NUM_CUSTOMERS = 500000


def make_batch(batch_id: int):
    return (
        spark.range(ROWS_PER_BATCH)
        .withColumn("order_id", F.concat(F.lit(f"B{batch_id:04d}-"), F.col("id").cast("string")))
        .withColumn("store_id", F.when(F.rand() < FLAGSHIP_WEIGHT, F.lit(FLAGSHIP_STORE_ID)).otherwise((F.rand()*59+1).cast("int")))
        .withColumn("customer_id", (F.rand() * NUM_CUSTOMERS).cast("int"))
        .withColumn("order_date", F.date_sub(F.current_date(), (F.rand() * 90).cast("int")))
        .withColumn("order_amount", F.round(F.rand() * 45 + 5, 2))
        .withColumn("item_count", (F.rand() * 5 + 1).cast("int"))
        .withColumn("product_category", F.element_at(F.array(*[F.lit(c) for c in CATEGORIES]), (F.rand()*len(CATEGORIES)).cast("int")+1))
        .withColumn("payment_type", F.element_at(F.array(*[F.lit(p) for p in PAYMENT_TYPES]), (F.rand()*len(PAYMENT_TYPES)).cast("int")+1))
        .drop("id")
    )


start = time.time()
for batch in range(NUM_BATCHES):
    make_batch(batch).write.format("delta").mode("append").saveAsTable(ORDERS)
    if batch % 25 == 0:
        print(f"  batch {batch:>3}/{NUM_BATCHES}  ({time.time() - start:6.1f}s elapsed)")

print(f"Done in {time.time() - start:.1f}s. Row count: {spark.table(ORDERS).count():,}")
snapshot(ORDERS, "A - before any optimization")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.3 — Inspect the damage
# MAGIC
# MAGIC For managed Unity Catalog tables, the physical file location isn't exposed —
# MAGIC `DESCRIBE DETAIL` is not a fallback here, it's the **only** tool, and it's all you
# MAGIC need: `numFiles`, `sizeInBytes`, and the derived average tell the whole story.

# COMMAND ----------

detail = spark.sql(f"DESCRIBE DETAIL {ORDERS}").select("numFiles", "sizeInBytes").collect()[0]
avg_file_size_kb = detail.sizeInBytes / detail.numFiles / 1024 if detail.numFiles else 0

print(f"{detail.numFiles} physical parquet files, {detail.sizeInBytes/1024/1024:.1f} MB total")
print(f"Average file size: {avg_file_size_kb:.1f} KB")
print(f"Target range: 16,384 KB (16MB) to 1,048,576 KB (1GB) per file")
print(f"These files are {'BELOW' if avg_file_size_kb < 16384 else 'within'} the healthy range.")

display(spark.sql(f"DESCRIBE DETAIL {ORDERS}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — small files & the Environment tab
# MAGIC
# MAGIC Free Edition has no Environment tab to show active Spark confs live — this is
# MAGIC exactly where a local session earns its keep.
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC detail = spark.sql(f"DESCRIBE DETAIL {ORDERS}").select("numFiles", "sizeInBytes").collect()[0]
# MAGIC print(f"{detail.numFiles} files, {detail.sizeInBytes/1024:.0f} KB total")
# MAGIC ```
# MAGIC **What to see in `localhost:4040`:** open the **Environment tab** — every active
# MAGIC Spark conf, live, with no code at all. This is the tab to reach for instead of
# MAGIC printing `spark.conf.get(...)` anywhere in this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.4 — Understanding the Delta Transaction Log (`_delta_log`)
# MAGIC
# MAGIC Every technique today writes to the same structure: `_delta_log`, sitting next to
# MAGIC the Parquet files. One JSON file per table **version**; each is a sequence of
# MAGIC **actions** — `add` (a file was added, with min/max `stats`), `remove` (logically
# MAGIC dropped, physically kept until `VACUUM`), `commitInfo` (the operation + its metrics
# MAGIC — exactly what `DESCRIBE HISTORY` reads and formats). Nothing mutates in place —
# MAGIC every operation appends a new commit, which is what gives Delta ACID transactions and
# MAGIC time travel. Commits collapse into a checkpoint every 10 by default
# MAGIC (`delta.checkpointInterval`).
# MAGIC
# MAGIC On managed UC tables we can't read these JSON files directly — `DESCRIBE HISTORY`
# MAGIC reads the exact same `commitInfo` action and formats it as a table.

# COMMAND ----------

history_df = spark.sql(f"DESCRIBE HISTORY {ORDERS}")
total_commits = history_df.count()
print(f"{total_commits} total commits (versions). Checkpoint every 10 by default -> ~{total_commits // 10} expected.")

print("\nMost recent commit (our last append batch):")
display(
    history_df.orderBy(F.desc("version")).limit(1)
    .select("version", "timestamp", "operation", "operationParameters", "operationMetrics")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### A.5 — Baseline query performance (before optimization)

# COMMAND ----------

sample_customer_id = spark.table(ORDERS).select("customer_id").limit(1).collect()[0].customer_id
print(f"Sample customer_id for our running filter query: {sample_customer_id}")

baseline_count = timed(
    "A.5 baseline (before optimize/zorder)",
    lambda: spark.sql(f"SELECT * FROM {ORDERS} WHERE customer_id = {sample_customer_id}").count(),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — baseline scan
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC sample_customer_id = spark.read.format("delta").load(ORDERS_PATH).select("customer_id").limit(1).collect()[0].customer_id
# MAGIC spark.sql(f"SELECT * FROM {ORDERS} WHERE customer_id = {sample_customer_id}").count()
# MAGIC ```
# MAGIC **Before you run C's companion below**, open `localhost:4040` → **SQL/DataFrame
# MAGIC tab** → click this query → expand the Delta scan node → **write down "number of
# MAGIC files read"**. This is your BEFORE number — Z-ORDER's companion gives you the AFTER.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section B — OPTIMIZE (Compaction / Bin-Packing)
# MAGIC
# MAGIC | Command | Default target file size |
# MAGIC |---|---|
# MAGIC | Manual `OPTIMIZE` | **1 GB** (`delta.targetFileSize`) |
# MAGIC | Auto Optimize (Databricks-managed) | **128 MB** |
# MAGIC
# MAGIC **Auto Optimize** has two independent parts: **Optimize Write** reshapes partition
# MAGIC sizes *during* the write; **Auto Compact** runs as a *separate* job right after.
# MAGIC
# MAGIC **Production guidance:** run `OPTIMIZE` on its own job, not inside the ingestion job.

# COMMAND ----------

optimize_result = spark.sql(f"OPTIMIZE {ORDERS}")
display(optimize_result.select("metrics.*"))

snapshot(ORDERS, "B - after OPTIMIZE")
show_snapshots()

# COMMAND ----------

# MAGIC %md
# MAGIC ### B.1 — Cross-check with `DESCRIBE HISTORY`

# COMMAND ----------

display(
    spark.sql(f"DESCRIBE HISTORY {ORDERS}")
    .where("operation = 'OPTIMIZE'").orderBy(F.desc("version")).limit(1)
    .select("version", "timestamp", "operationParameters.zOrderBy",
            "operationMetrics.numRemovedFiles", "operationMetrics.numAddedFiles",
            "operationMetrics.numRemovedBytes", "operationMetrics.numAddedBytes")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — OPTIMIZE
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC before = spark.sql(f"DESCRIBE DETAIL {ORDERS}").select("numFiles").collect()[0].numFiles
# MAGIC spark.sql(f"OPTIMIZE {ORDERS}")
# MAGIC after = spark.sql(f"DESCRIBE DETAIL {ORDERS}").select("numFiles").collect()[0].numFiles
# MAGIC print(f"numFiles: {before} -> {after}")
# MAGIC ```
# MAGIC **What to see — BEFORE:** `localhost:4040` → **Jobs tab** is either empty or shows
# MAGIC only the 25 small append-batch jobs from setup, each tiny.
# MAGIC **AFTER running this cell:** a new job appears. Click it → **Stages tab** → almost
# MAGIC entirely **file I/O time**, close to **zero shuffle read/write** — compaction
# MAGIC rewrites files, it never redistributes rows by key. Compare this stage's shape
# MAGIC against Section F's broadcast join later, which looks completely different.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section C — Z-ORDER
# MAGIC
# MAGIC `ZORDER BY` physically co-locates rows with similar values, so Delta's min/max
# MAGIC data-skipping stats (first 32 columns, automatic) become far more selective.
# MAGIC
# MAGIC **Column selection:** high-cardinality, filter/join columns — `customer_id` (500,000
# MAGIC distinct values) is textbook. **Never Z-ORDER on more than 4 columns.**
# MAGIC
# MAGIC > Databricks recommends *not* partitioning tables under 1TB — why `orders` stays
# MAGIC > unpartitioned here. Day 2 revisits partitioning explicitly for the DPP/DFP lab.

# COMMAND ----------

zorder_result = spark.sql(f"OPTIMIZE {ORDERS} ZORDER BY (customer_id)")
display(zorder_result.select("metrics.*"))

snapshot(ORDERS, "C - after OPTIMIZE ZORDER BY")
show_snapshots()

# COMMAND ----------

# MAGIC %md
# MAGIC ### C.1 — ZCube stats from `DESCRIBE HISTORY`
# MAGIC
# MAGIC On managed UC tables we can't read per-file `ZCUBE_ID` tags from the raw log — but
# MAGIC `DESCRIBE HISTORY`'s `operationMetrics` gives the aggregate picture: how many files
# MAGIC were already in a ZCube (`totalConsideredFiles` vs. skipped) before this run touched
# MAGIC them at all — the mechanism behind **incremental Z-Ordering**.

# COMMAND ----------

zorder_history = (
    spark.sql(f"DESCRIBE HISTORY {ORDERS}")
    .where("operation = 'OPTIMIZE' AND operationParameters.zOrderBy is not null")
    .orderBy(F.desc("version")).limit(1)
)
display(
    zorder_history.select(
        "version", "timestamp", "operationParameters.zOrderBy",
        "operationMetrics.numAddedFiles", "operationMetrics.numRemovedFiles",
        "operationMetrics.totalConsideredFiles", "operationMetrics.totalFilesSkipped",
    )
)
print("Try it live: re-run OPTIMIZE ... ZORDER BY (customer_id) with no new data written —")
print("totalConsideredFiles should drop near zero, since files are already ZCube-tagged.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### C.2 — Data skipping: does the same filter query read fewer files now?

# COMMAND ----------

after_zorder_count = timed(
    "C.2 after optimize + zorder",
    lambda: spark.sql(f"SELECT * FROM {ORDERS} WHERE customer_id = {sample_customer_id}").count(),
)
assert after_zorder_count == baseline_count, "row count changed — investigate before trusting the timing comparison"
show_timings()

spark.sql(f"SELECT * FROM {ORDERS} WHERE customer_id = {sample_customer_id}").explain(mode="formatted")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — Z-ORDER, before/after data skipping
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC spark.sql(f"OPTIMIZE {ORDERS} ZORDER BY (customer_id)")
# MAGIC spark.sql(f"SELECT * FROM {ORDERS} WHERE customer_id = {sample_customer_id}").count()
# MAGIC ```
# MAGIC **What to see:** SQL/DataFrame tab → this query's scan node → "number of files read"
# MAGIC — compare directly against the number you wrote down in A.5's companion. On this
# MAGIC small local dataset expect a visible drop, though the effect is far more dramatic at
# MAGIC the full 1,000,000-row scale you just ran in Databricks above. Also check
# MAGIC `PushedFilters` on the same node — same signal `.explain()` prints as text.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C.3 — Cross-check with `DESCRIBE HISTORY`

# COMMAND ----------

display(
    spark.sql(f"DESCRIBE HISTORY {ORDERS}")
    .where("operation = 'OPTIMIZE'").orderBy(F.desc("version")).limit(1)
    .select("version", "timestamp", "operationParameters.zOrderBy",
            "operationMetrics.numRemovedFiles", "operationMetrics.numAddedFiles")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section D — Liquid Clustering
# MAGIC
# MAGIC `CLUSTER BY` uses the same ZCube mechanism as Z-Order, as a first-class, stateful
# MAGIC table property instead of a command you re-run:
# MAGIC
# MAGIC | | Z-ORDER | Liquid Clustering |
# MAGIC |---|---|---|
# MAGIC | Invocation | `OPTIMIZE t ZORDER BY (cols)` every time | `CLUSTER BY (cols)` set once |
# MAGIC | Changing columns | Full table rewrite | `ALTER TABLE t CLUSTER BY (new_cols)` — only *new* data clusters on the new key |
# MAGIC | Max columns | 4 (hard guidance) | More flexible |
# MAGIC | Best fit | Existing tables, one-off reorganization | New tables, or evolving query patterns |
# MAGIC
# MAGIC **When to use which** — Z-ORDER for existing tables you're reorganizing once, with a
# MAGIC stable set of ≤4 filter columns; Liquid Clustering for new tables, or whenever the
# MAGIC clustering key might change later. Databricks' own default recommendation for new UC
# MAGIC managed tables going forward is Liquid Clustering.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {ORDERS_LC} (
  order_id STRING, store_id INT, customer_id INT, order_date DATE,
  order_amount DOUBLE, item_count INT, product_category STRING, payment_type STRING
)
USING DELTA
CLUSTER BY (customer_id, store_id)
""")

for lc_batch in range(3):
    make_batch(lc_batch).withColumn("order_id", F.concat(F.lit(f"LC{lc_batch}-"), F.col("order_id"))).write.format("delta").mode("append").saveAsTable(ORDERS_LC)

spark.sql(f"OPTIMIZE {ORDERS_LC}")
display(spark.sql(f"DESCRIBE DETAIL {ORDERS_LC}").select("clusteringColumns", "numFiles", "sizeInBytes"))
print(f"To re-cluster later without a rewrite: ALTER TABLE {ORDERS_LC} CLUSTER BY (store_id); OPTIMIZE {ORDERS_LC};")

# COMMAND ----------

# MAGIC %md
# MAGIC ### D.2 — `CLUSTER BY AUTO`: letting Databricks pick the clustering columns
# MAGIC
# MAGIC Manual `CLUSTER BY (cols)` still requires you to know the right columns up front.
# MAGIC **`CLUSTER BY AUTO`** hands that decision to Databricks: it observes actual query
# MAGIC patterns (filters, joins) over time and adjusts clustering keys automatically,
# MAGIC without a full rewrite — the same "set once" simplicity as manual Liquid Clustering,
# MAGIC but without having to correctly guess the key in advance.
# MAGIC
# MAGIC **When to use `AUTO` vs. an explicit key:** `AUTO` when query patterns are still
# MAGIC settling or vary across teams/dashboards; an explicit key when you already know
# MAGIC exactly which 1–4 columns dominate your filters and want deterministic, predictable
# MAGIC clustering behavior.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {ORDERS_LC_AUTO} (
  order_id STRING, store_id INT, customer_id INT, order_date DATE,
  order_amount DOUBLE, item_count INT, product_category STRING, payment_type STRING
)
USING DELTA
CLUSTER BY AUTO
""")

for lc_batch in range(3):
    make_batch(lc_batch).withColumn("order_id", F.concat(F.lit(f"LCA{lc_batch}-"), F.col("order_id"))).write.format("delta").mode("append").saveAsTable(ORDERS_LC_AUTO)

# Run a few representative queries so Databricks has query patterns to learn from —
# AUTO doesn't pick a key on day one, it adapts as usage accumulates.
spark.sql(f"SELECT * FROM {ORDERS_LC_AUTO} WHERE customer_id = {sample_customer_id}").count()
spark.sql(f"SELECT * FROM {ORDERS_LC_AUTO} WHERE store_id = 101").count()

display(spark.sql(f"DESCRIBE DETAIL {ORDERS_LC_AUTO}").select("clusteringColumns", "numFiles"))
print("clusteringColumns may show empty until Databricks has observed enough query activity")
print("to choose a key — this is expected on a freshly created table with only a couple of queries run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — Liquid Clustering
# MAGIC
# MAGIC `CLUSTER BY AUTO` is a Databricks-managed decision (it needs observed workload
# MAGIC history) — no local equivalent. Manual `CLUSTER BY (cols)` **is** genuine OSS Delta
# MAGIC (3.1.0+), so that part of this section runs for real locally:
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC ORDERS_LC_PATH = f"{BASE_DIR}/orders_lc"
# MAGIC ORDERS_LC = f"delta.`{ORDERS_LC_PATH}`"
# MAGIC spark.sql(f"""
# MAGIC CREATE TABLE IF NOT EXISTS {ORDERS_LC} (
# MAGIC   order_id STRING, store_id INT, customer_id INT, order_amount DOUBLE
# MAGIC ) USING DELTA CLUSTER BY (customer_id, store_id)
# MAGIC """)
# MAGIC for b in range(5):
# MAGIC     make_batch(b, n=2000).withColumn("order_id", F.concat(F.lit(f"LC{b}-"), F.col("order_id"))) \
# MAGIC         .write.format("delta").mode("append").save(ORDERS_LC_PATH)
# MAGIC spark.sql(f"OPTIMIZE {ORDERS_LC}")
# MAGIC ```
# MAGIC **What to see:** same shape as Section B's companion — Jobs/Stages tab, mostly file
# MAGIC I/O, no shuffle. Liquid Clustering and plain OPTIMIZE look identical in the Spark
# MAGIC UI; the difference is entirely in what gets persisted as table metadata
# MAGIC (`clusteringColumns`), not in how the job executes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section E — VACUUM
# MAGIC
# MAGIC | Setting | Default |
# MAGIC |---|---|
# MAGIC | Data file retention | **7 days** (`delta.deletedFileRetentionDuration`) |
# MAGIC | Transaction log retention | **30 days** (`delta.logRetentionDuration`) |
# MAGIC
# MAGIC **On this serverless environment specifically:** VACUUM enforces a **hard 168-hour
# MAGIC minimum with no override** — stronger than classic compute, where the check can be
# MAGIC (carefully) disabled. That means you genuinely cannot force a visible deletion here.
# MAGIC That's not a gap in this notebook — it's the platform correctly refusing to let a
# MAGIC training session do something unsafe. The VS Code companion below is where you
# MAGIC actually watch a file get deleted.

# COMMAND ----------

dry_run_files = spark.sql(f"VACUUM {ORDERS} DRY RUN").collect()
print(f"Files eligible for deletion at the default 7-day retention: {len(dry_run_files)}")
print("Expected: close to zero — the files OPTIMIZE/ZORDER just replaced are only minutes old.")

logical_before = spark.sql(f"DESCRIBE DETAIL {ORDERS}").collect()[0].numFiles
vacuum_result = spark.sql(f"VACUUM {ORDERS}")
display(vacuum_result)
logical_after = spark.sql(f"DESCRIBE DETAIL {ORDERS}").collect()[0].numFiles

print(f"\nLogical files (current version): {logical_before} -> {logical_after}  <- unchanged, as expected")
print("VACUUM only ever removes files the CURRENT version no longer references — never active ones.")
print("Note: on this platform, retention cannot be lowered below 168 hours (7 days), so nothing")
print("eligible for deletion exists yet — this run itself is a no-op, by design.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### E.1 — Cross-check with `DESCRIBE HISTORY`
# MAGIC
# MAGIC `VACUUM` shows up as **two** rows — `VACUUM START` (what it found eligible) and
# MAGIC `VACUUM END` (what it actually deleted).

# COMMAND ----------

display(
    spark.sql(f"DESCRIBE HISTORY {ORDERS}")
    .where("operation IN ('VACUUM START', 'VACUUM END')")
    .orderBy(F.desc("version")).limit(2)
    .select("version", "timestamp", "operation",
            "operationMetrics.numFilesToDelete", "operationMetrics.sizeOfDataToDelete",
            "operationMetrics.numDeletedFiles", "operationMetrics.numVacuumedDirectories")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — VACUUM, actually deleting a file
# MAGIC
# MAGIC This is disposable local test data, so we can safely override the retention check —
# MAGIC something Free Edition deliberately won't let you do.
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC import os
# MAGIC
# MAGIC def physical_file_count(path):
# MAGIC     return len([f for f in os.scandir(path.replace("file:", "")) if f.name.endswith(".parquet")])
# MAGIC
# MAGIC before_physical = physical_file_count(ORDERS_PATH)
# MAGIC spark.sql(f"OPTIMIZE {ORDERS}")   # creates files that immediately become superseded
# MAGIC
# MAGIC spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
# MAGIC display_result = spark.sql(f"VACUUM {ORDERS} RETAIN 0 HOURS").collect()
# MAGIC spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "true")   # always restore this
# MAGIC
# MAGIC after_physical = physical_file_count(ORDERS_PATH)
# MAGIC print(f"Physical files: {before_physical} -> {after_physical}  ({before_physical - after_physical} removed)")
# MAGIC ```
# MAGIC **What to see — BEFORE:** Jobs tab shows the OPTIMIZE job. **AFTER:** a new VACUUM
# MAGIC job appears — open it, almost entirely file-deletion I/O, no shuffle, similar shape
# MAGIC to OPTIMIZE's job but doing the opposite (removing files instead of writing them).
# MAGIC The printed before/after physical file count is the only place in this entire course
# MAGIC you see VACUUM's deletion happen in real time.
# MAGIC
# MAGIC **Flag to learners: `RETAIN 0 HOURS` with the check disabled is training-only, on
# MAGIC disposable local data. Never do this against a production table.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section F — Broadcast Joins
# MAGIC
# MAGIC | Setting | Default |
# MAGIC |---|---|
# MAGIC | Static auto-broadcast threshold | **10 MB** (`spark.sql.autoBroadcastJoinThreshold`) |
# MAGIC | AQE runtime conversion threshold | **30 MB** |
# MAGIC | Hard limit | never > 1GB on disk; **8GB in-memory** cap |
# MAGIC
# MAGIC **When to broadcast:** the small side is confidently under ~30MB, or you're
# MAGIC comfortable forcing it below 10MB with an explicit hint. Never broadcast over 1GB on
# MAGIC disk; stay well clear of the 8GB in-memory cap, especially on heavily-compressed data.
# MAGIC
# MAGIC `spark.conf.get('spark.sql.autoBroadcastJoinThreshold')` isn't directly accessible on
# MAGIC serverless — we use `safe_conf()` from Setup instead of letting that crash the cell.

# COMMAND ----------

threshold_display = safe_conf('spark.sql.autoBroadcastJoinThreshold', '10485760 (10MB, serverless default)')
stores_size_kb = spark.sql(f"DESCRIBE DETAIL {STORES}").collect()[0].sizeInBytes / 1024
print(f"autoBroadcastJoinThreshold = {threshold_display}   |   stores table size = {stores_size_kb:.1f} KB")

# COMMAND ----------

# F.1 — Unhinted join: let Spark's planner decide
unhinted_join = spark.sql(f"""
    SELECT o.order_id, o.order_amount, s.store_name, s.region
    FROM {ORDERS} o JOIN {STORES} s ON o.store_id = s.store_id
""")
unhinted_join.explain(mode="formatted")

# COMMAND ----------

# F.2 — Explicitly broadcast-hinted join
hinted_join = (
    spark.table(ORDERS).alias("o")
    .join(broadcast(spark.table(STORES)).alias("s"), on="store_id")
    .select("order_id", "order_amount", "store_name", "region")
)
hinted_join.explain(mode="formatted")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — broadcast join
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC from pyspark.sql.functions import broadcast
# MAGIC unhinted = spark.read.format("delta").load(ORDERS_PATH).join(spark.read.format("delta").load(STORES_PATH), "store_id")
# MAGIC unhinted.collect()
# MAGIC hinted = spark.read.format("delta").load(ORDERS_PATH).join(broadcast(spark.read.format("delta").load(STORES_PATH)), "store_id")
# MAGIC hinted.collect()
# MAGIC ```
# MAGIC **What to see:** SQL/DataFrame tab, compare both queries' DAGs. `unhinted` may
# MAGIC already show `BroadcastExchange` (Spark's planner or AQE deciding for you) —
# MAGIC `hinted` should show it deterministically. Either way, neither should show a plain
# MAGIC `Exchange` on the `stores` side — that shuffle is exactly what broadcasting avoids.

# COMMAND ----------

# MAGIC %md
# MAGIC ### F.3 — Guardrails to remember
# MAGIC - Broadcast hash join is **not supported for full outer joins**
# MAGIC - Right outer join: only the **left** side can be broadcast; other left joins: only the **right**
# MAGIC - **Never broadcast a table larger than 1GB on disk**
# MAGIC - Hard **8GB in-memory** cap regardless of on-disk size — compression can blow past this silently

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section G — Deletion Vectors & Predictive I/O
# MAGIC *(Databricks-managed, Photon-exclusive — conceptual only, no local Spark UI equivalent)*
# MAGIC
# MAGIC **Deletion Vectors:** instead of rewriting an entire Parquet file when one row is
# MAGIC deleted or updated, Delta marks the deleted rows in a small bitmap file alongside the
# MAGIC original — the original stays untouched. Rewrites are deferred to the next
# MAGIC `OPTIMIZE`, which cleans up files with a lot of accumulated deletion-vector "noise".
# MAGIC
# MAGIC **Predictive I/O:** exclusive to the **Photon** engine. Two parts —
# MAGIC **accelerated reads** (an ML model picks the most efficient scan/filter access
# MAGIC pattern) and **accelerated updates** (uses Deletion Vectors to avoid full-file
# MAGIC rewrites on `DELETE`/`UPDATE`/`MERGE`). Requires Photon-enabled compute or a
# MAGIC serverless/pro SQL warehouse, DBR 11.3 LTS+.
# MAGIC
# MAGIC **Why no VS Code companion here:** both features are tied to Databricks' own
# MAGIC managed runtime and Photon specifically — there's no equivalent to install locally.
# MAGIC Treat this as a whiteboard topic, same as Predictive Optimization on Day 2.

# COMMAND ----------

try:
    spark.sql(f"ALTER TABLE {ORDERS} SET TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')")
    display(spark.sql(f"SHOW TBLPROPERTIES {ORDERS}").where("key = 'delta.enableDeletionVectors'"))
except Exception as e:
    print(f"If this errors on your workspace/edition, that's expected — note the error and move on: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section H — Shallow & Deep Clones
# MAGIC
# MAGIC | | Shallow Clone | Deep Clone |
# MAGIC |---|---|---|
# MAGIC | Copies | Metadata only — references source's existing data files | Metadata **and** all data files |
# MAGIC | Speed/cost | Fast, cheap | Slower, full storage cost |
# MAGIC | Risk | Breaks if the source is `VACUUM`ed and referenced files are removed | Fully independent, safe from source changes |
# MAGIC | Best for | Dev/test/experimentation without duplicating storage | Backup, migration, disaster recovery |
# MAGIC | OSS Delta support | Yes | **No — Databricks-managed only** |
# MAGIC
# MAGIC **When to use which:** shallow for a quick, disposable dev/test copy where you don't
# MAGIC mind it breaking if someone VACUUMs the source; deep when you need a fully
# MAGIC independent copy that survives the source table's lifecycle — backups, migrations,
# MAGIC cross-region copies.

# COMMAND ----------

spark.sql(f"CREATE OR REPLACE TABLE {ORDERS_CLONE} SHALLOW CLONE {ORDERS}")
display(spark.sql(f"DESCRIBE DETAIL {ORDERS_CLONE}").select("numFiles", "sizeInBytes"))
print(f"{ORDERS_CLONE} created — check numFiles/sizeInBytes above against {ORDERS} from Section F:")
print("nearly instant, and sizeInBytes reflects REFERENCED data, not a physical copy.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — Shallow Clone
# MAGIC
# MAGIC Genuinely OSS Delta-supported — Deep Clone is not, so there's no companion for that
# MAGIC half of this section.
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC CLONE_PATH = f"{BASE_DIR}/orders_shallow_clone"
# MAGIC spark.sql(f"CREATE OR REPLACE TABLE delta.`{CLONE_PATH}` SHALLOW CLONE {ORDERS}")
# MAGIC ```
# MAGIC **What to see:** Jobs tab — a very short-lived job, almost no I/O at all (no data is
# MAGIC copied, just metadata). Contrast this against Section B's OPTIMIZE companion job,
# MAGIC which does real file I/O — cloning is close to instantaneous specifically *because*
# MAGIC it skips that work.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section I — Edge Case: Watching a Query Fail in the Spark UI
# MAGIC
# MAGIC Every section so far showed a technique working. Production systems also need you to
# MAGIC diagnose a technique or job that's **failing** — and this is the one class of
# MAGIC problem Free Edition genuinely cannot show you, since there's no Jobs/Stages view to
# MAGIC inspect a failure in. This section runs on Databricks so you see the error Spark
# MAGIC surfaces directly in the notebook — but the VS Code companion is where you actually
# MAGIC see what a **failed job looks like** in the Spark UI itself.
# MAGIC
# MAGIC We use a Python UDF that divides by a column that's zero for some rows, unguarded —
# MAGIC deterministic, reliable, and a good callback to the general advice to avoid Python
# MAGIC UDFs where a native Spark function will do: a bug like this one is exactly the kind
# MAGIC of failure mode a Python UDF makes easy to introduce and hard to see coming.

# COMMAND ----------

from pyspark.sql.types import DoubleType


def risky_divide(order_amount, item_count):
    return order_amount / item_count   # no guard against item_count == 0 — the bug, on purpose


risky_divide_udf = F.udf(risky_divide, DoubleType())

try:
    result = (
        spark.table(ORDERS)
        .withColumn("item_count_broken", F.when(F.rand() < 0.001, F.lit(0)).otherwise(F.col("item_count")))
        .withColumn("amount_per_item", risky_divide_udf(F.col("order_amount"), F.col("item_count_broken")))
        .collect()
    )
    print("No failure this run — the 0.1% injected zero-rate is random. Re-run the cell if this happened.")
except Exception as e:
    print("Query failed, as expected. The exception below is what Spark surfaces on the driver side —")
    print("this notebook environment has no Spark UI to inspect the failed job/stage/task further:\n")
    print(str(e)[:1500])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🖥️ VS Code Spark UI companion — seeing the failure in Jobs/Stages
# MAGIC
# MAGIC ```python
# MAGIC # %%
# MAGIC from pyspark.sql.types import DoubleType
# MAGIC
# MAGIC def risky_divide(order_amount, item_count):
# MAGIC     return order_amount / item_count   # unguarded — will raise ZeroDivisionError on item_count == 0
# MAGIC
# MAGIC risky_divide_udf = F.udf(risky_divide, DoubleType())
# MAGIC
# MAGIC broken = (
# MAGIC     spark.read.format("delta").load(ORDERS_PATH)
# MAGIC     .withColumn("item_count", F.when(F.rand() < 0.05, F.lit(0)).otherwise(F.lit(2)))   # 5% zeros — reliably triggers fast
# MAGIC     .withColumn("amount_per_item", risky_divide_udf(F.col("order_amount"), F.col("item_count")))
# MAGIC )
# MAGIC try:
# MAGIC     broken.collect()
# MAGIC except Exception as e:
# MAGIC     print("Failed as expected — now go look at the Spark UI.")
# MAGIC ```
# MAGIC **What to see in `localhost:4040`:**
# MAGIC - **Jobs tab** — this job shows a **red/failed status**, unlike every successful job
# MAGIC   so far
# MAGIC - Click into it → **Stages tab** — the failed stage shows a **non-zero "Failed"
# MAGIC   count** in its task summary
# MAGIC - Click into the stage → the task table shows **individual failed tasks** — Spark
# MAGIC   retries a failed task automatically (`spark.task.maxFailures`, default **4**) before
# MAGIC   giving up on the stage, so expect to see **more task attempts than partitions**
# MAGIC - Click into one failed task → its detail panel includes the **full Python
# MAGIC   traceback** — the actual `ZeroDivisionError` and the line of your UDF that raised
# MAGIC   it, exactly the debugging information the driver-side exception in the Databricks
# MAGIC   cell above can't show you as clearly
# MAGIC
# MAGIC This is the concrete case for keeping VS Code's Spark UI in your toolkit even once
# MAGIC you have classic Databricks compute: driver-side error messages tell you *that*
# MAGIC something failed; the Spark UI's failed-stage task table tells you *which partition,
# MAGIC which attempt, and why*.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Full Operations Timeline

# COMMAND ----------

display(
    spark.sql(f"DESCRIBE HISTORY {ORDERS}")
    .where("operation IN ('OPTIMIZE', 'VACUUM START', 'VACUUM END')")
    .orderBy("version")
    .select("version", "timestamp", "operation", "operationParameters.zOrderBy",
            "operationMetrics.numRemovedFiles", "operationMetrics.numAddedFiles",
            "operationMetrics.numFilesToDelete", "operationMetrics.numDeletedFiles")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Day 1 Recap
# MAGIC
# MAGIC | Technique | What it fixes | Default(s) / when to use |
# MAGIC |---|---|---|
# MAGIC | `OPTIMIZE` | Small-file fragmentation | 1GB target (manual) / 128MB (Auto Optimize) |
# MAGIC | `OPTIMIZE ... ZORDER BY` | Poor data skipping | Max 4 cols; existing tables, stable filter columns |
# MAGIC | Liquid Clustering (`CLUSTER BY`) | Same as Z-Order, incremental | New tables, or evolving clustering keys |
# MAGIC | `CLUSTER BY AUTO` | Same, without choosing the key yourself | Query patterns still settling or vary by team |
# MAGIC | `VACUUM` | Storage bloat | 7-day retention (this env: 168h hard minimum, no override) |
# MAGIC | Broadcast joins | Unnecessary shuffle | 10MB static / 30MB AQE / 1GB-disk & 8GB-memory hard caps |
# MAGIC | Deletion Vectors + Predictive I/O | Full-file rewrites on DELETE/UPDATE/MERGE | Photon-only, DBR 11.3 LTS+ |
# MAGIC | Shallow Clone | Cheap dev/test copies | Fine with the copy breaking if source is VACUUMed |
# MAGIC | Deep Clone | Independent, durable copies | Backup/migration/DR — Databricks-managed only |
# MAGIC
# MAGIC **On Databricks Free Edition specifically:** `DESCRIBE DETAIL`/`DESCRIBE HISTORY`
# MAGIC replace all direct file/log access; VACUUM can't be forced to delete visibly; several
# MAGIC `spark.conf.get()` calls need a fallback. None of this is a downgrade on classic
# MAGIC compute — it's a stricter, universally-portable way to write the same notebook.
# MAGIC
# MAGIC **Take-home for Day 2:** every VS Code companion above shared one local session —
# MAGIC Day 2 opens a fresh one, since none of its techniques need Delta at all.
