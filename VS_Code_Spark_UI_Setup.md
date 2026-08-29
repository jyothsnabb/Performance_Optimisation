# VS Code + Spark UI Setup — Ubuntu, macOS, Windows

Sets up a local PySpark environment with a real Spark UI (`http://localhost:4040`),
runnable cell-by-cell from VS Code. Used for the standalone Spark UI concept demos
(shuffle, skew, broadcast, AQE) that accompany the Data + AI Academy optimization course.

---

## 1. Install Java (JDK 11 or 17)

### Ubuntu
```bash
java -version
```
If missing or wrong version:
```bash
sudo apt update
sudo apt install openjdk-17-jdk -y
```
Find the install path and set `JAVA_HOME`:
```bash
sudo update-alternatives --config java
# note the path shown (e.g. /usr/lib/jvm/java-17-openjdk-amd64/bin/java), drop the trailing /bin/java
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
```

### macOS
```bash
brew install openjdk@17
# Apple Silicon:
sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
# Intel:
sudo ln -sfn /usr/local/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
```
Set `JAVA_HOME` (default shell is zsh):
```bash
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v17)' >> ~/.zshrc
source ~/.zshrc
```

### Windows (PowerShell)
```powershell
winget install EclipseAdoptium.Temurin.17.JDK
```
Open a **new** terminal window, then verify:
```powershell
java -version
```
If `JAVA_HOME` wasn't set by the installer:
```powershell
setx JAVA_HOME "C:\Program Files\Eclipse Adoptium\jdk-17.0.x-hotspot"
```
Close and reopen the terminal again after this.

---

## 2. Create a Python virtual environment and install PySpark

### Ubuntu
```bash
sudo apt install python3-venv python3-pip -y
mkdir ~/spark-ui-test && cd ~/spark-ui-test
python3 -m venv venv
source venv/bin/activate
pip install pyspark pandas ipython
```

### macOS
```bash
mkdir ~/spark-ui-test && cd ~/spark-ui-test
python3 -m venv venv
source venv/bin/activate
pip install pyspark pandas ipython
```

### Windows (PowerShell)
```powershell
mkdir C:\spark-ui-test
cd C:\spark-ui-test
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pyspark pandas ipython
```
If `Activate.ps1` is blocked by execution policy, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

> **Version note:** if you plan to also use Delta Lake locally (`delta-spark`), pin
> versions carefully — `delta-spark` and `pyspark` have version-compatibility
> constraints that shift between releases, and the very newest `pyspark` release isn't
> always compatible with the very newest `delta-spark` release. For plain Spark UI demos
> (shuffle, skew, broadcast, AQE — no Delta tables involved), the plain
> `pip install pyspark` above is all you need, no version pinning required.

---

## 3. Install VS Code extensions

Open VS Code → Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`) → install both:

| Extension | Publisher | Why |
|---|---|---|
| **Python** | Microsoft | Interpreter selection, IntelliSense, debugging |
| **Jupyter** | Microsoft | Required for cell-by-cell (`# %%`) execution — the Python extension alone does not give you this |

Both are required — installing only Python is the most common reason "Run Cell" links
don't appear.

---

## 4. Point VS Code at the virtual environment

`Ctrl+Shift+P` / `Cmd+Shift+P` → **Python: Select Interpreter** → choose the one ending
in `./venv/bin/python` (or `venv\Scripts\python.exe` on Windows). If it's not listed,
choose "Enter interpreter path" and browse to it directly.

---

## 5. Write and run a cell-based script

Any `.py` file with `# %%` markers becomes a notebook-like file in VS Code once the
Jupyter extension is active. Each `# %%` starts a new cell; a `# %% [markdown]` cell
renders as formatted text instead of running as code.

```python
# %%
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Demo").master("local[*]").getOrCreate()
print(f"Spark UI: {spark.sparkContext.uiWebUrl}")

# %%
df = spark.range(1000000)
df.groupBy((df.id % 10)).count().show()
```

### Running cells

- A **"Run Cell"** link appears directly above each `# %%` line — click it to run just
  that cell
- With the cursor inside a cell: **Shift+Enter** runs it and moves to the next cell;
  **Ctrl+Enter** (`Cmd+Enter` on Mac) runs it and stays in place
- First run of any cell in a file prompts you to pick a kernel — choose the venv
  interpreter from step 4
- An **Interactive Window** panel opens showing output. It stays alive across cells, so
  variables (including the `spark` session) persist as you run cells in any order

Once running, open **`http://localhost:4040`** in a browser to see the Spark UI live —
it stays up for as long as the Interactive Window's kernel is running, i.e. as long as
`spark` hasn't been stopped and the window hasn't been closed.

---

## 6. Troubleshooting: "Run Cell" / Shift+Enter not working

Work through these in order:

1. **Confirm the Jupyter extension (by Microsoft) is actually installed** — Extensions
   panel, search "Jupyter", it should say "Installed", not show an Install button. This
   is the #1 cause of missing "Run Cell" links.
2. **Reload the window** — `Ctrl+Shift+P` → "Developer: Reload Window". Needed after a
   fresh extension install before the links reliably appear.
3. **Bypass the links entirely** — click inside a cell, then `Ctrl+Shift+P` → "Jupyter:
   Run Current Cell" → Enter. Works even if the visual links aren't rendering. If this
   command doesn't appear in the palette at all, the extension isn't active — go back to
   step 1.
4. **Check for a keybinding conflict** — `Ctrl+Shift+P` → "Preferences: Open Keyboard
   Shortcuts" → search "Jupyter: Run Current Cell" → confirm it has a binding (default is
   usually `Ctrl+Enter` / `Shift+Enter`); assign one manually if it's blank.
5. **Check CodeLens isn't globally disabled** — Settings (`Ctrl+,`) → search "codeLens" →
   ensure `editor.codeLens` is checked. If off, no CodeLens links render anywhere, in any
   file.
6. **Rule out a VS Code fork** — the Jupyter extension is Microsoft-proprietary and isn't
   available on Open VSX, the marketplace used by forks like VSCodium, Cursor, or
   Windsurf. Check **Help → About** to confirm you're on official VS Code. On a fork,
   installing "Jupyter" from the extensions panel may silently fail or install a broken
   substitute.

---

## 7. Verifying it all worked

- [ ] `java -version` prints a JDK version (11 or 17)
- [ ] `(venv)` shows in the terminal prompt after activation
- [ ] `pip show pyspark` shows an installed version
- [ ] Extensions panel shows both **Python** and **Jupyter** (Microsoft) as installed
- [ ] VS Code's interpreter (bottom status bar, or `Python: Select Interpreter`) points
      at the venv
- [ ] A "Run Cell" link appears above a `# %%` line
- [ ] Running a cell that builds a `SparkSession` prints a `Spark UI:` URL, and that URL
      opens a live Spark UI in the browser
