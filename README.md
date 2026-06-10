# Optimization Workshop

A short, hands-on workshop on how optimization algorithms (SGD, Momentum, AdaGrad,
Adam, and Muon) actually behave when training neural networks. Everything runs in
two Jupyter notebooks and is **CPU-only** — no GPU required, and it runs fine on a
normal laptop.

- **`notebooks/phase1_lowdim.ipynb`** — watch the optimizers move on simple 2-D loss
  surfaces (bowls, valleys, saddles). Great for building intuition.
- **`notebooks/phase2_highdim.ipynb`** — train a small ReLU network on a harder,
  high-dimensional classification task and compare the optimizers, including a look
  at *why* Muon converges faster.

---

## 1. Setup from scratch (you have nothing installed)

We use a tool called [`uv`](https://docs.astral.sh/uv/), which installs the correct
Python version and all the libraries for you automatically. You only need to do
this once.

### Step 1 — Get the code

Either:

- **Download the ZIP**: click the green **Code** button on the project page →
  **Download ZIP** → unzip it somewhere easy to find (e.g. your Desktop), **or**
- **Use git** (if you have it): `git clone <REPO_URL>` and then `cd` into the folder.

### Step 2 — Install `uv`

Open a terminal and run the command for your operating system.

**Windows** (open **PowerShell** — search for it in the Start menu):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux** (open the **Terminal** app):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After it finishes, **close and reopen your terminal** so it picks up the new `uv`
command. Check it worked:

```bash
uv --version
```

### Step 3 — Go into the project folder

In the terminal, move into the folder you downloaded/unzipped. For example:

```bash
cd Desktop/optimization-workshop
```

(On Windows the path uses backslashes, e.g. `cd Desktop\optimization-workshop`.)

> Tip: type `cd ` (with a space) and then drag the folder from your file explorer
> into the terminal — it fills in the path for you.

### Step 4 — Install everything

```bash
uv sync
```

This creates an isolated environment and downloads Python, NumPy, PyTorch (CPU),
matplotlib, and Jupyter. The first run downloads a few hundred MB, so it can take a
few minutes. You only do this once.

### Step 5 — Launch the notebooks

```bash
uv run jupyter notebook
```

Your web browser opens automatically. Click into the **`notebooks/`** folder and
open `phase1_lowdim.ipynb` to begin. When you're done, go back to the terminal and
press `Ctrl + C` to stop the server.

---

## 2. Running a notebook

- Run cells **top to bottom**: click a cell and press **`Shift + Enter`**, or use
  *Run → Run All Cells*. The first few cells set things up, so always run them first.
- Each section has a short **"What to notice"** note explaining what the plot shows.
- Look for the **"Try it yourself"** section at the bottom — change a learning rate,
  pick a different objective, or swap optimizers, and re-run to see what happens.
- If you edit the helper code in `workshoplib/` while the notebook is open, the
  notebooks use `%autoreload`, so your changes take effect on the next cell run
  without restarting.

### Quick check that your setup works

```bash
uv run python smoke_test.py
```

If it prints a few shapes and the words `Sequential` and `Adam`, you're good to go.

---

## 3. What's in this repo

| Path | What it is |
| --- | --- |
| `notebooks/phase1_lowdim.ipynb` | Phase 1: optimizers on 2-D loss surfaces, with contour/trajectory and loss-curve plots. |
| `notebooks/phase2_highdim.ipynb` | Phase 2: training a ReLU network on high-dimensional data; optimizer comparison and the Muon analysis. |
| `workshoplib/` | Reusable helper code imported by the notebooks (see below). |
| `slides_figures/` | PDF versions of selected Phase 1 plots, for slides. |
| `smoke_test.py` | A tiny script to confirm the environment is set up correctly. |
| `pyproject.toml` | The list of required libraries and Python version. |
| `uv.lock` | Exact, reproducible versions of every dependency (used by `uv sync`). |
| `PROJECT_BRIEF.md` | The teaching goals and design of the workshop. |
| `AGENTS.md` | Notes for AI/code assistants working on this repo (not needed to run it). |
| `README.md` | This file. |

### Inside `workshoplib/`

| File | What it does |
| --- | --- |
| `objectives.py` | The 2-D test functions for Phase 1 (quadratic bowl, ill-conditioned and rotated quadratics, absolute-value valley, Rosenbrock, saddle, Beale) and their suggested learning rates. |
| `optimizers.py` | From-scratch implementations of SGD, Momentum, AdaGrad, and Adam, plus `run_descent` to trace an optimizer's path on an objective. |
| `viz.py` | Plotting helpers: contour + trajectory plots, loss curves, and training curves. |
| `odt.py` | Generates the "oblique decision tree" (ODT) labelling rule used to create the Phase 2 dataset. |
| `datagen.py` | Builds the Phase 2 classification dataset (train/validation tensors) from the ODT. |
| `model.py` | Builds the small ReLU classifier (`make_mlp`) used in Phase 2. |
| `optimization.py` | PyTorch optimizer factory for Phase 2, including a custom **Muon** optimizer. |
| `training.py` | The mini-batch training loop, which also records loss/accuracy and parameter snapshots. |
| `analysis.py` | Tools for the Phase 2 Muon story: measuring how first-layer neurons align with the ODT hyperplanes. |

---

## 4. Troubleshooting

- **`uv` not found after installing it** — close the terminal completely and open a
  new one, then try again.
- **`uv sync` is slow or seems stuck** — it's downloading PyTorch the first time; give
  it a few minutes on a normal connection.
- **The browser didn't open** — look in the terminal for a line starting with
  `http://localhost:8888/...` and paste that into your browser.
- **A plot didn't appear** — make sure you ran the setup cells at the top of the
  notebook first, then re-run the cell.
- **Want to start clean** — delete the `.venv` folder and run `uv sync` again.
