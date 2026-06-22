# EmotiWave

**Project:** EEG Emotion Recognition (DREAMER)

**Summary:**
EmotiWave is an EEG emotion recognition pipeline built on the DREAMER dataset. It converts the original MATLAB recordings into windowed, baseline-corrected features (Differential Entropy and related statistics) and trains efficient XGBoost models for valence/arousal/dominance classification. The codebase focuses on reproducible preprocessing, windowed feature extraction, and leak-resistant evaluation (subject-dependent and LOSO).


**Quick Start**

Prerequisites:
- Python 3.10+ (3.11 recommended)
- ~8 GB RAM (more for parallel feature extraction)
- Optional: GPU for XGBoost if available

Clone repository:

```bash
git clone https://github.com/DemonSlayer256/EmotiWave.git
cd EmotiWave
```

Linux / macOS

Create a virtual environment and activate it:

```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Windows (PowerShell / CMD)

Create and activate a virtual environment (PowerShell):

```powershell
py -3 -m venv env
.\env\Scripts\Activate.ps1    # PowerShell
# or in CMD:
env\Scripts\activate.bat
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Download the DREAMER dataset:

The repository expects a file named `DREAMER.mat` to live in `data/raw/`.
Replace `$DREAMER_URL` with the official dataset URL you obtained (e.g., from the DREAMER paper or dataset host).

```bash
mkdir -p data/raw
curl -L -o data/raw/DREAMER.mat "$DREAMER_URL"
```

PowerShell alternative for download:

```powershell
mkdir data\raw -Force
Invoke-WebRequest -Uri $env:DREAMER_URL -OutFile data\raw\DREAMER.mat
```

If you already have `DREAMER.mat`, copy it into `data/raw/`.

Process the raw MATLAB file into NumPy trial files and `metadata.json`:

```bash
python src/data/process_data.py
```

This writes per-trial stimulus and baseline arrays into `data/interim/` and creates `data/interim/metadata.json`.

Extract features (windowed, baseline-corrected):

```bash
python src/data/feature_extraction.py
```

This creates processed CSVs in `data/processed/` such as `features_valence.csv` and `features_quadrant.csv`.

Train models / run evaluation:

```bash
python src/data/train.py
```

This runs the XGBoost pipeline (subject-dependent and subject-independent evaluations). Output metrics and plots are saved under `outputs/` and `logs/`.

Folder structure (top-level):
- `data/`
	- `raw/` — place `DREAMER.mat` here
	- `interim/` — per-trial NumPy files and `metadata.json` (created by `process_data.py`)
	- `processed/` — CSV feature matrices (created by `feature_extraction.py`)
- `src/` — project source code
	- `data/process_data.py` — converts DREAMER.mat to trial files and metadata
	- `data/feature_extraction.py` — windowed feature extraction and baseline correction
	- `data/train.py` — training and evaluation pipelines (XGBoost)
- `models/` — saved models (not tracked by default)
- `outputs/` — evaluation metrics, plots, SHAP outputs
- `logs/` — experiment logs and run notes

Notes & tips:
- Feature extraction is parallelized via `joblib`. Set `os.environ["JOBLIB_NUM_THREADS"]` or limit `n_jobs` inside the script if you need to conserve CPU.
- If you run into memory pressure during feature extraction, reduce `n_jobs` or run `feature_extraction.py` on fewer subjects at a time by editing `metadata` loading.
- The code expects 14 EEG channels sampled at 128 Hz. Do not change file formats unless you update the loader.

Troubleshooting:
- Missing `DREAMER.mat`: obtain from the original dataset website and place in `data/raw/`.
- Errors installing native packages (e.g., `numba`, `llvmlite`): ensure your system has appropriate build tools or use a prebuilt Python distribution (conda) and adjust `requirements.txt` accordingly.

Contributing:
- Fork the repo, create a branch, and open a PR. Add tests and update `docs/` for major changes.

License: Check repository root for license information (none included by default).

Contact: Open issues or PRs on the repository for questions or suggestions.
