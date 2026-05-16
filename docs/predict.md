# predict.py — User Guide

CLI wrapper around ColabFold for running AlphaFold2 structure predictions.

## Quick Start

```bash
# Single protein
python scripts/predict.py --job baseline --input LAT1

# Multiple proteins
python scripts/predict.py --job baseline --input LAT1 ZnT8 MCT1

# All 12 proteins
python scripts/predict.py --job baseline
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--job` | *(required)* | Name for this run. Results go to `results/<job>/`. |
| `--input` | all proteins | One or more protein names to predict. |
| `--drive` | `None` | External base path for results (e.g. Google Drive). |
| `--num-models` | `5` | Number of AF2 model checkpoints (1–5). |
| `--num-seeds` | `5` | Random seeds per model. Total predictions = models × seeds. |
| `--max-msa` | `512:5120` | MSA depth as `max_seq:max_extra_seq`. |
| `--msa-mode` | `custom` | `custom` (precomputed `.a3m`) or `mmseqs2_uniref_env`. |

## Input Data

Each protein needs a directory under `data/<protein>/` containing either:

- `<protein>.a3m` — precomputed MSA (preferred with `--msa-mode custom`)
- `<protein>.fasta` — raw sequence (fallback; MSA will be fetched remotely)

## Output Structure

```
results/<job>/
  └── <protein>/
        ├── <protein>_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb
        ├── <protein>_unrelaxed_rank_001_alphafold2_model_2_seed_000.pdb
        ├── <protein>_scores_rank_001_alphafold2_model_1_seed_000.json
        └── ...
```

With `--drive`, the base path changes: `<drive>/<job>/<protein>/...`

> **Tip (Colab):** If you're running on Google Colab, use a shared Google Drive
> folder so that all collaborators have access to the results. Add the shared
> folder as a shortcut to your own Drive ("Add shortcut to Drive"), then pass
> the mounted path via `--drive`.

## Crash Recovery

Each (model, seed) pair is run independently. Before each prediction, the script
checks whether the output PDB already exists — if so, it skips it. This means you
can safely re-run the same command after a crash and only the missing predictions
will be computed.

## Examples

```bash
# Quick test: 1 model, 1 seed
python scripts/predict.py --job test --input LAT1 --num-models 1 --num-seeds 1

# Full baseline: 5 models × 5 seeds = 25 predictions per protein
python scripts/predict.py --job baseline

# MSA depth ablation
python scripts/predict.py --job msa_depth_16 --input LAT1 --max-msa 16:32

# Save to Google Drive (Colab)
python scripts/predict.py --job baseline --drive /content/drive/MyDrive/results
```
