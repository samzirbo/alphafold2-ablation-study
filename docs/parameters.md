# Prediction Parameters

Reference for the AlphaFold2 parameters used in `predict.py`. All values are set
in `RUN_CONFIG` and passed to ColabFold's `run()` function. Parameters are grouped
logically; each entry lists possible values, the value we use, and why.

---

## 1. Model Selection

### `model_type`

| Options | Our value |
|---|---|
| `"auto"`, `"alphafold2"`, `"alphafold2_ptm"`, `"alphafold2_multimer_v1"`, `"alphafold2_multimer_v2"`, `"alphafold2_multimer_v3"`, `"deepfold_v1"` | `"alphafold2"` |

Which AF2 weight set to load. We use the original monomer model from Jumper et al. (2021).

### `num_models`

| Options | Our value |
|---|---|
| `1`, `2`, `3`, `4`, `5` | `5` |

Number of distinct checkpoints (`model_1` through `model_5`) to
run per seed. All 5 maximize structural diversity.

---

## 2. MSA & Evolutionary Information

### `msa_mode`

| Options | Our value |
|---|---|
| `"mmseqs2_uniref_env"`, `"mmseqs2_uniref"`, `"single_sequence"`, `"custom"` | `"custom"` |

How the MSA is obtained. `"mmseqs2_uniref_env"` fetches MSAs at runtime via the
ColabFold API; `"custom"` reads a precomputed `.a3m` file from `data/<protein>/`.
We use `"custom"` to ensure reproducibility and to apply offline MSA perturbations
(subsampling, column masking, etc.) before prediction.

### `max_msa`

| Options | Our value |
|---|---|
| `"max_seq:max_extra_seq"` string, or `None` for model defaults | `"512:5120"` (default) |

Controls MSA depth at inference. `max_seq` is the number of MSA sequences fed to
the Evoformer attention; `max_extra_seq` provides additional evolutionary signal
via a cheaper summary pathway. This is the primary ablation axis.

Ablation values following del Alamo et al.:

| `--max-msa` | Effective depth |
|---|---|
| `16:32` | 32 |
| `32:64` | 64 |
| `64:128` | 128 |
| `128:256` | 256 |
| `256:512` | 512 |
| `512:1024` | 1024 |
| `1024:2048` | 2048 |
| `512:5120` | 5120 (default) |

---

## 3. Templates

### `use_templates`

| Options | Our value |
|---|---|
| `True`, `False` | `False` |

Whether to include structural templates in the input features. Disabled to prevent
the model from copying known experimental conformations — predictions rely solely
on the MSA, which is the signal we want to study.

---

## 4. Recycling

### `num_recycles`

| Options | Our value |
|---|---|
| `None` (auto), `0`, `1`, `3`, `6`, `12`, `24`, `48` | `1` |

Number of iterative refinement passes through the Evoformer and structure module.
Set to 1 following del Alamo et al. — fewer recycles preserve conformational diversity that would otherwise
collapse toward a single state.

### `recycle_early_stop_tolerance`

| Options | Our value |
|---|---|
| `None` (auto), `0.0`, `0.5`, `1.0` | `0.0` |

Stop recycling early if Cα-RMSD between iterations falls below this threshold (Å).
With only 1 recycle, early stopping is irrelevant — set to 0.0 to guarantee the single recycle always executes.

---

## 5. Conformational Sampling

### `num_seeds`

| Options | Our value |
|---|---|
| Any positive integer (default `1`) | `5` |

Number of random seeds per model checkpoint. Total predictions per protein =
`num_models × num_seeds`. With 5 models and 5 seeds this gives 25 predictions,
matching the ensemble size in del Alamo et al.

### `random_seed`

| Options | Our value |
|---|---|
| Any integer (default `0`) | `0` |

Base seed for reproducibility. The script runs seeds `0, 1, ..., num_seeds - 1`.
Each seed produces a different random crop of the MSA and different initial
weights for the recycling, leading to structural variation.

### `use_dropout`

| Options | Our value |
|---|---|
| `True`, `False` | `False` |

Keep dropout layers active at inference. We keep it `False` —
diversity in our study comes from MSA perturbation and multiple seeds.

---

## 6. Relaxation

### `num_relax`

| Options | Our value |
|---|---|
| `0` (skip), or any positive integer | `0` |

How many top-ranked structures to energy-minimize with OpenMM/Amber. Skipped
entirely — it is computationally expensive and unnecessary for our analysis.

---

## 7. Parameters Left at ColabFold Defaults

Not set in `RUN_CONFIG`; they inherit defaults from ColabFold's `run()`.

