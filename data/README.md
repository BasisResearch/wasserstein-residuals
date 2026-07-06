# Datasets

The benchmark data artifacts for the paper experiments are committed here so a
fresh clone runs end-to-end without a separate fetch step. Datasets reach the
model through `stitching.data.load_data(cfg)`.

| Directory | Arrays | Loaded by | Origin |
| --- | --- | --- | --- |
| `chiral/` | `chiral-simulation.npz` | `chiral` (§5.5) | Simulated in this repo (regenerate with `python data/chiral/generate.py`) |
| `cis/` | `cis-simulation.npz` | `cis` (§5.4) | Simulated (McKean–Vlasov: double-well + Gaussian interaction) |
| `RNA_PCA_5/` | `data.npy`, `sample_labels.npy`, `split_70_30_seed0.npz` | `rna` (§5.3) | **Third-party** — PCA-reduced scRNA-seq, see attribution below |

The synthetic-potential experiments (`synthetic` §5.2, `wavy_valley` §5.1, and
`doublewell`) are simulated live from a registered potential (`--potential`) and
need no data files.

## Third-party attribution — `RNA_PCA_5/`

The RNA arrays are a **derivative** of published single-cell data, redistributed
here in PCA-reduced form (first 5 principal components, per-component
standardised) with attribution:

- **Raw scRNA-seq (embryoid body):** Moon, van Dijk, Wang, Gigante, Burkhardt,
  Chen, Yim, van den Elzen, Hirn, Coifman, Ivanova, Wolf, Krishnaswamy.
  *Visualizing structure and transitions in high-dimensional biological data.*
  Nature Biotechnology 37, 1482–1492 (2019). Data: Mendeley `v6n743h5ng`.
- **PCA-embedded artifact (`eb_velocity_v5.npz`):** Tong, Huang, Wolf, van Dijk,
  Krishnaswamy. *TrajectoryNet: A Dynamic Optimal Transport Network for Modeling
  Cellular Dynamics.* ICML 2020. (Code MIT-licensed; https://github.com/KrishnaswamyLab/TrajectoryNet.)

`data.npy` here reproduces the standardised first-5-PC embedding of that raw;
`sample_labels.npy` are the integer timepoint labels; `split_70_30_seed0.npz`
is a deterministic particle split. Please cite the two references above when
using this dataset. See the repository `NOTICE` file for the license summary.

## Regenerating

`chiral-simulation.npz` is reproducible from `data/chiral/generate.py` (the
canonical bytes are committed so results match without a re-run). The `cis` and
`rna` arrays are committed directly; their upstream build pipelines are not part
of this release.
