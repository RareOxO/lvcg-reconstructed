# LVCG

## Learning Cardiac Latent Representations in Vectorcardiogram Space

*ICML 2026 · Official PyTorch implementation*

**[🌐 Project Page](https://bosonhwang.github.io/LVCG-page/) · [📄 arXiv](https://arxiv.org/abs/2605.31249)**

<p align="center">
  <img src="asses/fig2.png" alt="LVCG framework" width="100%">
</p>

**LVCG** is the first general self-supervised framework for 12-lead ECG that learns in **latent vectorcardiogram (VCG) space** rather than raw lead signals. Because standard ECG is multiple linear views of the same cardiac field, lead-space learning entangles pathology with electrode geometry and generalizes poorly under domain shift; inspired by the Frank VCG model, LVCG instead recovers a unified 3D field and learns **view-invariant** representations via self-supervised multi-lead reconstruction—lifting visible leads, encoding beat morphology through a token bottleneck, modeling inter-beat dynamics, and projecting back to missing leads with a non-learnable geometry layer. This physically grounded design yields compact embeddings that transfer strongly to linear probing, multi-lead view reconstruction, and non-cardiac detection, especially with scarce labels and cross-site shift.

## Installation

```bash
conda create -n lvcg python=3.10
conda activate lvcg
pip install -e .
```

Requirements: PyTorch 2.x, NumPy, SciPy, PyYAML, WFDB (MIMIC loading), scikit-learn (probing), pandas (AI-READI).

## Data

PhysioNet datasets require [credentialed access](https://physionet.org/settings/credentialing/).

| Dataset | Role | Official link |
|---------|------|---------------|
| MIMIC-IV ECG | Pretraining | https://physionet.org/content/mimic-iv-ecg/1.0/ |
| PTB-XL | Probing (4 tasks) | https://physionet.org/content/ptb-xl/1.0.3/ |
| CPSC 2018 / ICBEB | Probing | https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/ |
| Chapman–Shaoxing (CSN) | Probing | https://physionet.org/content/ecg-arrhythmia/1.0.0/ |
| AI-READI | Probing (`cardio_relevant`) | https://aireadi.org/dataset · https://fairhub.io/datasets/1 |

**After download:** see [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) (manifest for MIMIC, folder layout for probing, and config paths).

## Training

LVCG pretraining on MIMIC-IV (self-supervised):

```bash
python scripts/train.py --config configs/train/lvcg_v5_gru.yaml
```

Checkpoints are written to `checkpoints/<run_id>/` (default run id `m5s1k1`). Use **`final.pt`** for downstream evaluation.

**Logs and resuming.** Each run writes `logs/<run_id>/train_log.jsonl`: every
`train.log_interval` steps the five loss terms (total, recon, temporal, beat, base), learning
rate, gradient norm and speed; every `train.eval_interval` steps the same terms on the first
`train.eval_batches` validation batches, with fixed masks; and a line for every checkpoint.
`run_info.json` holds the config, environment and data sizes. Every
`train.resume_interval` steps `checkpoints/<run_id>/last.pt` is refreshed with the model,
optimiser, step and random states. To continue an interrupted run, repeat the same command
with `--resume`:

```bash
python scripts/train.py --config configs/train/lvcg_v5_gru.yaml --model.vectorized_stitcher true --resume
```

The training order is a seeded shuffle per epoch (`train.seed`), so a resumed run picks up
at the exact batch it stopped at; the tests check that 3 steps plus a resume to 6 give the
same weights as 6 steps straight. A fresh run refuses to start where checkpoints already
exist, so a run id is never overwritten by accident.

**Faster beat stitching (optional).** The decoder's `BeatStitcher` loops over every beat in
Python and dominates step time. `lvcg/models/blocks/stitcher_vectorized.py` computes the
same output in a few tensor operations; it matches the loop to 1e-10 in double precision
(outputs and gradients, including the full pretraining loss) and has no parameters, so
checkpoints load either way. It is off by default:

```bash
python scripts/train.py --config configs/train/lvcg_v5_gru.yaml --model.vectorized_stitcher true
```

or set `model.vectorized_stitcher: true` in the YAML. On an RTX 4060 Laptop GPU a batch-64
pretraining step drops from 0.60 s to 0.21 s.

**Faster decoder upsampling (optional).** The beat decoder doubles its length five times
with `F.interpolate(mode="linear")`, whose CUDA backward took 52% of the GPU time of a
step. `model.fast_upsample: true` (or `--model.fast_upsample true`) uses the closed form of
a 2x linear interpolation instead (`lvcg/models/blocks/fast_upsample.py`): identical to
1e-12 in double precision, values and gradients, and a further 0.31 s to 0.16 s per step on
the same laptop GPU. It has no parameters either, and both speed switches may be changed
when resuming a run.

```bash
python scripts/train.py --config configs/train/lvcg_v5_gru.yaml \
  --model.vectorized_stitcher true --model.fast_upsample true
```

`scripts/benchmark_batch.py` measures throughput on the GPU at hand.

```bash
python scripts/train.py --config configs/train/lvcg_v5_gru.yaml \
  --data.meta_root /path/to/mimic_manifest.jsonl
```

## Evaluation

Linear probing on 7 datasets with **10% training labels** and **seed 42**:

```bash
python scripts/evaluate.py \
  --config configs/eval/probing.yaml \
  --checkpoint checkpoints/m5s1k1/final.pt
```

Results are appended to `probing/results/probing_results.csv`.

Single dataset:

```bash
python probing/run_probing.py --config configs/eval/probing.yaml \
  --models lvcg --dataset ptbxl_super_class --ratio 0.1 --seed 42
```

## Repository layout

```
lvcg/                 # Model, data pipeline, training utilities
scripts/              # train.py, evaluate.py, data helpers
configs/              # train and eval YAML
probing/              # Linear probing + data_splits/
docs/                 # DATA_PREPARATION.md, ARCHITECTURE.md
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the LVCG data flow and model design.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{huang2026lvcg,
  title     = {Learning Cardiac Latent Representations in Vectorcardiogram Space},
  author    = {Huang, Bosong and Zhao, Panzhen and Li, Zengxiang and Lee, Patricia
               and Jin, Wei and Liew, Alan Wee-Chung and Jin, Ming and Pan, Shirui},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
