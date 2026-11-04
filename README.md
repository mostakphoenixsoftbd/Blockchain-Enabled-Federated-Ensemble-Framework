# Blockchain-governed federation of tree ensembles (BFE)

Code for the study *Blockchain-Governed Federation of Tree Ensembles for In-Hospital Mortality
Prediction: Poisoning Resistance, Calibration and Cost in Simulated MIMIC-IV Hospital Consortia*.

Hospitals train XGBoost and random-forest models locally, commit SHA-256 digests of the models to a
Solidity smart contract (`contracts/FederatedAuditRegistry.sol`), post peer validation AUROCs bound to
those digests, and the contract admits a model if its median peer AUROC is at least 0.70. Admitted
models are pooled with federated recalibration. The code simulates four-hospital consortia from
MIMIC-IV v3.1 and compares BFE with centralised, local-only, FedAvg, FedProx, robust FedAvg
(median, trimmed mean, Multi-Krum) and federated logistic regression under seven attack scenarios.

## Data

No patient data are included. MIMIC-IV v3.1 is available to credentialed users from PhysioNet
(https://doi.org/10.13026/kpb9-mt58) under the PhysioNet Credentialed Health Data Use Agreement.
Set `MIMIC_DIR` to the folder that contains the `hosp` and `icu` sub-folders. Do not share the
derived files the pipeline writes to `data/` or the prediction files in `results/runs/`.

## Requirements

Python 3.14 with the packages in `requirements.txt` (CPU only). The contract is compiled with
solc 0.8.24 through py-solc-x and run on a local Ethereum Virtual Machine (eth-tester / py-evm).
Peak memory is about 3 GB per run; run configurations one at a time.

## Pipeline (run from `src/`)

1. `python 01_build_cohort.py` — cohort and features (`data/cohort.parquet`).
2. `python 01b_build_ext_features.py [--chart]` — optional extended predictors.
3. `bash run_all.sh` — all configurations (simulated hospitals at several skew levels with attacks,
   data-scarce regimes, care-era partition, extended attacks and calibration variants); skips finished runs.
4. `python 02b_robust_fl.py --seed N` — median FedAvg comparator.
5. `python 03_chain_benchmark.py` — gas and latency scaling with consortium size, tamper test.
6. `python 04_analyze.py`, `04b_extra_stats.py`, `04c_ext_features.py`, `04d_revision_stats.py`,
   `04e_revision_figs.py` — tables and figures.

`results/tables/` contains the aggregate tables reported in the paper (means over seeds).

## Licence

MIT (see `LICENSE`).
