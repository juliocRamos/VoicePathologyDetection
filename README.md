# VoicePathologyDetection

Master's degree project for binary voice-pathology detection with HUPA
and the Saarbruecken Voice Database (SVD).

Install the runtime dependencies in a Python 3.10+ environment:

```bash
python -m pip install -r requirements.txt
```

## Pipeline

The experiment pipeline separates physical dataset facts from
experiment-specific cohort decisions:

```text
DatasetAdapter
    -> raw manifest
TrainingManifestBuilder
    -> curated manifest + exclusions + duplicate audit
ManifestAudioSampleLoader
    -> AudioSample loaded from one manifest row
AudioPreprocessor
    -> deterministic per-sample signal transformations
FeatureExtractionRunner
    -> acoustic and glottal features
ModelTrainingRunner
    -> speaker-grouped holdout and grouped cross-validation
```

The raw manifests are never overwritten. Cohort builders own inclusion
criteria such as age, task, duration, file readability, and physical
deduplication. `AudioPreprocessor` owns only signal transformations and
technical signal invariants.

## Running an experiment

Preparation is the safe default:

```bash
.venv/bin/python main.py --dataset svd
```

Select the last stage explicitly when extracting features or training:

```bash
.venv/bin/python main.py --dataset svd --stage features
.venv/bin/python main.py --dataset svd --stage train
.venv/bin/python main.py --dataset hupa --stage train
.venv/bin/python main.py --dataset cross --stage train
.venv/bin/python main.py --dataset pooled --stage train
```

These four commands provide five comparable protocols. All use adults,
the sustained vowel `/a/`, and normal SVD phonation:

```text
1. HUPA       -> grouped HUPA holdout
2. SVD        -> grouped SVD holdout
3. HUPA       -> external test on all SVD
4. SVD        -> external test on all HUPA
5. HUPA + SVD -> grouped mixed holdout, reported globally and per database
```

In every protocol, held-out data are not used for model selection.
Feature scenario, model family, preprocessing, and hyperparameters are
selected by grouped cross-validation on the training partition. The
selected pipeline is refitted on that complete partition and evaluated
once on the holdout.

The `cross` command executes both external-validation directions:

```text
HUPA -> grouped CV selects one pipeline -> refit HUPA -> test once on SVD
SVD  -> grouped CV selects one pipeline -> refit SVD  -> test once on HUPA
```

Only sustained vowel `/a/` at the normal SVD condition is used in this
mode to match the HUPA vocal task, and both cohorts contain only adults
with valid age. The destination database is never used to choose the
feature scenario, model family, hyperparameters, imputer, scaler, or
feature selector. Training-only candidate rankings are saved as
`source_model_selection.csv`; only the selected pipeline is evaluated
on the destination database. Cross-database artifacts are written below
`data/CROSS_DATABASE/experiments/`.

The `pooled` command combines the harmonized HUPA and SVD cohorts.
Speaker groups are prefixed as `HUPA::<speaker_id>` and
`SVD::<speaker_id>`, preventing identifier collisions. Holdout and
inner-CV folds are stratified by the four database/class combinations
and remain speaker-disjoint. The selected model produces four metric
rows:

```text
overall
base:HUPA
base:SVD
base:macro
```

The macro row gives equal influence to both databases regardless of
their sample counts. Pooled artifacts are written below
`data/POOLED/experiments/`.

Experiment artifacts are written below:

```text
data/<DATASET>/experiments/<timestamp>_<dataset>_<experiment>/
```

Each prepared run contains the raw and curated manifests, excluded
samples, duplicate groups, a preparation summary, preprocessing
profiles, and figures. Training runs additionally contain split
assignments, fitted models, predictions, cross-validation results, and
test metrics.

## Validation guarantees

- SVD diagnostic copies are consolidated by SHA-256.
- HUPA physical duplicates are consolidated and conflicting clinical
  metadata are reported.
- All sessions from one SVD `speaker_id` remain in one partition.
- Imputation, scaling, feature selection, and model fitting occur inside
  cross-validation.
- Class balancing is applied only while fitting training folds.
- Confidence intervals are bootstrapped by speaker rather than by
  individual recording.
- Cross-database tests use only acoustic features available in both
  databases and save the audited schema in `splits/feature_schema.csv`.
- Cross-database reports compare age, sex, class, speaker count, and
  pathology distributions between the adult HUPA and SVD cohorts.
- Pooled training uses only numeric acoustic features available in both
  databases and saves `reports/pooled_feature_schema.csv`.
- Pooled holdout and CV folds contain every available database/class
  stratum and never split a prefixed speaker group.
- SMOTE is not used by default; class imbalance is handled with class
  or sample weights without synthesizing medical observations.

## Neural-network training curves

The MLP saves per-epoch CSV and PNG histories below
`training/figures/training_curves/`. A separate diagnostic fit uses a
speaker-grouped validation fold and records training/validation loss,
accuracy, and balanced accuracy for both the scikit-learn CPU MLP and
the PyTorch MLP. Its auditable group assignments are saved below
`training/splits/`.

The diagnostic fit exists only to visualize optimization and
overfitting. It does not replace the final pipeline, which is refitted
on all source training data before holdout or external evaluation.
Cross-database generalization must be read from the metrics and
prediction artifacts produced for each direction.

## Known HUPA identity limitation

The available segmented HUPA metadata does not expose an independent
speaker identifier. After physical deduplication, each canonical
acoustic file is therefore treated as one assumed speaker. This prevents
byte-identical files from crossing partitions, but it cannot prove that
two different recordings do not belong to the same person. Replace
`speaker_id` with a verified subject identifier if one becomes
available.

Run the unit tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
