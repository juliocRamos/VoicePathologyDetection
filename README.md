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
.venv/bin/python main.py --dataset svd --stage train --compute-backend cuda
.venv/bin/python main.py --dataset hupa --stage train --compute-backend cuda
.venv/bin/python main.py --dataset cross --stage train --compute-backend cuda
.venv/bin/python main.py --dataset pooled --stage train --compute-backend cuda
```

New cross/pooled runs can reuse already validated feature artifacts
instead of preprocessing and extracting both databases again:

```bash
.venv/bin/python main.py \
  --dataset cross \
  --stage train \
  --compute-backend cuda \
  --hupa-source-experiment data/HUPA/experiments/<run-directory> \
  --svd-source-experiment data/SVD/experiments/<run-directory>
```

Interrupted feature/training runs can be resumed in the same experiment
directory:

```bash
.venv/bin/python main.py \
  --dataset svd \
  --stage train \
  --compute-backend cuda \
  --resume-experiment data/SVD/experiments/<run-directory>
```

Resume mode validates the persisted configuration and protocol hash,
reuses extracted features and completed model-selection candidates,
skips completed repeated nested-CV folds, and checkpoints every grouped
SVM learning-curve fit. A completed `metrics.csv` makes the resumed
training stage idempotent. Cross and pooled runs also reuse their
persisted source experiment roots and per-direction training outputs.

These four commands provide five comparable protocols. All use adults,
the sustained vowel `/a/`, and normal SVD phonation:

```text
1. HUPA       -> grouped HUPA holdout
2. SVD        -> grouped SVD holdout
3. HUPA       -> external test on all SVD
4. SVD        -> external test on all HUPA
5. HUPA + SVD -> grouped mixed holdout, reported globally and per database
```

CUDA executions under `gpu_confirmatory_v2` provide the original
confirmatory results. The complementary multivowel analysis uses
`gpu_multivowel_extension_v1`. CPU support remains available for
development but uses a different MLP implementation and must not be
mixed with CUDA results. The complete rationale is recorded in
`EXPERIMENTAL_PROTOCOL.md`.

A detailed, code-aligned description of data loading, cohort
preparation, signal preprocessing, feature extraction, training,
evaluation, and every overfitting-control decision is available in
[`PIPELINE_DOCUMENTATION.md`](PIPELINE_DOCUMENTATION.md).

In every protocol, held-out data are not used for model selection.
Feature scenario, model family, preprocessing, and hyperparameters are
selected by grouped cross-validation on the training partition. The
selected pipeline is refitted on that complete partition and remains
the single primary holdout result. A prespecified secondary analysis
also evaluates the training-CV-selected SVM and MLP champions on the
same holdout; holdout performance is never used to choose or rank
candidates.

Selection follows a one-standard-error rule. Configurations whose mean
grouped-CV score is statistically indistinguishable from the numerical
maximum are resolved in favor of lower capacity within each model
family: fewer selected features, smaller hidden layers, fewer
neural-network epochs, lower `C` and `gamma`, and stronger
regularization. Global comparison between model families is
family-neutral and prioritizes lower CV variability, a smaller
training–CV gap, and fewer selected features. A minimum balanced-
accuracy tolerance of `0.005` prevents negligible score differences
from selecting a more complex pipeline.

By default, the `cross` command executes both external-validation
directions:

```text
HUPA -> grouped CV selects one pipeline -> refit HUPA -> test once on SVD
SVD  -> grouped CV selects one pipeline -> refit SVD  -> test once on HUPA
```

Only sustained vowel `/a/` at the normal SVD condition is used in this
mode to match the HUPA vocal task, and both cohorts contain only adults
with valid age. The destination database is never used to choose the
feature scenario, model family, hyperparameters, imputer, scaler, or
feature selector. Training-only candidate rankings are saved as
`source_model_selection.csv`. Primary metrics remain in `metrics.csv`;
the prespecified SVM-versus-MLP analysis is saved separately as
`family_comparison_metrics.csv`. Cross-database artifacts are written
below `data/CROSS_DATABASE/experiments/`.

The optional multivowel SVD extension keeps only normal phonation while
including `/a/`, `/i/`, and `/u/`:

```bash
.venv/bin/python main.py \
  --dataset svd \
  --stage train \
  --compute-backend cuda \
  --svd-vowels a i u \
  --experiment-name svd_multivowel_gpu_confirmatory_v2
```

Its holdout remains speaker-disjoint and reports `overall`, `vowel:a`,
`vowel:i`, `vowel:u`, and `vowel:macro` metric rows. A directional
cross-database follow-up can reuse that completed SVD feature source
and an existing HUPA feature source:

```bash
.venv/bin/python main.py \
  --dataset cross \
  --stage train \
  --compute-backend cuda \
  --svd-vowels a i u \
  --cross-direction svd-to-hupa \
  --hupa-source-experiment data/HUPA/experiments/<run-directory> \
  --svd-source-experiment data/SVD/experiments/<multivowel-run-directory> \
  --experiment-name svd_multivowel_to_hupa_gpu_confirmatory_v2
```

The original `/a/`-only cohort and bidirectional cross protocol remain
the defaults. Resume validation rejects a source experiment whose SVD
vowel cohort differs from the requested configuration.

The default training protocol also runs repeated nested grouped CV only
on the source training partition (`3` outer folds × `2` repetitions,
with the regular `5` grouped folds used for inner selection). It
estimates source-domain performance and model-selection stability
without accessing the holdout or external database. This diagnostic
substantially increases training time. The two repetitions are retained
as a time-bounded compromise: they provide six outer estimates, which
are all reported, while the final holdout is evaluated once with the
prespecified seed `42`. No seed is selected by holdout performance.

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
test metrics. The resolved plan and its SHA-256 hash are stored in
`training/experimental_protocol.json` and
`training/experimental_protocol.md`.

## Validation guarantees

- SVD diagnostic copies are consolidated by SHA-256.
- HUPA physical duplicates are consolidated and conflicting clinical
  metadata are reported.
- All sessions from one SVD `speaker_id` remain in one partition.
- Imputation, scaling, feature selection, and model fitting occur inside
  cross-validation.
- Class balancing is applied only while fitting training folds.
- Grid-search artifacts include training scores, validation scores, and
  their generalization gap.
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

## Overfitting diagnostics

When an SVM is selected, a speaker-grouped learning curve is generated
at `25%`, `50%`, `75%`, and `100%` of the available training groups.
Raw fold results, aggregated mean/standard-deviation curves, and
auditable assignments are saved below
`training/figures/learning_curves/` and `training/splits/`.

Repeated nested-CV artifacts are saved as:

```text
training/metrics/repeated_nested_cv_results.csv
training/metrics/repeated_nested_cv_summary.csv
training/metrics/repeated_nested_cv_selection_stability.csv
training/splits/repeated_nested_cv_assignments.csv
```

The MLP saves per-epoch CSV and PNG histories below
`training/figures/training_curves/`. A separate diagnostic fit uses a
speaker-grouped validation fold and records training/validation loss,
accuracy, and balanced accuracy. Confirmatory results use only the
PyTorch CUDA MLP; the scikit-learn CPU implementation is a development
fallback. Auditable group assignments are saved below
`training/splits/`.

The diagnostic fit exists only to visualize optimization and
overfitting. It does not replace the final pipeline, which is refitted
on all source training data before holdout or external evaluation.
The MLP epoch budget (5, 10, 15, or 20 epochs) is selected by the same
speaker-grouped source cross-validation used for the other
hyperparameters. The moderate regularization profile evaluates one
hidden layer with 8 or 16 units. The strong profile additionally
evaluates the tapered `(16, 8)` architecture. Removing the 30-epoch
candidate keeps the total CUDA search budget unchanged while testing the
advisor-requested deeper MLP. CUDA candidates use mini-batches, mandatory
univariate feature selection, dropout, weight decay, and label smoothing;
no random internal validation split or group-unaware early stopping is
used.
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
