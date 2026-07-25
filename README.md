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
```

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
