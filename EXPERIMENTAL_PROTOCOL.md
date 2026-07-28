# Confirmatory experimental protocol

## Scope

Only experiments executed with the CUDA backend under protocol
`gpu_confirmatory_v2` are eligible for the final paper. The scikit-learn
CPU MLP remains a development fallback and must not be pooled with or
compared directly against the confirmatory PyTorch results.

The protocol must be frozen before the final experiment batch. Any
change to preprocessing, model grids, cohort rules, splitting, metrics,
or selection policy requires a new protocol version.

## Shared preprocessing

All confirmatory model families use the same preprocessing search space
inside grouped cross-validation:

| Step | Candidates |
|---|---|
| Imputation | Median |
| Scaling | `StandardScaler` |
| Feature selection | ANOVA F-test at 10%, 25%, or 50% |

This shared space prevents a model from receiving a preprocessing
advantage unrelated to its architecture. Feature selection is fitted
only on the corresponding training fold.

## Model-specific search spaces

Model-specific differences are limited to parameters that represent
the capacity or optimization of that architecture.

### Linear SVM

- `C`: 0.01, 0.1, 1, or 10.
- Balanced class weights.

The linear SVM is the lower-capacity margin baseline. `C` controls the
trade-off between margin width and training errors.

### RBF SVM

- `C`: 0.01, 0.1, 1, or 10.
- `gamma`: `scale`, 0.0001, 0.001, or 0.01.
- Balanced class weights.

The RBF model represents nonlinear acoustic interactions. The grid
includes low `C` and `gamma` values so that smooth, strongly regularized
boundaries are explicitly considered.

### PyTorch MLP

- Moderate profile architecture: 8 or 16 units.
- Strong profile architectures: 8 units, 16 units, or tapered 16 → 8.
- Epoch budget: 5, 10, 15, or 20.
- Batch size: 32.
- Learning rate: 0.001.
- Moderate profile: dropout 0.2, weight decay 0.001, label smoothing
  0.05.
- Strong profile: dropout 0.4, weight decay 0.01, label smoothing 0.10.
- Class-weighted cross-entropy.

The MLP is intentionally compact because the number of acoustic
features is large relative to the HUPA cohort. The tapered 16 → 8
architecture addresses the prespecified deeper-network hypothesis only
under strong regularization. Epoch budget is selected by grouped source
CV instead of a random internal early-stopping split. Removing the
30-epoch candidate keeps the number of CUDA grid candidates unchanged.
The bundled regularization profiles must be interpreted as profiles,
not as an ablation of their individual components.

## Selection policy

Balanced accuracy is the primary selection metric. The
one-standard-error rule is applied within each model family, with a
minimum tolerance of 0.005 balanced-accuracy points.

There is no fixed preference between linear SVM, RBF SVM, and MLP.
Model-family candidates within the global acceptance threshold are
resolved using:

1. Lower grouped-CV standard deviation.
2. Smaller absolute training–CV generalization gap.
3. Fewer selected features.
4. Higher mean grouped-CV balanced accuracy.

The numerical maximum and the parsimonious selection are both retained
in the artifacts.

## Evaluation

- Holdout and external databases are never used for selection.
- All available speaker groups remain disjoint across partitions.
- Repeated nested grouped CV uses three outer folds, two repetitions,
  and five inner folds.
- The global champion is refitted on the complete source training
  partition and remains the single primary result.
- As a prespecified secondary analysis, the training-CV-selected SVM
  and MLP champions are refitted on the same complete training
  partition and evaluated on the holdout or external database.
- Candidate rankings use training CV only; holdout metrics are not used
  to choose or rank models.
- Cross-database results are interpreted as transportability estimates;
  performance loss may combine model overfitting and database shift.

## Seed and runtime policy

- The final grouped holdout is created once with seed 42.
- Inner grouped CV, model initialization, and grouped bootstrap use the
  prespecified master seed 42.
- Repeated nested grouped CV uses outer seeds 42 and 43, producing six
  outer estimates.
- Every nested estimate is retained and summarized; no best seed or run
  is selected.
- The two nested repetitions are a time-bounded compromise. Increasing
  the number after observing results would require a new protocol
  version.
- Multiple full holdout executions and multiple MLP restarts are not
  part of the primary protocol.

## Reproducibility

Each training directory contains:

```text
experimental_protocol.json
experimental_protocol.md
```

The JSON contains the resolved configuration, feature scenarios, model
grids, rationales, seed policy, deterministic CUDA policy, execution
environment, and a SHA-256 protocol hash. PyTorch requests deterministic
algorithms, disables cuDNN benchmarking, and records warnings rather
than aborting if an operation has no deterministic CUDA implementation.
Metrics and selection tables include the same version and hash. Results
with different hashes must not be combined as if they came from the
same protocol.
