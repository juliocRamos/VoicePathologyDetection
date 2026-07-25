from enum import Enum


class ExperimentStage(str, Enum):
    PREPARE = "prepare"
    FEATURES = "features"
    TRAIN = "train"

    @property
    def includes_feature_extraction(self) -> bool:
        return self in {
            ExperimentStage.FEATURES,
            ExperimentStage.TRAIN,
        }

    @property
    def includes_training(self) -> bool:
        return self is ExperimentStage.TRAIN
