from dataclasses import dataclass

@dataclass
class TrainingConfig:
    label_col: str = "label"
    positive_label: str = "pathological"

    test_size: float = 0.15
    validation_size: float = 0.15
    random_state: int = 42

    balance_train: bool = False

    svm_c_values: tuple[int, ...] = (128, 64)
    mlp_alpha: float = 0.0001
    mlp_lr_init: float = 0.001
    mlp_max_iter: int = 500
    mlp_early_stoping:bool = True