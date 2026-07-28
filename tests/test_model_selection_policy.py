import unittest

from sklearn.feature_selection import SelectPercentile, f_classif

from classes.experiment.training.model_selection_policy import (
    ModelSelectionPolicy,
)


class ModelSelectionPolicyTests(unittest.TestCase):
    def test_selects_simpler_candidate_within_one_standard_error(
        self,
    ) -> None:
        selection = ModelSelectionPolicy.select_grid_candidate(
            mean_scores=[0.800, 0.799, 0.750],
            std_scores=[0.050, 0.040, 0.010],
            params=[
                {
                    "classifier__C": 10.0,
                    "selector": SelectPercentile(
                        score_func=f_classif,
                        percentile=50,
                    ),
                },
                {
                    "classifier__C": 0.1,
                    "selector": SelectPercentile(
                        score_func=f_classif,
                        percentile=25,
                    ),
                },
                {
                    "classifier__C": 0.01,
                    "selector": SelectPercentile(
                        score_func=f_classif,
                        percentile=10,
                    ),
                },
            ],
            model_name="svm_rbf",
            cv_folds=5,
            minimum_score_tolerance=0.005,
        )

        self.assertEqual(selection.numerical_best_index, 0)
        self.assertEqual(selection.selected_index, 1)
        self.assertGreater(selection.selection_threshold, 0.75)

    def test_source_complexity_is_neutral_between_families(
        self,
    ) -> None:
        linear_key = (
            ModelSelectionPolicy.source_candidate_complexity_key(
                n_input_features=200,
                best_params={
                    "classifier__C": 1.0,
                    "selector": SelectPercentile(
                        score_func=f_classif,
                        percentile=50,
                    ),
                },
            )
        )
        rbf_key = (
            ModelSelectionPolicy.source_candidate_complexity_key(
                n_input_features=100,
                best_params={
                    "classifier__C": 1.0,
                    "classifier__gamma": 0.001,
                    "selector": SelectPercentile(
                        score_func=f_classif,
                        percentile=50,
                    ),
                },
            )
        )

        self.assertLess(rbf_key, linear_key)


if __name__ == "__main__":
    unittest.main()
