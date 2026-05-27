from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class DatasetVisualizer:
    def __init__(self, manifest: pd.DataFrame, output_dir: Path | None = None):
        self.manifest = manifest
        self.output_dir = output_dir

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)


    def _save_or_show(self, filename: str | None = None) -> None:
        plt.tight_layout()

        if self.output_dir is not None and filename is not None:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")

        plt.close()

    def plot_class_distribution(self) -> None:
        counts = self.manifest["label"].value_counts(dropna=False)

        counts.plot(kind="bar")
        plt.title("Class distribution")
        plt.xlabel("Class")
        plt.ylabel("Number of samples")

        self._save_or_show(filename="class_distribution.png")

    def plot_sex_distribution(self) -> None:
        counts = self.manifest["sex"].value_counts(dropna=False)

        counts.plot(kind="bar")
        plt.title("Gender distribution")
        plt.xlabel("Gender")
        plt.ylabel("Number of samples")

        self._save_or_show(filename="gender_distribution.png")

    def plot_class_by_gender(self) -> None:
        table = pd.crosstab(self.manifest["sex"], self.manifest["label"])

        table.plot(kind="bar")
        plt.title("Label by gender distribution")
        plt.xlabel("Gender")
        plt.ylabel("Number of samples")
        plt.legend(title="Class", loc="upper left")

        self._save_or_show("class_by_gender.png")

    def plot_duration_distribution(self) -> None:
        df = self.manifest.dropna(subset=["duration"])

        df["duration"].plot(kind="hist", bins=30, figsize=(10, 5))
        plt.title("Signal distribution distribution")
        plt.xlabel("Duration (s)")
        plt.ylabel("Number of samples")

        self._save_or_show(filename="duration_distribution.png")

    def plot_duration_by_class(self) -> None:
        df = self.manifest.dropna(subset=["duration", "label"])

        df.boxplot(column="duration", by="label")
        plt.title("Signal duration by class")
        plt.suptitle("")
        plt.xlabel("Class")
        plt.ylabel("Duration (s)")

        self._save_or_show("duration_by_class.png")

    def plot_age_by_class(self) -> None:
        df = self.manifest.dropna(subset=["age", "label"])

        df.boxplot(column="age", by="label")
        plt.title("Age by class distribution")
        plt.suptitle("")
        plt.xlabel("Class")
        plt.ylabel("Age")

        self._save_or_show("age_by_class.png")

    def plot_top_pathologies(self, top_n: int = 15) -> None:
        df = self.manifest.dropna(subset=["pathology"])

        counts = df["pathology"].value_counts().head(top_n)
        counts.sort_values().plot(kind="barh")

        plt.title(f"{top_n} most frequent pathologies")
        plt.xlabel("Number of samples")
        plt.ylabel("Pathology")

        self._save_or_show("top_pathologies.png")

    def plot_age_vs_duration_by_class(self) -> None:
        df = self.manifest.dropna(subset=["age", "duration", "label"])

        fig, ax = plt.subplots()

        for label, group in df.groupby("label"):
            ax.scatter(
                group["age"],
                group["duration"],
                alpha=0.7,
                label=label
            )

        ax.set_title("Relation between age and audio_loader signal duration")
        ax.set_xlabel("Age")
        ax.set_ylabel("Duration (s)")
        ax.legend(title="Class")

        self._save_or_show("age_vs_duration_by_class.png")

    def plot_grbas_by_class(self) -> None:
        grbas_cols = ["grbas_g", "grbas_r", "grbas_a", "grbas_b", "grbas_s"]

        available_cols = [col for col in grbas_cols if col in self.manifest.columns]

        if not available_cols:
            raise ValueError("No GRBAS column found in manifest.")

        df = self.manifest.dropna(subset=["label"])

        table = df.groupby("label")[available_cols].mean().T

        table.plot(kind="bar")
        plt.title("Mean GRBAS score by class")
        plt.xlabel("GRBAS dimension")
        plt.ylabel("Mean value")
        plt.legend(title="Class")

        self._save_or_show("grbas_by_class.png")

    def generate_basic_report(self) -> None:
        self.plot_class_distribution()
        self.plot_sex_distribution()
        self.plot_class_by_gender()
        self.plot_duration_distribution()
        self.plot_duration_by_class()
        self.plot_age_by_class()
        self.plot_top_pathologies()
        self.plot_age_vs_duration_by_class()
        self.plot_grbas_by_class()