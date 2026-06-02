from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class DatasetVisualizer:
    def __init__(self, manifest: pd.DataFrame, output_dir: Path | None = None):
        self.manifest = manifest.copy()
        self.output_dir = output_dir

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_or_show(self, filename: str | None = None) -> None:
        plt.tight_layout()

        if self.output_dir is not None and filename is not None:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")

        plt.close()

    @staticmethod
    def _add_bar_labels(ax, fmt: str = "{:.0f}") -> None:
        for container in ax.containers:
            ax.bar_label(
                container,
                labels=[
                    fmt.format(value.get_height())
                    for value in container
                ],
                padding=3,
                fontsize=9,
            )

    @staticmethod
    def _add_horizontal_bar_labels(ax, fmt: str = "{:.0f}") -> None:
        for container in ax.containers:
            ax.bar_label(
                container,
                labels=[
                    fmt.format(value.get_width())
                    for value in container
                ],
                padding=3,
                fontsize=9,
            )

    @staticmethod
    def _add_histogram_labels(ax, fmt: str = "{:.0f}") -> None:
        for patch in ax.patches:
            height = patch.get_height()

            if height <= 0:
                continue

            x = patch.get_x() + patch.get_width() / 2

            ax.text(
                x,
                height,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    @staticmethod
    def _add_boxplot_median_labels(ax, df: pd.DataFrame, value_col: str, group_col: str) -> None:
        medians = df.groupby(group_col)[value_col].median()

        for index, (_, median_value) in enumerate(medians.items(), start=1):
            ax.text(
                index,
                median_value,
                f"{median_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    @staticmethod
    def _add_boxplot_stats_to_xticks(
            ax,
            df: pd.DataFrame,
            value_col: str,
            group_col: str,
            value_name: str = "mediana",
            decimals: int = 2,
            unit: str = "",
    ) -> None:
        stats = (
            df.groupby(group_col)[value_col]
            .agg(["median", "count"])
            .copy()
        )

        stats.index = stats.index.map(str)

        current_labels = [
            tick.get_text()
            for tick in ax.get_xticklabels()
        ]

        new_labels = []

        for label in current_labels:
            if label in stats.index:
                median_value = stats.loc[label, "median"]
                count_value = int(stats.loc[label, "count"])

                new_labels.append(
                    f"{label}\n"
                    f"{value_name}={median_value:.{decimals}f}{unit}\n"
                    f"n={count_value}"
                )
            else:
                new_labels.append(label)

        ax.set_xticklabels(new_labels)

    def plot_class_distribution(self) -> None:
        counts = self.manifest["label"].value_counts(dropna=False)

        ax = counts.plot(kind="bar", figsize=(8, 5))
        ax.set_title("Distribuição das amostras por classe")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Qtd. amostras")

        self._add_bar_labels(ax)

        self._save_or_show(filename="class_distribution.png")

    def plot_sex_distribution(self) -> None:
        counts = self.manifest["sex"].value_counts(dropna=False)

        ax = counts.plot(kind="bar", figsize=(8, 5))
        ax.set_title("Distribuição das amostras por gênero")
        ax.set_xlabel("Gênero")
        ax.set_ylabel("Qtd. amostras")

        self._add_bar_labels(ax)

        self._save_or_show(filename="gender_distribution.png")

    def plot_class_by_gender(self) -> None:
        table = pd.crosstab(self.manifest["sex"], self.manifest["label"])

        ax = table.plot(kind="bar", figsize=(9, 5))
        ax.set_title("Distribuição das classes por gênero")
        ax.set_xlabel("Gênero")
        ax.set_ylabel("Qtd. amostras")
        ax.legend(title="Classe", loc="upper left")

        self._add_bar_labels(ax)

        self._save_or_show("class_by_gender.png")

    def plot_duration_distribution(self) -> None:
        df = self.manifest.dropna(subset=["duration"]).copy()

        ax = df["duration"].plot(kind="hist", bins=30, figsize=(10, 5))
        ax.set_title("Distribuição da duração dos sinais")
        ax.set_xlabel("Duração (s)")
        ax.set_ylabel("Qtd. amostras")

        self._add_histogram_labels(ax)

        self._save_or_show(filename="duration_distribution.png")

    def plot_duration_by_class(self) -> None:
        df = self.manifest.dropna(subset=["duration", "label"]).copy()

        ax = df.boxplot(
            column="duration",
            by="label",
            figsize=(8, 5),
        )

        ax.set_title("Distribuição da duração dos sinais por classe")
        plt.suptitle("")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Duração (s)")

        self._add_boxplot_stats_to_xticks(
            ax=ax,
            df=df,
            value_col="duration",
            group_col="label",
            value_name="mediana",
            decimals=2,
            unit="s",
        )

        self._save_or_show("duration_by_class.png")

    def plot_age_by_class(self) -> None:
        df = self.manifest.dropna(subset=["age", "label"]).copy()

        ax = df.boxplot(column="age", by="label", figsize=(8, 5))
        ax.set_title("Distribuição da idade por classe")
        plt.suptitle("")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Idade")

        self._add_boxplot_median_labels(
            ax=ax,
            df=df,
            value_col="age",
            group_col="label",
        )

        self._save_or_show("age_by_class.png")

    def plot_top_pathologies(self, top_n: int = 15) -> None:
        df = self.manifest.dropna(subset=["pathology"]).copy()

        counts = df["pathology"].value_counts().head(top_n)
        counts = counts.sort_values()

        ax = counts.plot(kind="barh", figsize=(10, 7))
        ax.set_title(f"{top_n} condições clínicas mais frequentes na base HUPA")
        ax.set_xlabel("Qtd. amostras")
        ax.set_ylabel("Patologia")

        self._add_horizontal_bar_labels(ax)

        self._save_or_show("top_pathologies.png")

    def plot_age_vs_duration_by_class(self) -> None:
        df = self.manifest.dropna(subset=["age", "duration", "label"]).copy()

        fig, ax = plt.subplots(figsize=(8, 5))

        for label, group in df.groupby("label"):
            ax.scatter(
                group["age"],
                group["duration"],
                alpha=0.7,
                label=f"{label} (n={len(group)})",
            )

        ax.set_title("Relação entre idade e duração dos sinais")
        ax.set_xlabel("Idade")
        ax.set_ylabel("Duração (s)")
        ax.legend(title="Classe")

        self._save_or_show("age_vs_duration_by_class.png")

    def plot_grbas_by_class(self) -> None:
        grbas_cols = ["grbas_g", "grbas_r", "grbas_a", "grbas_b", "grbas_s"]

        available_cols = [col for col in grbas_cols if col in self.manifest.columns]

        if not available_cols:
            raise ValueError("No GRBAS column found in manifest.")

        df = self.manifest.dropna(subset=["label"]).copy()

        table = df.groupby("label")[available_cols].mean().T

        ax = table.plot(kind="bar", figsize=(10, 5))
        ax.set_title("Valores médios das dimensões GRBAS por classe")
        ax.set_xlabel("Dimensões GRBAS")
        ax.set_ylabel("Valor médio")
        ax.legend(title="Classe")

        self._add_bar_labels(ax, fmt="{:.2f}")

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