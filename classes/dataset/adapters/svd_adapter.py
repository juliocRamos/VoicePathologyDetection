from pathlib import Path
import re
import unicodedata
import pandas as pd

from classes.audio_sample.audio_loader.audio_file_reader import AudioFileReader
from classes.dataset.adapters.dataset_adapter import DatasetAdapter


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_recording_id(value) -> str:
    if pd.isna(value):
        return ""

    return re.sub(r"\.0$", "", str(value).strip())


class SVDAdapter(DatasetAdapter):
    DEFAULT_HEALTHY_GROUPS = {
        "healthy",
        "normal",
        "normale",
        "normales",
        "gesund",
        "gesunde",
        "stimmgesunde",
    }

    def __init__(
        self,
        root_dir: str | Path,
        healthy_pathology_groups: set[str] | None = None,
        audio_reader: AudioFileReader | None = None,
    ):
        super().__init__(root_dir)

        self.root_dir = Path(root_dir)
        self.audio_reader = audio_reader or AudioFileReader()

        self.healthy_pathology_groups = {
            normalize_text(group)
            for group in (
                healthy_pathology_groups or self.DEFAULT_HEALTHY_GROUPS
            )
        }

    def build_audio_index(self) -> pd.DataFrame:
        records = []

        for overview_path in sorted(self.root_dir.rglob("overview.csv")):
            pathology_dir = overview_path.parent
            pathology_group = pathology_dir.name
            pathology_group_key = normalize_text(pathology_group)

            for recording_dir in sorted(pathology_dir.iterdir()):
                if not recording_dir.is_dir():
                    continue

                vowels_dir = recording_dir / "vowels"

                if not vowels_dir.exists():
                    continue

                folder_recording_id = normalize_recording_id(recording_dir.name)

                for audio_path in sorted(vowels_dir.glob("*.nsp")):
                    parsed_name = self._parse_vowel_filename(audio_path)

                    if parsed_name is None:
                        print(f"[WARNING] Unexpected SVD filename: {audio_path}")
                        continue

                    filename_recording_id, vowel, condition = parsed_name

                    try:
                        audio = self.audio_reader.read(audio_path)

                        samplerate = audio.sample_rate
                        duration = audio.duration
                        channels = audio.channels

                        audio_read_status = "ok"
                        audio_read_error = None

                    except Exception as ex:
                        samplerate = None
                        duration = None
                        channels = None

                        audio_read_status = "error"
                        audio_read_error = str(ex)

                    relative_path = audio_path.relative_to(self.root_dir)
                    sample_id = self._build_sample_id(relative_path)

                    records.append({
                        "sample_id": sample_id,
                        "base": "SVD",
                        "filepath": str(audio_path),
                        "relative_path": str(relative_path),
                        "filename": audio_path.name,
                        "file_stem": audio_path.stem,
                        "pathology_group": pathology_group,
                        "pathology_group_key": pathology_group_key,
                        "recording_id": folder_recording_id,
                        "filename_recording_id": filename_recording_id,
                        "recording_id_matches_folder": (
                            folder_recording_id == filename_recording_id
                        ),
                        "label": self._infer_label(pathology_group_key),
                        "vowel": vowel,
                        "condition": condition,
                        "pitch": condition,
                        "samplerate": samplerate,
                        "duration": duration,
                        "channels": channels,
                        "audio_read_status": audio_read_status,
                        "audio_read_error": audio_read_error,
                    })

        audio_df = pd.DataFrame(records)

        if audio_df.empty:
            raise ValueError(f"No .nsp files found in: {self.root_dir}")

        if audio_df["sample_id"].duplicated().any():
            duplicated = audio_df[
                audio_df["sample_id"].duplicated(keep=False)
            ]

            raise ValueError(
                "Duplicated sample_id values found in SVD audio index:\n"
                f"{duplicated[['sample_id', 'relative_path']].head(30)}"
            )

        return audio_df

    def load_metadata(self) -> pd.DataFrame:
        records = []

        for overview_path in sorted(self.root_dir.rglob("overview.csv")):
            pathology_group = overview_path.parent.name
            pathology_group_key = normalize_text(pathology_group)

            df = pd.read_csv(
                overview_path,
                dtype={
                    "AufnahmeID": "string",
                    "SprecherID": "string",
                },
            ).copy()

            df = df.rename(columns={
                "AufnahmeID": "recording_id",
                "AufnahmeTyp": "recording_type",
                "AufnahmeDatum": "recording_date",
                "Diagnose": "diagnosis",
                "SprecherID": "speaker_id",
                "Geburtsdatum": "birth_date",
                "Geschlecht": "sex",
                "Pathologien": "pathology",
            })

            required_columns = {"recording_id", "speaker_id"}

            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f"Missing columns in {overview_path}: {missing_columns}"
                )

            df["recording_id"] = df["recording_id"].apply(
                normalize_recording_id
            )
            df["speaker_id"] = df["speaker_id"].apply(
                normalize_recording_id
            )

            df["recording_date"] = pd.to_datetime(
                df["recording_date"],
                errors="coerce",
            )
            df["birth_date"] = pd.to_datetime(
                df["birth_date"],
                errors="coerce",
            )

            df["age"] = (
                (df["recording_date"] - df["birth_date"]).dt.days
                / 365.2425
            ).round(1)

            df["sex"] = (
                df["sex"]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace({
                    "m": "male",
                    "w": "female",
                })
            )

            df["pathology_group"] = pathology_group
            df["pathology_group_key"] = pathology_group_key
            df["label"] = self._infer_label(pathology_group_key)
            df["metadata_path"] = str(overview_path)

            records.append(df)

        metadata_df = pd.concat(records, ignore_index=True)
        metadata_df = self._resolve_duplicate_metadata(metadata_df)
        return metadata_df

    def build_manifest(self) -> pd.DataFrame:
        audio_df = self.build_audio_index()
        metadata_df = self.load_metadata()

        manifest = audio_df.merge(
            metadata_df,
            on=["pathology_group_key", "recording_id", "label"],
            how="left",
            suffixes=("", "_metadata"),
            validate="many_to_one",
        )

        if len(manifest) != len(audio_df):
            raise ValueError(
                "Manifest size mismatch: "
                f"audio_df={len(audio_df)}, manifest={len(manifest)}"
            )

        manifest["pathology"] = manifest["pathology"].fillna(
            manifest["pathology_group"]
        )

        return manifest

    def validate_manifest(self, manifest: pd.DataFrame) -> None:
        print("Total indexed files:", len(manifest))
        print("Unique samples:", manifest["sample_id"].nunique())
        print("Unique recordings:", manifest["recording_id"].nunique())
        print("Unique speakers:", manifest["speaker_id"].nunique())

        print("\nDistribution by class:")
        print(manifest["label"].value_counts(dropna=False))

        print("\nDistribution by pathology group:")
        print(manifest["pathology_group"].value_counts().head(30))

        print("\nCoverage by vowel and condition:")
        print(pd.crosstab(manifest["vowel"], manifest["condition"]))

        print("\nMetadata match:")
        print(manifest["speaker_id"].notna().value_counts())

        mismatches = manifest[
            ~manifest["recording_id_matches_folder"]
        ]

        if not mismatches.empty:
            print("\nFilename/folder recording-id mismatches:")
            print(
                mismatches[
                    [
                        "relative_path",
                        "recording_id",
                        "filename_recording_id",
                    ]
                ].head(30)
            )

    def _infer_label(self, pathology_group_key: str) -> str:
        if pathology_group_key in self.healthy_pathology_groups:
            return "healthy"

        return "pathological"

    @staticmethod
    def _parse_vowel_filename(
        audio_path: Path,
    ) -> tuple[str, str, str] | None:
        filename = audio_path.name.strip().lower()

        # sustained vowels:
        sustained_match = re.fullmatch(
            r"(?P<recording_id>\d+)-"
            r"(?P<vowel>[aiu])_"
            r"(?P<condition>h|l|lhl|n)"
            r"\.nsp",
            filename,
        )

        if sustained_match is not None:
            return (
                normalize_recording_id(
                    sustained_match.group("recording_id")
                ),
                sustained_match.group("vowel"),
                sustained_match.group("condition")
            )

        # Articulatory sequence
        # 1242-iau.nsp
        sequence_match = re.fullmatch(
            r"(?P<recording_id>\d+)-iau\.nsp",
            filename
        )

        if sequence_match is not None:
            return (
                normalize_recording_id(
                    sequence_match.group("recording_id")
                ),
                "iau",
                "sequence"
            )

        return None

    @staticmethod
    def _build_sample_id(relative_path: Path) -> str:
        normalized_path = normalize_text(
            str(relative_path.with_suffix(""))
        )
        return f"svd_{normalized_path}"

    @staticmethod
    def _resolve_duplicate_metadata(
            metadata_df: pd.DataFrame,
    ) -> pd.DataFrame:
        key_columns = [
            "pathology_group_key",
            "recording_id",
        ]

        duplicated_mask = metadata_df.duplicated(
            subset=key_columns,
            keep=False,
        )

        if not duplicated_mask.any():
            return metadata_df

        duplicated_df = metadata_df.loc[
            duplicated_mask
        ].copy()

        rows_to_remove: list[int] = []
        conflicting_groups: list[pd.DataFrame] = []

        # Essas colunas indicam apenas a origem física do registro.
        # Diferenças nelas não significam necessariamente que os
        # metadados clínicos sejam diferentes.
        ignored_comparison_columns = {
            "metadata_path",
            "pathology_group",
        }

        comparison_columns = [
            column
            for column in metadata_df.columns
            if column not in ignored_comparison_columns
        ]

        grouped_duplicates = duplicated_df.groupby(
            key_columns,
            dropna=False,
            sort=False,
        )

        for _, group in grouped_duplicates:
            comparable_rows = (
                group[comparison_columns]
                .astype("string")
                .fillna("<NA>")
                .drop_duplicates()
            )

            if len(comparable_rows) == 1:
                # As linhas têm o mesmo conteúdo.
                # Mantém a primeira e remove as demais.
                rows_to_remove.extend(
                    group.index[1:].tolist()
                )
            else:
                # A chave é a mesma, mas algum metadado difere.
                conflicting_groups.append(group)

        if conflicting_groups:
            conflicts = pd.concat(
                conflicting_groups,
                ignore_index=False,
            )

            diagnostic_columns = [
                "pathology_group",
                "pathology_group_key",
                "recording_id",
                "recording_type",
                "recording_date",
                "diagnosis",
                "speaker_id",
                "birth_date",
                "age",
                "sex",
                "pathology",
                "metadata_path",
            ]

            diagnostic_columns = [
                column
                for column in diagnostic_columns
                if column in conflicts.columns
            ]

            raise ValueError(
                "Conflicting metadata rows found for the same "
                "[pathology_group_key, recording_id].\n"
                "These rows cannot be resolved automatically:\n\n"
                f"{conflicts[diagnostic_columns].to_string(index=False)}"
            )

        if rows_to_remove:
            print(
                "[WARNING] Removing "
                f"{len(rows_to_remove)} exactly duplicated "
                "metadata row(s)."
            )

            metadata_df = (
                metadata_df
                .drop(index=rows_to_remove)
                .reset_index(drop=True)
            )

        return metadata_df
