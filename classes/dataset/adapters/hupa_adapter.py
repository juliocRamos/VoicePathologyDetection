from pathlib import Path

import pandas as pd

from classes.audio_sample.audio_loader.audio_file_reader import AudioFileReader
from classes.dataset.adapters.dataset_adapter import DatasetAdapter
from classes.dataset.adapters.normalization import (
    normalize_filename_key,
    normalize_text,
)


class HUPADatasetAdapter(DatasetAdapter):
    def __init__(
        self,
        root_dir,
        metadata_filename="HUPA segmentada.xls",
        audio_reader: AudioFileReader | None = None,
    ):
        super().__init__(root_dir)

        self.root_dir = Path(root_dir)
        self.metadata_path = self.root_dir / metadata_filename
        self.audio_reader = audio_reader or AudioFileReader()

    def build_audio_index(self) -> pd.DataFrame:
        records = []

        for wav_path in sorted(self.root_dir.rglob("*.wav")):
            relative_path = wav_path.relative_to(self.root_dir)
            folder = relative_path.parts[0] if len(relative_path.parts) > 1 else None

            try:
                audio = self.audio_reader.read(wav_path)
                samplerate = audio.sample_rate
                duration = audio.duration
                channels = audio.channels
                audio_read_status = "ok"
                audio_read_error = None
            except Exception as exc:
                samplerate = None
                duration = None
                channels = None
                audio_read_status = "error"
                audio_read_error = str(exc)

            file_key = normalize_filename_key(wav_path.name)
            label = self._infer_label_from_relative_path(relative_path)

            sample_id = self._build_sample_id(relative_path)

            records.append({
                "sample_id": sample_id,
                "base": "HUPA",
                "filepath": str(wav_path),
                "relative_path": str(relative_path),
                "filename": wav_path.name,
                "file_stem": wav_path.stem,
                "file_key": file_key,
                "folder": folder,
                "label": label,
                "vowel": "a",
                "pitch": None,
                "samplerate": samplerate,
                "duration": duration,
                "channels": channels,
                "audio_read_status": audio_read_status,
                "audio_read_error": audio_read_error,
            })

        audio_df = pd.DataFrame(records)

        if audio_df["sample_id"].duplicated().any():
            duplicated = audio_df[audio_df["sample_id"].duplicated(keep=False)]
            raise ValueError(
                "Duplicated sample_id values found in audio index:\n"
                f"{duplicated[['sample_id', 'relative_path']].head(30)}"
            )

        return audio_df

    def load_metadata_sheet(self, sheet_name: str, label: str) -> pd.DataFrame:
        df = pd.read_excel(
            self.metadata_path,
            sheet_name=sheet_name,
            header=1,
        )

        df = df.dropna(how="all").copy()

        df = df.rename(columns={
            "Archivo": "metadata_filename",
            "Fs": "metadata_fs",
            "Tipo": "audio_type",
            "EGG": "egg",
            "edad": "age",
            "sexo": "sex",
            "G": "grbas_g",
            "R": "grbas_r",
            "A": "grbas_a",
            "B": "grbas_b",
            "S": "grbas_s",
            "Total": "grbas_total",
            "Codigo": "pathology_code",
            "Patología": "pathology",
            "F0": "f0_metadata",
            "F1": "f1_metadata",
            "F2": "f2_metadata",
            "F3": "f3_metadata",
            "Formantes": "formants_quality",
            "Picos": "peaks_quality",
            "Jitter": "jitter_quality",
            "Comentarios": "comments",
        })

        string_columns = [
            "metadata_filename",
            "audio_type",
            "egg",
            "sex",
            "pathology_code",
            "pathology",
            "formants_quality",
            "peaks_quality",
            "jitter_quality",
            "comments",
        ]

        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: pd.NA if pd.isna(x) else str(x).strip()
                )

        numeric_columns = [
            "metadata_fs",
            "age",
            "grbas_g",
            "grbas_r",
            "grbas_a",
            "grbas_b",
            "grbas_s",
            "grbas_total",
            "f0_metadata",
            "f1_metadata",
            "f2_metadata",
            "f3_metadata",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["metadata_file_key"] = df["metadata_filename"].apply(normalize_filename_key)
        df["label"] = label
        df["metadata_sheet"] = sheet_name

        return df

    def load_metadata(self) -> pd.DataFrame:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        normal_df = self.load_metadata_sheet("Normales", label="healthy")
        pathological_df = self.load_metadata_sheet("Patológicos", label="pathological")

        metadata_df = pd.concat(
            [normal_df, pathological_df],
            ignore_index=True,
        )

        if "sex" in metadata_df.columns:
            metadata_df["sex"] = metadata_df["sex"].replace({
                "H": "male",
                "M": "female",
                "h": "male",
                "m": "female",
            })

        return metadata_df

    def build_manifest(self) -> pd.DataFrame:
        audio_df = self.build_audio_index()
        metadata_df = self.load_metadata()

        self._print_key_diagnostics(audio_df, metadata_df)

        self._validate_composite_keys(
            audio_df=audio_df,
            metadata_df=metadata_df,
        )

        manifest = audio_df.merge(
            metadata_df,
            left_on=["file_key", "label"],
            right_on=["metadata_file_key", "label"],
            how="left",
            suffixes=("", "_metadata"),
            validate="one_to_one",
        )

        if len(manifest) != len(audio_df):
            raise ValueError(
                f"Manifest size mismatch. "
                f"audio_df={len(audio_df)}, manifest={len(manifest)}"
            )

        manifest["speaker_id"] = manifest["sample_id"]
        manifest["speaker_id_source"] = "sample_id_assumption"

        # Toda a HUPA segmentada utilizada neste experimento corresponde à vogal /a/
        manifest["vowel"] = "a"

        return manifest

    def validate_manifest(self, manifest: pd.DataFrame) -> None:
        print("Total indexed files:", len(manifest))

        if "sample_id" in manifest.columns:
            print("Unique sample_id:", manifest["sample_id"].nunique())
            print("Duplicated sample_id:", manifest["sample_id"].duplicated().sum())

        print("With metadata:", manifest["metadata_file_key"].notna().sum())
        print("Without metadata:", manifest["metadata_file_key"].isna().sum())

        print("\nDistribution by class:")
        print(manifest["label"].value_counts(dropna=False))

        print("\nDistribution by sex:")
        print(manifest["sex"].value_counts(dropna=False))

        print("\nDistribution by pathology:")
        print(manifest["pathology"].value_counts(dropna=False).head(20))

        unmatched = manifest[manifest["metadata_file_key"].isna()]

        if len(unmatched) > 0:
            print("\nFiles without metadata:")
            print(unmatched[["relative_path", "filename", "file_key", "label"]].head(30))

    @staticmethod
    def _build_sample_id(relative_path: Path) -> str:
        normalized_path = normalize_text(str(relative_path.with_suffix("")))
        return f"hupa_{normalized_path}"

    @staticmethod
    def _infer_label_from_relative_path(relative_path: Path) -> str | None:
        path_parts = [
            normalize_text(part)
            for part in relative_path.parts
        ]

        path_text = "_".join(path_parts)

        pathological_markers = {
            "pathol",
            "pathological",
            "patologico",
            "patologicos",
            "patologica",
            "patologicas",
            "patolog",
        }

        healthy_markers = {
            "normal",
            "normales",
            "healthy",
        }

        if any(marker in path_text for marker in pathological_markers):
            return "pathological"

        if any(marker in path_text for marker in healthy_markers):
            return "healthy"

        return None

    @staticmethod
    def _print_key_diagnostics(
        audio_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
    ) -> None:
        print("\nAudio index shape:", audio_df.shape)
        print("Metadata shape:", metadata_df.shape)

        print("\nAudio unique file_key:", audio_df["file_key"].nunique())
        print("Metadata unique metadata_file_key:", metadata_df["metadata_file_key"].nunique())

        print("\nAudio duplicated file_key:", audio_df["file_key"].duplicated().sum())
        print("Metadata duplicated metadata_file_key:", metadata_df["metadata_file_key"].duplicated().sum())

        audio_composite_dups = audio_df.duplicated(
            subset=["file_key", "label"],
            keep=False,
        )

        metadata_composite_dups = metadata_df.duplicated(
            subset=["metadata_file_key", "label"],
            keep=False,
        )

        print("\nAudio duplicated composite keys [file_key, label]:", audio_composite_dups.sum())
        print("Metadata duplicated composite keys [metadata_file_key, label]:", metadata_composite_dups.sum())

        if audio_composite_dups.any():
            print("\nDuplicated audio composite keys:")
            print(
                audio_df.loc[
                    audio_composite_dups,
                    ["file_key", "label", "relative_path", "filename"],
                ]
                .sort_values(["file_key", "label"])
                .head(50)
            )

        if metadata_composite_dups.any():
            print("\nDuplicated metadata composite keys:")
            cols = [
                "metadata_file_key",
                "label",
                "metadata_filename",
                "sex",
                "age",
                "pathology",
            ]
            existing_cols = [col for col in cols if col in metadata_df.columns]

            print(
                metadata_df.loc[metadata_composite_dups, existing_cols]
                .sort_values(["metadata_file_key", "label"])
                .head(50)
            )

    @staticmethod
    def _validate_composite_keys(
        audio_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
    ) -> None:
        missing_audio_labels = audio_df["label"].isna().sum()

        if missing_audio_labels > 0:
            missing = audio_df[audio_df["label"].isna()]
            raise ValueError(
                "Some audio files have no inferred label from path:\n"
                f"{missing[['relative_path', 'filename', 'file_key']].head(30)}"
            )

        audio_dups = audio_df[
            audio_df.duplicated(subset=["file_key", "label"], keep=False)
        ].sort_values(["file_key", "label"])

        metadata_dups = metadata_df[
            metadata_df.duplicated(subset=["metadata_file_key", "label"], keep=False)
        ].sort_values(["metadata_file_key", "label"])

        if not audio_dups.empty:
            raise ValueError(
                "Duplicated composite keys found in audio_df. "
                "The pair [file_key, label] is still ambiguous:\n"
                f"{audio_dups[['file_key', 'label', 'relative_path', 'filename']].head(50)}"
            )

        if not metadata_dups.empty:
            cols = [
                "metadata_file_key",
                "label",
                "metadata_filename",
                "sex",
                "age",
                "pathology",
            ]
            existing_cols = [col for col in cols if col in metadata_df.columns]

            raise ValueError(
                "Duplicated composite keys found in metadata_df. "
                "The pair [metadata_file_key, label] is still ambiguous:\n"
                f"{metadata_dups[existing_cols].head(50)}"
            )
