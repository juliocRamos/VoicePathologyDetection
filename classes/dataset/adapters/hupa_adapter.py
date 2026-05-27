from pathlib import Path
import re
import unicodedata
import pandas as pd
import soundfile as sf

from classes.dataset.adapters.dataset_adapter import DatasetAdapter


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text

def normalize_filename_key(filename: str) -> str:
    return normalize_text(Path(str(filename)).stem)


class HUPADatasetAdapter(DatasetAdapter):
    def __init__(self, root_dir, metadata_filename="HUPA segmentada.xls"):
        super().__init__(root_dir)
        self.root_dir = Path(root_dir)
        self.metadata_path = self.root_dir / metadata_filename

    def build_audio_index(self) -> pd.DataFrame:
        records = []

        for wav_path in sorted(self.root_dir.rglob("*.wav")):
            relative_path = wav_path.relative_to(self.root_dir)
            folder = relative_path.parts[0] if len(relative_path.parts) > 1 else None

            try:
                info = sf.info(wav_path)
                samplerate = info.samplerate
                duration = info.duration
                channels = info.channels
            except Exception:
                samplerate = None
                duration = None
                channels = None

            file_key = normalize_filename_key(wav_path.name)

            records.append({
                "sample_id": f"hupa_{file_key}",
                "base": "HUPA",
                "filepath": str(wav_path),
                "relative_path": str(relative_path),
                "filename": wav_path.name,
                "file_stem": wav_path.stem,
                "file_key": file_key,
                "folder": folder,
                "vowel": "a",
                "pitch": None,
                "samplerate": samplerate,
                "duration": duration,
                "channels": channels,
            })

        return pd.DataFrame(records)

    def load_metadata_sheet(self, sheet_name: str, label: str) -> pd.DataFrame:
        df = pd.read_excel(
            self.metadata_path,
            sheet_name=sheet_name,
            header=1
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

        return df

    def load_metadata(self) -> pd.DataFrame:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Arquivo de metadados não encontrado: {self.metadata_path}")

        normal_df = self.load_metadata_sheet("Normales", label="healthy")
        pathological_df = self.load_metadata_sheet("Patológicos", label="pathological")

        metadata_df = pd.concat(
            [normal_df, pathological_df],
            ignore_index=True
        )

        metadata_df["sex"] = metadata_df["sex"].replace({
            "H": "male",
            "M": "female"
        })

        return metadata_df

    def build_manifest(self) -> pd.DataFrame:
        audio_df = self.build_audio_index()
        metadata_df = self.load_metadata()

        manifest = audio_df.merge(
            metadata_df,
            left_on="file_key",
            right_on="metadata_file_key",
            how="left",
            suffixes=("", "_metadata")
        )

        manifest["speaker_id"] = manifest["file_key"]

        # Como toda a HUPA segmentada aqui é vogal /a/
        manifest["vowel"] = "a"

        return manifest

    def validate_manifest(self, manifest: pd.DataFrame) -> None:
        print("Total indexed files:", len(manifest))
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
            print(unmatched[["relative_path", "filename", "file_key"]].head(30))