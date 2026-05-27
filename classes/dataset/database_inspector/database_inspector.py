from pathlib import Path
from typing import Counter
import soundfile as sf

import pandas as pd

from classes.dataset.database_inspector.database_inspector_report import DatabaseInspectorReport


class DatabaseInspector:
    AUDIO_EXTENSIONS = {
        ".wav": ".flac",
        ".mp3": ".ogg",
    }

    METADATA_EXTENSIONS = {
        ".csv", ".tsv",
        ".xlsx", ".xls",
        ".json", "txt"
    }

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"No such directory: {self.root_dir}")

    def list_files(self) -> list[Path]:
        return [p for p in self.root_dir.rglob("*") if p.is_file()]

    def list_audio_files(self) -> list[Path]:
        return [
            p for p in self.list_files()
            if p.suffix.lower() in self.AUDIO_EXTENSIONS
        ]

    def list_metadata_files(self) -> list[Path]:
        return [
            p for p in self.list_files()
            if p.suffix.lower() in self.METADATA_EXTENSIONS
        ]

    def inspect(self, max_examples: int = 30) -> DatabaseInspectorReport:
        files = self.list_files()
        audio_files = self.list_audio_files()
        metadata_files = self.list_metadata_files()

        extensions = Counter(p.suffix.lower() for p in files if p.suffix)

        return DatabaseInspectorReport(
            root_dir=self.root_dir,
            total_files=len(files),
            audio_files=len(audio_files),
            metadata_files=len(metadata_files),
            extensions=dict(extensions),
            audio_examples=[
                str(p.relative_to(self.root_dir))
                for p in audio_files[:max_examples]
            ],
            metadata_examples=[
                str(p.relative_to(self.root_dir))
                for p in metadata_files[:max_examples]
            ],
        )

    def audio_summary(self, max_files: int | None = None) -> pd.DataFrame:
        audio_files = self.list_audio_files()

        if max_files is not None:
            audio_files = audio_files[:max_files]

        records = []

        for path in audio_files:
            try:
                info = sf.info(path)

                records.append({
                    "filepath": str(path),
                    "relative_path": str(path.relative_to(self.root_dir)),
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "duration": info.duration,
                    "frames": info.frames,
                    "format": info.format,
                    "subtype": info.subtype,
                })
            except Exception as exc:
                records.append({
                    "filepath": str(path),
                    "relative_path": str(path.relative_to(self.root_dir)),
                    "error": str(exc)
                })

        return pd.DataFrame(records)


    def print_report(self, max_examples: int = 30) -> None:
        report = self.inspect(max_examples=max_examples)

        print(f"Directory: {self.root_dir}")
        print(f"Total files: {report.total_files}")
        print(f"Audio files: {report.audio_files}")
        print(f"Metadata files: {report.metadata_files}")

        print("\nExtensions found:")
        for ext, count in sorted(report.extensions.items()):
            print(f"     {ext}: {count}")


        print("\nAudio file examples:")
        for example in report.audio_examples:
            print(f"     {example}")


        print("\nMetadata file examples:")
        for example in report.metadata_examples:
            print(f"     {example}")