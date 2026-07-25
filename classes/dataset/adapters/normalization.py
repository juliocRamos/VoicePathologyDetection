from pathlib import Path
import re
import unicodedata

import pandas as pd


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_recording_id(value) -> str:
    if pd.isna(value):
        return ""

    return re.sub(r"\.0$", "", str(value).strip())


def normalize_filename_key(filename: str) -> str:
    if pd.isna(filename):
        return ""

    return normalize_text(Path(str(filename)).stem)
