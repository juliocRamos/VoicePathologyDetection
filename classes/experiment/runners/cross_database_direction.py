from __future__ import annotations

from enum import Enum


class CrossDatabaseDirection(str, Enum):
    BOTH = "both"
    HUPA_TO_SVD = "hupa-to-svd"
    SVD_TO_HUPA = "svd-to-hupa"

    def database_pairs(self) -> tuple[tuple[str, str], ...]:
        if self is CrossDatabaseDirection.HUPA_TO_SVD:
            return (("HUPA", "SVD"),)

        if self is CrossDatabaseDirection.SVD_TO_HUPA:
            return (("SVD", "HUPA"),)

        return (
            ("HUPA", "SVD"),
            ("SVD", "HUPA"),
        )
