"""
Lightweight CSV-backed catalog loader for lookup values.

Used by the query builder to suggest/validate filter values based on real data
exports (countries, vessel types, waste types, facilities, fuel types).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set
import difflib


def _read_semicolon_csv(path: Path) -> List[Dict[str, str]]:
    """Read a semicolon-delimited CSV with optional UTF-8 BOM."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


@dataclass
class DataCatalog:
    """Access to small reference datasets exported from the CDNI system."""

    base_path: Path = Path("data")

    def __post_init__(self):
        self._cache: Dict[str, List[Dict[str, str]]] = {}

    # ---------- Loading helpers ----------
    def _load(self, filename: str) -> List[Dict[str, str]]:
        if filename not in self._cache:
            path = self.base_path / filename
            if not path.exists():
                self._cache[filename] = []
            else:
                self._cache[filename] = _read_semicolon_csv(path)
        return self._cache[filename]

    # ---------- Country helpers ----------
    def country_records(self) -> List[Dict[str, str]]:
        return self._load("country.csv")

    def country_tokens(self) -> Set[str]:
        tokens: Set[str] = set()
        for row in self.country_records():
            for key in ("Name", "ISO2", "ISO3"):
                val = (row.get(key) or "").strip()
                if val:
                    tokens.add(val.lower())
        return tokens

    def closest_countries(self, value: str, n: int = 3) -> List[str]:
        value_lower = value.lower()
        candidates = [row.get("Name", "") for row in self.country_records()]
        return difflib.get_close_matches(value_lower, [c.lower() for c in candidates], n=n)

    # ---------- Waste helpers ----------
    def waste_type_records(self) -> List[Dict[str, str]]:
        return self._load("wastetype.csv")

    def waste_type_names(self) -> List[str]:
        return [row.get("Name", "") for row in self.waste_type_records() if row.get("Name")]

    def waste_type_with_units(self) -> List[str]:
        """Return waste type labels including unit id if available."""
        labels = []
        for row in self.waste_type_records():
            name = row.get("Name", "")
            uom = row.get("UnitOfMeasurementId") or ""
            if name:
                label = name if not uom else f"{name} (UoM {uom})"
                labels.append(label)
        return labels

    # ---------- Facility helpers ----------
    def facility_records(self) -> List[Dict[str, str]]:
        return self._load("gosfacility.csv")

    def facility_names(self, limit: Optional[int] = None) -> List[str]:
        names = [row.get("Name", "").strip() for row in self.facility_records() if row.get("Name")]
        return names[:limit] if limit else names

    # ---------- Vessel type helpers ----------
    def vessel_type_names(self) -> List[str]:
        rows = self._load("vesseltype.csv")
        return [row.get("Name", "") for row in rows if row.get("Name")]

    # ---------- Fuel type helpers ----------
    def fuel_type_names(self) -> List[str]:
        rows = self._load("fueltype.csv")
        return [row.get("Name", "") for row in rows if row.get("Name")]

    # ---------- Generic fuzzy matching ----------
    @staticmethod
    def closest(values: Sequence[str], query: str, n: int = 3) -> List[str]:
        return difflib.get_close_matches(query.lower(), [v.lower() for v in values], n=n)
