"""
Adapter to dump per-cell (U_i, Gamma_i) data from existing evaluation
pipeline into the cells.json format expected by run_experiment.py.

"""

import json
from pathlib import Path
from typing import Iterable, Optional, Union


# Default Gamma_i values used in the paper. Override per-call if costs differ
DEFAULT_GAMMAS = {
    "tau_agg_comp":    0.0,
    "tau_sens":        0.6,
    "tau_noise":       0.8,
    "tau_count":       0.4,
    "tau_group_count": 0.3,
    "tau_row_contrib": 0.6,
}


def classify_region_by_count(taxi_count: int) -> str:
    """
    Default region classifier for Rural/Suburban/Urban split.
    """
    if taxi_count < 30:
        return "Rural"
    elif taxi_count < 200:
        return "Suburban"
    else:
        return "Urban"


class CellDumper:
    """Accumulates per-cell records and writes cells.json on .dump()."""

    def __init__(self, default_gammas: Optional[dict] = None):
        self.cells: list[dict] = []
        self.default_gammas = default_gammas or DEFAULT_GAMMAS

    def add_cell(
        self,
        cell_id: Union[str, int],
        region: str,
        epsilon_remain: float,
        utilities: dict[str, float],
        gammas: Optional[dict[str, float]] = None,
    ):
       
        if region not in {"Rural", "Suburban", "Urban"}:
            raise ValueError(
                f"region must be Rural/Suburban/Urban, got {region!r}"
            )
        if epsilon_remain <= 0:
            raise ValueError(f"epsilon_remain must be > 0, got {epsilon_remain}")

        gammas_to_use = gammas or self.default_gammas
        
        all_fields = set(utilities) | set(gammas_to_use)
        missing_util = all_fields - set(utilities)
        missing_gamma = all_fields - set(gammas_to_use)
        if missing_util:
            raise KeyError(
                f"cell {cell_id}: fields with cost but no utility: {missing_util}"
            )
        if missing_gamma:
            raise KeyError(
                f"cell {cell_id}: fields with utility but no cost: {missing_gamma}"
            )

        # Sort by name for stable ordering
        field_list = [
            {
                "name": fname,
                "gamma": float(gammas_to_use[fname]),
                "utility": float(utilities[fname]),
            }
            for fname in sorted(all_fields)
        ]

        self.cells.append({
            "cell_id": str(cell_id),
            "region": region,
            "epsilon_remain": float(epsilon_remain),
            "fields": field_list,
        })

    def dump(self, path: Union[str, Path]):
        """Write the accumulated cells to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.cells, f, indent=2)
        # Quick sanity print so you know what got written
        regions = {}
        for c in self.cells:
            regions[c["region"]] = regions.get(c["region"], 0) + 1
        breakdown = ", ".join(f"{r}={n}" for r, n in sorted(regions.items()))
        print(f"Wrote {len(self.cells)} cells to {path}  ({breakdown})")


def dump_from_records(
    records: Iterable[dict],
    out_path: Union[str, Path],
    default_gammas: Optional[dict] = None,
):
    """
    One-shot conversion from a list of dicts. Each record must have:

        {
          "cell_id":         <str|int>,
          "region":          "Rural"|"Suburban"|"Urban",
          "epsilon_remain":  <float>,
          "utilities":       {field_name: U_i, ...},
          "gammas":          {field_name: Gamma_i, ...}     # optional
        }
    """
    dumper = CellDumper(default_gammas=default_gammas)
    for r in records:
        dumper.add_cell(
            cell_id=r["cell_id"],
            region=r["region"],
            epsilon_remain=r["epsilon_remain"],
            utilities=r["utilities"],
            gammas=r.get("gammas"),
        )
    dumper.dump(out_path)


# ─── Self-test: run `python cell_dumper.py` to verify on synthetic data ─────
if __name__ == "__main__":
    dumper = CellDumper()
    dumper.add_cell(
        cell_id="r3c7", region="Urban", epsilon_remain=1.0,
        utilities={
            "tau_agg_comp":    300.0,
            "tau_sens":        3120.0,
            "tau_noise":       4080.5,
            "tau_count":       1450.0,
            "tau_group_count": 980.0,
            "tau_row_contrib": 2310.0,
        },
    )
    dumper.add_cell(
        cell_id="r1c2", region="Rural", epsilon_remain=1.0,
        utilities={
            "tau_agg_comp":    1.5,
            "tau_sens":        4.2,
            "tau_noise":       5.1,
            "tau_count":       2.3,
            "tau_group_count": 1.8,
            "tau_row_contrib": 3.5,
        },
    )
    dumper.dump("data/cells_test.json")
    print("Self-test passed. Inspect data/cells_test.json")