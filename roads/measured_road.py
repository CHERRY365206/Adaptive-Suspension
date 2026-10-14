"""Measured road profile loader and interpolator for real experimental data.

Supports high-resolution road profilometer data, such as the LiRA-CD dataset
(Danish Road Directorate P79 Profilometer laser survey).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from roads.base import BaseRoadProfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "measured_roads"


class MeasuredRoadProfile(BaseRoadProfile):
    """Road profile generator backed by physical measured road elevation data.

    Enforces strict SI units (meters for distance and elevation), monotonic distances,
    data validation, and accurate continuous interpolation.
    """

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        distance_m: Optional[np.ndarray] = None,
        elevation_m: Optional[np.ndarray] = None,
        name: str = "MeasuredRoad",
        route_type: str = "measured",
        interpolation_kind: str = "linear",
        boundary_mode: str = "clamp",
        description: str = "Real measured road surface profile",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(seed=None, name=name)
        self.route_type = route_type
        self.interpolation_kind = interpolation_kind
        self.boundary_mode = boundary_mode
        self.description = description
        self.metadata = metadata or {}
        self.file_path = Path(file_path) if file_path else None

        if self.file_path is not None and self.file_path.exists():
            self._load_from_csv(self.file_path)
        elif distance_m is not None and elevation_m is not None:
            self._init_from_arrays(distance_m, elevation_m)
        else:
            raise ValueError("Must provide either an existing file_path or both distance_m and elevation_m arrays.")

    def _load_from_csv(self, path: Path) -> None:
        """Load, validate, and preprocess CSV containing measured road data."""
        df = pd.read_csv(path)
        
        # Check required columns
        dist_col = None
        elev_col = None
        for c in ["distance_m", "Distance [m]", "distance", "Distance", "s"]:
            if c in df.columns:
                dist_col = c
                break
        for c in ["elevation_m", "elevation", "zr", "Elevation", "laser5_mm", "Laser 5 [mm]"]:
            if c in df.columns:
                elev_col = c
                break

        if dist_col is None or elev_col is None:
            raise ValueError(f"CSV {path} must contain distance and elevation columns. Found: {list(df.columns)}")

        s_arr = df[dist_col].to_numpy(dtype=np.float64)
        z_arr = df[elev_col].to_numpy(dtype=np.float64)

        # Unit conversion if elevation in mm
        if "mm" in elev_col.lower():
            z_arr = z_arr / 1000.0

        self._init_from_arrays(s_arr, z_arr)

    def _init_from_arrays(self, s: np.ndarray, z: np.ndarray) -> None:
        """Validate arrays and build interpolator."""
        s = np.asarray(s, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        if len(s) < 2 or len(z) < 2:
            raise ValueError("Measured road profile must contain at least 2 points.")
        if len(s) != len(z):
            raise ValueError(f"Dimension mismatch between distance ({len(s)}) and elevation ({len(z)}).")

        # Check for NaN / Inf
        valid_mask = np.isfinite(s) & np.isfinite(z)
        if not np.all(valid_mask):
            s = s[valid_mask]
            z = z[valid_mask]

        # Shift distance to start at 0
        if s[0] != 0.0:
            s = s - s[0]

        # Monotonicity check and duplicate removal
        diffs = np.diff(s)
        if np.any(diffs <= 0):
            # Sort and remove duplicates
            order = np.argsort(s)
            s = s[order]
            z = z[order]
            _, unique_idx = np.unique(s, return_index=True)
            s = s[unique_idx]
            z = z[unique_idx]

        self.s_data = s
        self.z_data = z
        self.s_min = float(s[0])
        self.s_max = float(s[-1])
        self.total_length_m = self.s_max - self.s_min
        self.mean_step_m = float(np.mean(np.diff(s)))
        self.elevation_std_m = float(np.std(z))
        self.elevation_min_m = float(np.min(z))
        self.elevation_max_m = float(np.max(z))

        # Build interpolator
        self._interp = interp1d(
            self.s_data,
            self.z_data,
            kind=self.interpolation_kind,
            bounds_error=False,
            fill_value=(self.z_data[0], self.z_data[-1])
        )

    def elevation_spatial(self, s: float | np.ndarray) -> float | np.ndarray:
        """Return road surface vertical displacement zr [m] at distance s [m]."""
        s_arr = np.asarray(s, dtype=np.float64)
        
        if self.boundary_mode == "loop" and self.total_length_m > 0:
            s_eval = np.mod(s_arr, self.total_length_m)
        elif self.boundary_mode == "clamp":
            s_eval = np.clip(s_arr, self.s_min, self.s_max)
        else:
            s_eval = s_arr

        z_res = self._interp(s_eval)
        if np.isscalar(s):
            return float(z_res.item())
        return z_res


def list_available_measured_roads(data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List all available measured road profiles in the data directory."""
    d_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    meta_path = d_dir / "metadata.json"
    available = []
    
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for key, tr in meta.get("traces", {}).items():
                csv_file = d_dir / tr["file"]
                if csv_file.exists():
                    available.append({
                        "key": key,
                        "name": tr["name"],
                        "file_path": str(csv_file),
                        "route_type": tr["route_type"],
                        "total_length_m": tr["total_length_m"],
                        "description": tr["description"]
                    })
        except Exception:
            pass

    # Fallback to inspecting directory CSVs
    if not available and d_dir.exists():
        for csv_file in d_dir.glob("*.csv"):
            available.append({
                "key": csv_file.stem,
                "name": csv_file.stem.replace("_", " ").title(),
                "file_path": str(csv_file),
                "route_type": "measured",
                "total_length_m": 0.0,
                "description": f"Measured road trace {csv_file.name}"
            })
            
    return available


def load_measured_road_profile(
    trace_key_or_path: Union[str, Path],
    data_dir: Optional[Path] = None
) -> MeasuredRoadProfile:
    """Convenience factory to load a measured road profile by key or file path."""
    d_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    path = Path(trace_key_or_path)
    
    if not path.exists():
        # Check if in d_dir
        candidate = d_dir / f"{trace_key_or_path}.csv"
        if candidate.exists():
            path = candidate
        else:
            candidate2 = d_dir / trace_key_or_path
            if candidate2.exists():
                path = candidate2

    if not path.exists():
        raise FileNotFoundError(f"Measured road dataset not found at {trace_key_or_path} (searched in {d_dir})")

    # Load metadata if present
    meta_file = d_dir / "metadata.json"
    trace_meta = {}
    name = path.stem.replace("_", " ").title()
    desc = "Physical laser profilometer road measurement"
    route_type = "measured"

    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                full_meta = json.load(f)
            key = path.stem
            if key in full_meta.get("traces", {}):
                trace_meta = full_meta["traces"][key]
                name = trace_meta.get("name", name)
                desc = trace_meta.get("description", desc)
                route_type = trace_meta.get("route_type", route_type)
        except Exception:
            pass

    return MeasuredRoadProfile(
        file_path=path,
        name=name,
        route_type=route_type,
        boundary_mode="loop",
        description=desc,
        metadata=trace_meta
    )
