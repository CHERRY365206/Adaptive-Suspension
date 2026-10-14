"""Downloader and preprocessor for real measured road profiles from LiRA-CD.

Dataset: Live Road Assessment Custom Dataset (LiRA-CD)
Source: Technical University of Denmark (DTU Data) & Danish Road Directorate (DRD)
Reference: Skar et al., Data in Brief (2023), PMC10375556, DOI: 10.11583/DTU.c.6659909
Instrument: P79 Profilometer (25-point laser road surface profilometer)
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import requests
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "measured_roads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LiRA-CD Figshare file IDs
DATASETS_CONFIG = {
    "lira_highway_m13": {
        "file_id": 41206187,
        "name": "Highway M13 (P79 Profilometer)",
        "route_type": "highway",
        "description": "Danish Motorway M13 surveyed by DRD P79 Profilometer in right wheelpath.",
        "byte_range": "0-1500000",  # ~1.5 MB covers ~6,000 points (~600 m of road)
    },
    "lira_urban_cph1": {
        "file_id": 40873694,
        "name": "Urban CPH1 Ring Road (P79 Profilometer)",
        "route_type": "urban",
        "description": "Copenhagen Urban Ring Road O2 (CPH1) surveyed by DRD P79 Profilometer.",
        "byte_range": "0-1500000",
    }
}


def download_and_process_dataset(key: str, cfg: dict) -> Path:
    url = f"https://ndownloader.figshare.com/files/{cfg['file_id']}"
    print(f"Fetching {cfg['name']} from {url} (Range: {cfg['byte_range']})...")
    
    headers = {"Range": f"bytes={cfg['byte_range']}"}
    response = requests.get(url, headers=headers, allow_redirects=True, timeout=30)
    response.raise_for_status()
    
    content = response.content.decode("latin1", errors="replace")
    lines = content.splitlines()
    
    # Header is first line
    header_line = lines[0]
    headers_list = [h.strip() for h in header_line.split(";")]
    
    # Parse rows until last (possibly truncated) line
    rows = []
    for line in lines[1:-1]:
        parts = [p.strip() for p in line.split(";")]
        if len(parts) >= 22:
            try:
                dist = float(parts[0])
                laser5 = float(parts[5])   # Left wheelpath [mm]
                laser21 = float(parts[21]) # Right wheelpath [mm]
                rows.append((dist, laser5, laser21))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No valid rows parsed for {key}")

    df_raw = pd.DataFrame(rows, columns=["raw_distance_m", "laser5_mm", "laser21_mm"])
    
    # Sort and enforce strict monotonicity
    df_raw = df_raw.drop_duplicates(subset=["raw_distance_m"]).sort_values("raw_distance_m")
    
    # Relative distance starting at 0
    d0 = df_raw["raw_distance_m"].iloc[0]
    df_raw["distance_m"] = df_raw["raw_distance_m"] - d0
    
    # Convert elevation from mm to SI meters [m]
    # Detrend / center elevation around 0 mean to isolate roughness from macro-slope
    elev_left_m = df_raw["laser5_mm"].to_numpy() / 1000.0
    elev_right_m = df_raw["laser21_mm"].to_numpy() / 1000.0
    
    # Remove linear trend / mean to isolate roughness irregularities
    poly_left = np.polyfit(df_raw["distance_m"], elev_left_m, deg=1)
    elev_left_detrend = elev_left_m - np.polyval(poly_left, df_raw["distance_m"])
    
    poly_right = np.polyfit(df_raw["distance_m"], elev_right_m, deg=1)
    elev_right_detrend = elev_right_m - np.polyval(poly_right, df_raw["distance_m"])
    
    # Average wheelpath elevation
    elev_avg = 0.5 * (elev_left_detrend + elev_right_detrend)
    
    df_clean = pd.DataFrame({
        "distance_m": df_raw["distance_m"].round(4),
        "elevation_m": elev_avg.round(6),
        "elevation_left_m": elev_left_detrend.round(6),
        "elevation_right_m": elev_right_detrend.round(6),
        "raw_distance_m": df_raw["raw_distance_m"].round(4)
    })
    
    out_csv = OUTPUT_DIR / f"{key}.csv"
    df_clean.to_csv(out_csv, index=False)
    print(f"  -> Saved {len(df_clean)} points ({df_clean['distance_m'].iloc[-1]:.1f} m) to {out_csv}")
    return out_csv


def main():
    metadata = {
        "dataset_name": "Live Road Assessment Custom Dataset (LiRA-CD) - Reference P79 Profilometer",
        "citation": "Skar et al., 'LiRA-CD: An open-source dataset for road condition modelling and research', Data in Brief, 2023.",
        "doi": "10.11583/DTU.c.6659909",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "survey_instrument": "Danish Road Directorate (DRD) P79 Profilometer van (25-point longitudinal/transverse laser beam)",
        "measurement_type": "Physical laser-measured longitudinal road elevation",
        "raw_units": {"distance": "meters", "elevation": "millimeters"},
        "processed_units": {"distance": "meters", "elevation": "meters"},
        "traces": {}
    }
    
    for key, cfg in DATASETS_CONFIG.items():
        try:
            out_file = download_and_process_dataset(key, cfg)
            df = pd.read_csv(out_file)
            ds = np.diff(df["distance_m"])
            metadata["traces"][key] = {
                "name": cfg["name"],
                "file": out_file.name,
                "route_type": cfg["route_type"],
                "total_points": len(df),
                "total_length_m": float(df["distance_m"].iloc[-1]),
                "mean_spatial_step_m": float(np.mean(ds)),
                "elevation_std_mm": float(np.std(df["elevation_m"]) * 1000.0),
                "elevation_min_mm": float(np.min(df["elevation_m"]) * 1000.0),
                "elevation_max_mm": float(np.max(df["elevation_m"]) * 1000.0),
                "description": cfg["description"]
            }
        except Exception as e:
            print(f"Error downloading {key}: {e}")

    meta_file = OUTPUT_DIR / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_file}")


if __name__ == "__main__":
    main()
