# Real Measured Road Profile Dataset Documentation

## 1. Overview & Research Motivation

A critical deficiency in synthetic suspension simulation benchmarks (such as simplified sinusoid speed bumps or idealized ISO 8608 white noise approximations) is that they assume homogeneous, stationary stochastic processes with idealized frequency distributions. Real road surfaces exhibit:
- Localized micro-irregularities (joints, aggregate texture).
- Discrete transient events (asphalt patching, utility covers, manholes, frost heaves).
- Non-stationary spatial power spectral density (PSD) with distinct spatial wavebands.

To satisfy rigorous academic standards, this project replaces synthetic road profiles as the primary experimental evidence with physical profilometer survey data from genuine road networks.

---

## 2. Evaluation of Candidate Road Datasets

| Dataset | Source / Reference | Nature of Data | Verification Status | Suitability Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset A: LiRA-CD** | Skar et al., *Data in Brief* 48 (2023) 109260; PMC10375556; Figshare collection `6659909` | **Physical High-Resolution Laser Profilometer Survey** (Danish Road Directorate P79 Profilometer) | **Verified & Acquired** (CC BY 4.0) | **SELECTED PRIMARY BENCHMARK**. Genuine physical laser triangulations along left wheelpath. |
| **Dataset B: Zenodo 4386256** | Zenodo Record `4386256` | Candidate profilometer road database | **Unavailable** (HTTP 404 / Missing record) | **Rejected**. Record does not exist in Zenodo public registry. |
| **Dataset C: Zenodo 17242919** | Foo & Tan (2024), Zenodo Record `17242919` | MATLAB scripts for synthetic ISO 8608 profiles | **Verified as Synthetic** | **Rejected as primary evidence**. Contains synthetic frequency-domain generators, not physical profilometer surveys. |

---

## 3. Dataset A (LiRA-CD) Detailed Specification

### 3.1 Provenance and Measurement Equipment
- **Provider**: Danish Road Directorate (*Vejdirektoratet*) in collaboration with the Technical University of Denmark (DTU).
- **Vehicle**: Specialized P79 Road Surface Profilometer Survey Vehicle.
- **Sensor**: High-precision Laser Triangulation Sensor (Laser 5, aligned exactly with the vehicle left wheelpath).
- **Sampling Frequency & Resolution**: Spatial step of $\Delta s \approx 0.1007\text{ m}$ (approx. 10 samples per meter).
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

### 3.2 Extracted Representative Traces

Two contrasting physical traces were extracted and stored in `data/measured_roads/`:

#### 1. Trace `lira_highway_m13.csv`
- **Location**: Motorway M13, Zealand, Denmark.
- **Road Category**: High-speed motorway / rural expressway.
- **Total Points**: 3,765 points.
- **Physical Section Length**: $379.1\text{ m}$.
- **Mean Spatial Interval**: $0.1007\text{ m}$.
- **Elevation Statistics**:
  - Standard Deviation $\sigma$: $6.78\text{ mm}$ ($0.00678\text{ m}$)
  - Peak-to-Peak Amplitude: $44.5\text{ mm}$ ($-20.8\text{ mm}$ to $+23.7\text{ mm}$)
  - Profile Characteristics: High-speed smooth asphalt with low long-wavelength waviness; typical high-comfort cruising excitation.

#### 2. Trace `lira_urban_cph1.csv`
- **Location**: Greater Copenhagen Urban Arterial, Denmark.
- **Road Category**: Urban commuter route with service joints, localized asphalt settlements, and utility cuts.
- **Total Points**: 3,750 points.
- **Physical Section Length**: $377.6\text{ m}$.
- **Mean Spatial Interval**: $0.1007\text{ m}$.
- **Elevation Statistics**:
  - Standard Deviation $\sigma$: $14.64\text{ mm}$ ($0.01464\text{ m}$)
  - Peak-to-Peak Amplitude: $131.4\text{ mm}$ ($-42.8\text{ mm}$ to $+88.6\text{ mm}$)
  - Profile Characteristics: High-roughness urban excitation with prominent localized transient bumps and severe tyre load variations.

---

## 4. Ingestion, Preprocessing & Quality Assurance Pipeline

The raw profilometer laser elevations undergo rigorous scientific preprocessing implemented in `scripts/download_measured_roads.py` and `roads/measured_road.py`:

```
Raw P79 Laser Survey Data (mm)
              │
              ▼
   Monotonic Distance Check ───► Remove duplicate s coordinates & sort
              │
              ▼
      Zero Baseline Offset  ───► Shift s[0] to 0.0 m
              │
              ▼
        Linear Detrending    ───► Remove macro-grade slope via linear regression
              │
              ▼
       Strict SI Conversion ───► Convert mm to SI meters (z_r * 1e-3)
              │
              ▼
 Continuous Interpolation   ───► scipy.interpolate.interp1d(kind='linear')
              │
              ▼
  Velocity-Scaled Time Map  ───► s(t) = v * t (exact temporal excitation)
```

### 4.1 Linear Detrending
Roads naturally possess long-wavelength macroscopic grading (hills, road banking, and highway inclines). Because vehicle suspension dynamics respond to relative surface roughness rather than static altitude changes, the macroscopic slope is removed via least-squares linear detrending:

$$z_{roughness}(s) = z_{raw}(s) - (m \cdot s + c)$$

where $m$ is the mean gradient and $c$ is the elevation offset. This guarantees zero mean elevation and eliminates artificial drift in quarter-car simulation state integrations.

### 4.2 Velocity-to-Time Transformation
For a vehicle travelling at constant forward velocity $v$ [m/s] ($v = \frac{v_{km/h}}{3.6}$), the road spatial coordinate $s$ relates directly to simulation time $t$ by:

$$s(t) = v \cdot t$$

The continuous elevation callable $z_r(t)$ evaluates the spatial interpolator:

$$z_r(t) = \text{interp1d}(v \cdot t)$$

If the simulation duration exceeds the trace length, the class supports both `boundary_mode="clamp"` (holds terminal elevation) and `boundary_mode="loop"` (wraps distance periodically: $s_{eval} = s \pmod{L_{total}}$).

---

## 5. Storage and File Integrity

The dataset files are structured as follows:

```
data/
  measured_roads/
    metadata.json                 # Comprehensive metadata, source, DOI, license, trace stats
    lira_highway_m13.csv          # Detrended SI distance_m and elevation_m (Highway M13)
    lira_urban_cph1.csv           # Detrended SI distance_m and elevation_m (Urban CPH1)
```

### CSV Schema:
- `distance_m`: Longitudinal spatial distance along road path [m] (strictly monotonically increasing, starts at 0.0).
- `elevation_m`: Detrended vertical road profile displacement [m] (SI units).

Every data file is committed directly to the project repository, ensuring 100% offline reproducibility without external download requirements during testing or examination.
