"""Streamlit Guided Simulator for AI-Assisted Adaptive Active Suspension in Electric Vehicles."""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from models.vehicle_parameters import (
    VehicleParameters,
    VehiclePreset,
    VEHICLE_PRESETS,
    get_vehicle_preset,
    validate_custom_parameters,
    RESEARCH_BENCHMARK_METADATA,
)
from models.quarter_car import QuarterCarModel, SimulationResult
from models.actuator import ActuatorModel
from roads.road_dataset import create_scenario
from roads.measured_road import load_measured_road_profile
from controllers.lqr import LQRConfig, LQRController
from controllers.controller_registry import get_nominal_configs, build_nominal_controllers
from controllers.adaptive_selector import DEFAULT_ROAD_TO_CONTROLLER_MAP
from controllers.mpc import MPCConfig, MPCController
from ai.predict_road import RoadPredictor
from ai.feature_extraction import SensorFeatureExtractor
from ai.dataset_generation import LABEL_MAP, INV_LABEL_MAP
from ai.train_classifier import train_road_classifier
from optimization.optimize_lqr import run_pso_tuning
from experiments.baselines import SuspensionBenchmarkSuite, SYSTEM_DISPLAY_NAMES
from experiments.metrics import compute_simulation_metrics
from experiments.runner import run_comparative_benchmark
from experiments.sensitivity_analysis import run_parameter_sensitivity_study
from experiments.unilateral_contact_study import run_unilateral_contact_comparison
from experiments.real_road_study import run_real_road_study
from dashboard.visualizations import (
    plot_simulation_time_series,
    plot_comparison_bar_chart,
    plot_pso_convergence_interactive,
    plot_feature_importance_interactive,
    plot_road_profile,
    plot_body_acceleration_comparison,
    plot_suspension_travel_comparison,
    plot_tyre_load_comparison,
    plot_actuator_force,
    plot_three_way_comparison
)


# Set page configuration
st.set_page_config(
    page_title="Adaptive EV Suspension Simulator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom minimal styling
st.markdown("""
<style>
    .simulator-title { font-size: 2.1rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0.1rem; }
    .simulator-subtitle { font-size: 1.05rem; color: #4b5563; margin-bottom: 1.2rem; }
    .decision-box { background-color: #f1f5f9; border-radius: 8px; padding: 1rem; border-left: 5px solid #2563eb; margin: 1rem 0; }
    .stButton>button { border-radius: 6px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# ROAD PROFILE OPTIONS
# ==============================================================================

ROAD_OPTIONS = {
    "Speed Bump (50mm Versine)": "bumpy",
    "Real Measured: LiRA Highway M13": "measured_highway",
    "Real Measured: LiRA Urban CPH1": "measured_urban",
    "Smooth Highway (ISO Class A)": "smooth",
    "Rough Road (ISO Class C/D)": "rough"
}


# ==============================================================================
# HEADER & SCENARIO SELECTION
# ==============================================================================
st.markdown('<div class="simulator-title">🚗 ADAPTIVE EV SUSPENSION SIMULATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="simulator-subtitle">Simulate how an intelligent suspension responds to different driving conditions.</div>', unsafe_allow_html=True)

st.subheader("DRIVING SCENARIO")

col_veh, col_road, col_ctrl, col_spd, col_dur = st.columns([1.1, 1.4, 1.3, 1.1, 0.9])

with col_veh:
    vehicle_choice = st.selectbox(
        "Vehicle",
        ["Sedan EV", "Compact EV", "Custom EV"],
        index=0,
        help="Select a passenger EV chassis preset or configure a custom vehicle."
    )

with col_road:
    road_choice = st.selectbox(
        "Road Condition",
        list(ROAD_OPTIONS.keys()),
        index=0,
        help="Road excitation surface (Real LiRA-CD laser traces, Speed Bump, or synthetic ISO)."
    )

with col_ctrl:
    ctrl_choice = st.selectbox(
        "Active Controller",
        [
            "Proposed AI + PSO Adaptive LQR",
            "Constrained Model Predictive Control (MPC)",
            "Fixed Nominal LQR"
        ],
        index=0,
        help="Select active controller to evaluate against passive reference baseline."
    )

with col_spd:
    speed_kmh = st.slider(
        "Vehicle Speed [km/h]",
        min_value=20.0,
        max_value=110.0,
        value=50.0,
        step=5.0
    )

with col_dur:
    duration_s = st.slider(
        "Simulation Duration [s]",
        min_value=2.0,
        max_value=10.0,
        value=6.0,
        step=1.0
    )


# Vehicle parameter assignment and physical validation
custom_valid = True
custom_errors: list[str] = []

if vehicle_choice in VEHICLE_PRESETS:
    preset = get_vehicle_preset(vehicle_choice)
    ms_val = preset.parameters.ms
    mu_val = preset.parameters.mu
    ks_val = preset.parameters.ks
    cs_val = preset.parameters.cs
    kt_val = preset.parameters.kt
    max_travel_val = preset.parameters.max_travel
    fmax_val = preset.f_max
    slew_val = preset.slew_rate

    if vehicle_choice == "Sedan EV":
        st.info(
            f"📄 **{preset.display_name}**\n\n"
            f"• **Source Citation:** {preset.source_citation}\n\n"
            f"• **Benchmark Provenance:** {preset.status_message}\n\n"
            f"• **Tyre Modeling:** {RESEARCH_BENCHMARK_METADATA['tyre_damping_note']}"
        )
    else:  # Compact EV fallback
        st.warning(
            f"⚠️ **{preset.display_name}**\n\n"
            f"{preset.status_message}\n\n"
            f"• **Benchmark Reference:** Magaji et al. (2025), DOI: 10.3390/math13152418\n\n"
            f"• **Tyre Modeling:** {RESEARCH_BENCHMARK_METADATA['tyre_damping_note']}"
        )

    with st.expander("🔍 View Active Research Benchmark Parameters & Units", expanded=False):
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            st.markdown(f"**Sprung Mass ($m_s$):** `{ms_val:.1f} kg`")
            st.markdown(f"**Unsprung Mass ($m_u$):** `{mu_val:.1f} kg`")
        with p_c2:
            st.markdown(f"**Suspension Stiffness ($k_s$):** `{ks_val:,.0f} N/m`")
            st.markdown(f"**Suspension Damping ($c_s$):** `{cs_val:,.0f} N·s/m`")
        with p_c3:
            st.markdown(f"**Tyre Stiffness ($k_t$):** `{kt_val:,.0f} N/m`")
            st.markdown(f"**Max Travel ($z_{{\\max}}$):** `±{max_travel_val * 1000:.0f} mm`")
        with p_c4:
            st.markdown(f"**Max Actuator Force ($F_{{\\max}}$):** `±{fmax_val:,.0f} N`")
            st.markdown(f"**Actuator Slew Limit ($\dot{{F}}_{{\\max}}$):** `{slew_val:,.0f} N/s`")
else:
    # Custom EV parameters with rigorous physical validation
    with st.expander("⚙️ Configure Custom Vehicle & Actuator Parameters", expanded=True):
        st.caption("Enter custom quarter-car parameters. Physical constraints ($m_u < m_s$, strictly positive mass, stiffness, and damping) are strictly enforced without silently overwriting your inputs.")
        c1, c2, c3 = st.columns(3)
        with c1:
            ms_val = st.number_input("Sprung Mass ms [kg]", min_value=100.0, max_value=1500.0, value=395.3, step=5.0, format="%.1f")
            mu_val = st.number_input("Unsprung Mass mu [kg]", min_value=10.0, max_value=300.0, value=48.3, step=1.0, format="%.1f")
        with c2:
            ks_val = st.number_input("Suspension Stiffness ks [N/m]", min_value=5000.0, max_value=100000.0, value=30010.0, step=500.0, format="%.0f")
            cs_val = st.number_input("Suspension Damping cs [N·s/m]", min_value=100.0, max_value=8000.0, value=1450.0, step=50.0, format="%.0f")
        with c3:
            kt_val = st.number_input("Tyre Stiffness kt [N/m]", min_value=50000.0, max_value=600000.0, value=340000.0, step=5000.0, format="%.0f")
            fmax_val = st.number_input("Maximum Actuator Force Fmax [N]", min_value=200.0, max_value=5000.0, value=1800.0, step=100.0, format="%.0f")
            slew_val = st.number_input("Actuator Slew Rate [N/s]", min_value=1000.0, max_value=50000.0, value=18000.0, step=1000.0, format="%.0f")
        max_travel_val = 0.08

        custom_valid, custom_errors = validate_custom_parameters(
            ms=ms_val,
            mu=mu_val,
            ks=ks_val,
            cs=cs_val,
            kt=kt_val,
            max_travel=max_travel_val,
            f_max=fmax_val,
            slew_rate=slew_val
        )

        if not custom_valid:
            for err in custom_errors:
                st.error(f"❌ {err}")
            st.warning("Please correct the invalid parameter values above before launching simulation.")
        else:
            tot_m = ms_val + mu_val
            f_bnc = np.sqrt(ks_val / ms_val) / (2.0 * np.pi)
            f_hop = np.sqrt((ks_val + kt_val) / mu_val) / (2.0 * np.pi)
            st.success(f"✓ Valid physical parameters: Total Mass = {tot_m:.1f} kg | Bounce Natural Freq = {f_bnc:.2f} Hz | Wheel-Hop Freq = {f_hop:.2f} Hz")


# ==============================================================================
# PRIMARY SIMULATION TRIGGER
# ==============================================================================
run_simulation_btn = st.button(
    "▶ RUN SMART SUSPENSION SIMULATION",
    type="primary",
    use_container_width=True
)

if "last_simulation" not in st.session_state:
    st.session_state["last_simulation"] = None


# ==============================================================================
# SIMULATION WORKFLOW EXECUTION
# ==============================================================================
if run_simulation_btn and not custom_valid:
    st.error("❌ Cannot launch simulation: One or more vehicle parameters violate physical constraints. Please review and resolve the errors above.")
elif run_simulation_btn:
    progress_box = st.status("Running Smart Suspension Simulation Pipeline...", expanded=True)

    try:
        # STEP 1: Vehicle Model
        progress_box.write("✓ Creating vehicle quarter-car physical model...")
        v_params = VehicleParameters(
            ms=ms_val,
            mu=mu_val,
            ks=ks_val,
            cs=cs_val,
            kt=kt_val,
            max_travel=max_travel_val,
            g=9.81
        )

        # STEP 2: Actuator Model
        progress_box.write(f"✓ Initializing actuator model (Fmax = ±{fmax_val:.0f} N, Slew = {slew_val:.0f} N/s)...")
        actuator = ActuatorModel(
            max_force=fmax_val,
            min_force=-fmax_val,
            slew_rate_limit=slew_val
        )

        # STEP 3: Generate Road Profile ONCE (Mandatory exact parity)
        road_type_key = ROAD_OPTIONS[road_choice]
        if road_type_key == "measured_highway":
            progress_box.write(f"✓ Loading real measured profilometer data: LiRA Highway M13 at {speed_kmh:.0f} km/h...")
            road_obj = load_measured_road_profile("lira_highway_m13")
            road_fn = road_obj.get_time_callable(speed_kmh / 3.6)
            gt_class = "smooth"
        elif road_type_key == "measured_urban":
            progress_box.write(f"✓ Loading real measured profilometer data: LiRA Urban CPH1 at {speed_kmh:.0f} km/h...")
            road_obj = load_measured_road_profile("lira_urban_cph1")
            road_fn = road_obj.get_time_callable(speed_kmh / 3.6)
            gt_class = "rough"
        else:
            progress_box.write(f"✓ Generating {road_choice} profile at {speed_kmh:.0f} km/h (Seed: 42)...")
            sc = create_scenario(
                scenario_id=101,
                road_type=road_type_key,
                speed_kmh=speed_kmh,
                split="test",
                duration_s=duration_s,
                seed=42
            )
            road_fn = sc.road_profile.get_time_callable(sc.speed_mps)
            gt_class = road_type_key

        # Build benchmark suite
        suite = SuspensionBenchmarkSuite(
            vehicle_params=v_params,
            actuator=actuator,
            classifier_path=PROJECT_ROOT / "artifacts/models/road_classifier.joblib",
            pso_configs_path=PROJECT_ROOT / "artifacts/controllers/pso_lqr_configs.json"
        )

        # STEP 4: Passive Suspension Simulation (u = 0)
        progress_box.write("✓ Simulating Passive Suspension response (u = 0)...")
        res_passive = suite.simulate_system(
            system_type="passive",
            road_func=road_fn,
            ground_truth_class=gt_class,
            duration_s=duration_s,
            dt=0.002
        )

        # STEP 5 & 6: Simulated Sensor Features & AI Classification
        progress_box.write("✓ Extracting rolling-window sensor features from chassis accelerometers...")
        classifier_path = PROJECT_ROOT / "artifacts/models/road_classifier.joblib"
        if not classifier_path.exists():
            raise FileNotFoundError("Road classifier model artifact not found. Please train the classifier model first.")

        artifact = joblib.load(classifier_path)
        clf_pipeline = artifact["pipeline"]
        inv_map = artifact.get("inv_label_map", INV_LABEL_MAP)
        extractor = SensorFeatureExtractor(sampling_rate_hz=500.0)

        # Window analysis on sensor response
        v_rel_pass = res_passive.zs_dot - res_passive.zu_dot
        n_pts = len(res_passive.t)
        w_len = min(n_pts, int(0.5 / 0.002))
        peak_idx = int(np.argmax(np.abs(res_passive.zs_ddot)))
        st_idx = max(0, min(peak_idx - w_len // 2, n_pts - w_len))
        end_idx = st_idx + w_len

        feats = extractor.extract_features(
            res_passive.zs_ddot[st_idx:end_idx],
            res_passive.zu_ddot[st_idx:end_idx],
            res_passive.suspension_travel[st_idx:end_idx],
            v_rel_pass[st_idx:end_idx]
        )
        probas = clf_pipeline.predict_proba(feats.reshape(1, -1))[0]
        pred_idx = int(np.argmax(probas))
        ai_conf = float(probas[pred_idx])
        ai_detected_road = inv_map.get(pred_idx, "bumpy")

        # STEP 7: Controller Selection & Setup
        controller_tag = DEFAULT_ROAD_TO_CONTROLLER_MAP.get(ai_detected_road, "balanced")
        controller_display = controller_tag.replace("_", " ").title() + " LQR"
        
        if "MPC" in ctrl_choice:
            active_sys_type = "mpc"
            active_display_name = "Constrained MPC"
            progress_box.write(f"✓ Target Controller: Constrained MPC (Receding Horizon Np=12, QP Optimization)...")
        elif "Adaptive" in ctrl_choice:
            active_sys_type = "proposed_ai_pso"
            active_display_name = f"AI Adaptive LQR ({controller_display})"
            progress_box.write(f"✓ AI detected road: '{ai_detected_road.capitalize()}' ({ai_conf*100.0:.1f}% confidence) → Selecting {controller_display}...")
        else:
            active_sys_type = "fixed_nominal_lqr"
            active_display_name = "Fixed Nominal LQR"
            progress_box.write(f"✓ Target Controller: Fixed Nominal LQR (Balanced baseline)...")

        # STEP 8: Active Suspension Simulation on EXACT SAME road realization
        progress_box.write(f"✓ Simulating {active_display_name} with actuator constraints...")
        res_adaptive = suite.simulate_system(
            system_type=active_sys_type,
            road_func=road_fn,
            ground_truth_class=gt_class,
            duration_s=duration_s,
            dt=0.002
        )

        # STEP 9: Compute Performance Metrics
        progress_box.write("✓ Computing standardized engineering performance metrics...")
        metrics_passive = compute_simulation_metrics(res_passive, ground_truth_class=gt_class)
        metrics_adaptive = compute_simulation_metrics(res_adaptive, ground_truth_class=gt_class)


        progress_box.update(label="✓ Simulation Completed Successfully!", state="complete", expanded=False)

        # Save to session state
        st.session_state["last_simulation"] = {
            "vehicle_name": vehicle_choice,
            "road_choice": road_choice,
            "speed_kmh": speed_kmh,
            "duration_s": duration_s,
            "v_params": v_params,
            "actuator": actuator,
            "ai_detected_road": ai_detected_road,
            "ai_conf": ai_conf,
            "controller_display": controller_display,
            "active_display_name": active_display_name,
            "res_passive": res_passive,
            "res_adaptive": res_adaptive,
            "metrics_passive": metrics_passive,
            "metrics_adaptive": metrics_adaptive
        }

    except Exception as e:
        progress_box.update(label="Simulation Error", state="error", expanded=True)
        st.error(f"Simulation could not be completed. Please check the selected parameters. Error: {e}")


# ==============================================================================
# DISPLAY RESULTS
# ==============================================================================
sim = st.session_state.get("last_simulation")

if sim is not None:
    res_p: SimulationResult = sim["res_passive"]
    res_a: SimulationResult = sim["res_adaptive"]
    m_p = sim["metrics_passive"]
    m_a = sim["metrics_adaptive"]
    v_par: VehicleParameters = sim["v_params"]
    act: ActuatorModel = sim["actuator"]

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 1. AI DECISION RESULT
    # --------------------------------------------------------------------------
    st.subheader("🧠 CONTROLLER DECISION & CLASSIFICATION")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Test Scenario", sim["road_choice"])
    with d2:
        st.metric("AI Detected Road", sim["ai_detected_road"].capitalize())
    with d3:
        st.metric("Detection Confidence", f"{sim['ai_conf'] * 100.0:.1f}%")
    with d4:
        st.metric("Active Controller", sim.get("active_display_name", sim["controller_display"]))


    st.markdown("---")

    # --------------------------------------------------------------------------
    # 2. MAIN SIMULATION RESULTS & GRAPHS
    # --------------------------------------------------------------------------
    st.subheader("📈 SIMULATION RESULTS")

    # Graph 1: Road Profile
    fig_road = plot_road_profile(
        res_p.t,
        res_p.zr,
        title=f"1. Road Surface Displacement Profile: {sim['road_choice']} ({sim['speed_kmh']:.0f} km/h)"
    )
    st.plotly_chart(fig_road, use_container_width=True)

    # Graph 2: Body Acceleration — PRIMARY RESULT GRAPH
    fig_acc = plot_body_acceleration_comparison(
        res_p,
        res_a,
        title="2. PRIMARY RESULT: Chassis Vertical Body Acceleration (Passive vs AI Adaptive)"
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # Graphs 3 & 4: Suspension Travel & Dynamic Tyre Load side by side
    col_g3, col_g4 = st.columns(2)
    with col_g3:
        fig_trv = plot_suspension_travel_comparison(
            res_p,
            res_a,
            max_travel=v_par.max_travel
        )
        st.plotly_chart(fig_trv, use_container_width=True)

    with col_g4:
        fig_tyre = plot_tyre_load_comparison(
            res_p,
            res_a,
            static_load=v_par.static_tyre_load
        )
        st.plotly_chart(fig_tyre, use_container_width=True)

    # Graph 5: Actuator Force (Dynamic limits)
    fig_act = plot_actuator_force(res_a, max_force=act.max_force)
    st.plotly_chart(fig_act, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3. PERFORMANCE COMPARISON TABLE
    # --------------------------------------------------------------------------
    st.subheader("📊 PERFORMANCE COMPARISON")

    accel_chg = ((m_a["rms_body_accel_mps2"] - m_p["rms_body_accel_mps2"]) / max(1e-6, m_p["rms_body_accel_mps2"])) * 100.0
    peak_chg = ((m_a["peak_body_accel_mps2"] - m_p["peak_body_accel_mps2"]) / max(1e-6, m_p["peak_body_accel_mps2"])) * 100.0
    trv_chg = ((m_a["rms_susp_travel_m"] - m_p["rms_susp_travel_m"]) / max(1e-6, m_p["rms_susp_travel_m"])) * 100.0
    tyre_chg = ((m_a["rms_dynamic_tyre_load_n"] - m_p["rms_dynamic_tyre_load_n"]) / max(1e-6, m_p["rms_dynamic_tyre_load_n"])) * 100.0
    load_diff = m_a["min_total_tyre_load_n"] - m_p["min_total_tyre_load_n"]

    comparison_df = pd.DataFrame([
        {
            "Performance Metric": "RMS Body Acceleration [m/s²]",
            "Passive Suspension": f"{m_p['rms_body_accel_mps2']:.4f}",
            "AI Adaptive Suspension": f"{m_a['rms_body_accel_mps2']:.4f}",
            "Relative Difference": f"{accel_chg:+.1f}%"
        },
        {
            "Performance Metric": "Peak Body Acceleration [m/s²]",
            "Passive Suspension": f"{m_p['peak_body_accel_mps2']:.4f}",
            "AI Adaptive Suspension": f"{m_a['peak_body_accel_mps2']:.4f}",
            "Relative Difference": f"{peak_chg:+.1f}%"
        },
        {
            "Performance Metric": "RMS Suspension Travel [mm]",
            "Passive Suspension": f"{m_p['rms_susp_travel_m'] * 1000.0:.2f}",
            "AI Adaptive Suspension": f"{m_a['rms_susp_travel_m'] * 1000.0:.2f}",
            "Relative Difference": f"{trv_chg:+.1f}%"
        },
        {
            "Performance Metric": "Dynamic Tyre Load Variation (RMS) [N]",
            "Passive Suspension": f"{m_p['rms_dynamic_tyre_load_n']:.1f}",
            "AI Adaptive Suspension": f"{m_a['rms_dynamic_tyre_load_n']:.1f}",
            "Relative Difference": f"{tyre_chg:+.1f}%"
        },
        {
            "Performance Metric": "Minimum Normal Tyre Load [N]",
            "Passive Suspension": f"{m_p['min_total_tyre_load_n']:.1f}",
            "AI Adaptive Suspension": f"{m_a['min_total_tyre_load_n']:.1f}",
            "Relative Difference": f"{load_diff:+.1f} N"
        },
        {
            "Performance Metric": "Actuator Force Saturation [%]",
            "Passive Suspension": "0.0%",
            "AI Adaptive Suspension": f"{m_a['saturation_pct']:.1f}%",
            "Relative Difference": "—"
        },
        {
            "Performance Metric": "Electrical Energy Drawn [J]",
            "Passive Suspension": "0.0",
            "AI Adaptive Suspension": f"{m_a['electrical_energy_j']:.1f}",
            "Relative Difference": "—"
        }
    ])

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # 4. FACTUAL RESULT SUMMARY
    # --------------------------------------------------------------------------
    st.subheader("RESULT SUMMARY")

    st.markdown(f"""
    - **Body Acceleration**: Passive `{m_p['rms_body_accel_mps2']:.3f} m/s²` → Adaptive `{m_a['rms_body_accel_mps2']:.3f} m/s²` ({accel_chg:+.1f}%)
    - **Suspension Travel**: Passive `{m_p['rms_susp_travel_m']*1000.0:.1f} mm` → Adaptive `{m_a['rms_susp_travel_m']*1000.0:.1f} mm` ({trv_chg:+.1f}%)
    - **Tyre-Load Variation**: Passive `{m_p['rms_dynamic_tyre_load_n']:.0f} N` → Adaptive `{m_a['rms_dynamic_tyre_load_n']:.0f} N` ({tyre_chg:+.1f}%)
    - **Actuator Saturation**: `{m_a['saturation_pct']:.1f}%` of duration operated at force limit
    """)

    # Scientifically neutral engineering trade-off interpretation
    if accel_chg < -5.0 and tyre_chg > 5.0:
        st.info("⚖️ **Engineering Trade-off**: Active control isolates chassis vertical vibration (enhancing ride comfort), accompanied by higher dynamic tyre loads. This demonstrates the fundamental comfort vs. road-holding compromise in suspension engineering.")
    elif accel_chg < -5.0:
        st.info("⚖️ **Engineering Observation**: Active control reduced chassis vertical acceleration while preserving dynamic tyre contact and rattle space within acceptable limits.")
    else:
        st.info("⚖️ **Engineering Observation**: The controller maintained stable vehicle motion with bounded actuator control efforts.")

    # --------------------------------------------------------------------------
    # 5. TRY ANOTHER SCENARIO
    # --------------------------------------------------------------------------
    if st.button("🔄 TRY ANOTHER SCENARIO", type="secondary"):
        st.session_state["last_simulation"] = None
        st.rerun()


# ==============================================================================
# ADVANCED TECHNICAL ANALYSIS (Preserved Academic & Research Inspection)
# ==============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)

with st.expander("⚙️ ADVANCED TECHNICAL ANALYSIS (Model Inspection, Optimization & Benchmarks)", expanded=False):
    st.caption("Detailed research validation modules for academic inspection, eigenvalue analysis, and comparative benchmarking.")

    adv_tabs = st.tabs([
        "⚙️ Physical Model & Poles",
        "🎛️ LQR Controller Design",
        "🐝 PSO Optimization",
        "📊 6-System Benchmark Suite",
        "🛣️ Real Measured Roads & MPC",
        "🔬 Parameter Sensitivity",
        "💥 Unilateral Tyre Contact",
        "⚠️ Assumptions & Limitations"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: PHYSICAL MODEL & POLES
    # --------------------------------------------------------------------------
    with adv_tabs[0]:
        st.subheader("Quarter-Car 2-DOF Analytical State-Space Formulation")
        v_ref = sim.get("v_params", VehicleParameters()) if sim is not None else VehicleParameters()
        qc_ref = QuarterCarModel(v_ref)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(r"""
            **Equations of Motion:**
            $$m_s \ddot{z}_s = -k_s(z_s - z_u) - c_s(\dot{z}_s - \dot{z}_u) + u$$
            $$m_u \ddot{z}_u = k_s(z_s - z_u) + c_s(\dot{z}_s - \dot{z}_u) - k_t(z_u - z_r) - u$$

            **State Vector:** $x = [z_s, \dot{z}_s, z_u, \dot{z}_u]^T$
            $$\dot{x} = A x + B u + E z_r$$
            """)
            st.write("**Matrix A:**")
            st.dataframe(pd.DataFrame(qc_ref.A, columns=["zs", "zs_dot", "zu", "zu_dot"]))
            st.write("**Matrix B:**", qc_ref.B.flatten().tolist())
            st.write("**Matrix E:**", qc_ref.E.flatten().tolist())

        with c2:
            st.markdown("**Natural Frequencies & Open-Loop Poles:**")
            st.metric("Sprung Mass Bounce Frequency", f"{v_ref.sprung_natural_frequency_hz:.2f} Hz")
            st.metric("Wheel-Hop Resonance Frequency", f"{v_ref.unsprung_natural_frequency_hz:.2f} Hz")
            st.metric("Static Normal Tyre Load", f"{v_ref.static_tyre_load:.1f} N")
            st.metric("Passive Sprung Damping Ratio", f"{v_ref.damping_ratio_sprung:.3f}")

            ol_poles = np.linalg.eigvals(qc_ref.A)
            st.write("**Open-Loop Eigenvalues:**")
            for p in ol_poles:
                st.write(f"- `{complex(p):.3f}`")

    # --------------------------------------------------------------------------
    # TAB 2: LQR CONTROLLER DESIGN & GAINS
    # --------------------------------------------------------------------------
    with adv_tabs[1]:
        st.subheader("Continuous Algebraic Riccati Equation (CARE) Formulations")
        st.markdown(r"""
        Minimizing chassis acceleration requires state-input cross-coupling $N$ since $a_s = C_a x + D_a u$:
        $$J = \int_0^\infty (x^T Q x + 2 x^T N u + u^T R u) dt$$
        """)
        nom_ctrls = build_nominal_controllers(v_ref)
        for name, ctrl in nom_ctrls.items():
            with st.expander(f"Controller: {name.replace('_', ' ').title()} LQR", expanded=(name == "balanced")):
                cfg = ctrl.config
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("q1 (Accel)", f"{cfg.q1:.1e}")
                cc2.metric("q2 (Travel)", f"{cfg.q2:.1e}")
                cc3.metric("q3 (Tyre Defl)", f"{cfg.q3:.1e}")
                cc4.metric("r0 (Effort)", f"{cfg.r0:.1e}")
                st.write("**Optimal Gain Vector K:**", ctrl.K.flatten().round(2).tolist())
                st.write("**Closed-Loop Poles:**", [f"{complex(e):.2f}" for e in ctrl.eigenvalues])

    # --------------------------------------------------------------------------
    # TAB 3: PSO OPTIMIZATION & CONVERGENCE
    # --------------------------------------------------------------------------
    with adv_tabs[2]:
        st.subheader("Offline Particle Swarm Optimization (PSO)")
        st.markdown("Optimizes diagonal log-parameters $(p_1, p_2, p_3, p_r)$ across isolated training scenarios.")

        pso_file = PROJECT_ROOT / "artifacts/optimization/pso_results.json"
        if st.button("🐝 Re-Run Offline PSO Tuning (Quick)", key="pso_btn"):
            with st.spinner("Executing swarm optimization..."):
                pso_data = run_pso_tuning(quick_mode=True)
                st.success("PSO optimization finished!")

        if pso_file.exists():
            with open(pso_file, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            fig_pso = plot_pso_convergence_interactive(p_data)
            st.plotly_chart(fig_pso, use_container_width=True)
        else:
            st.info("PSO tuning results file not found.")

    # --------------------------------------------------------------------------
    # TAB 4: 6-SYSTEM BENCHMARK SUITE
    # --------------------------------------------------------------------------
    with adv_tabs[3]:
        st.subheader("Official 6-System Comparative Benchmark Suite")
        st.markdown("Compares all 6 suspension systems across held-out test scenarios.")

        csv_path = PROJECT_ROOT / "results/tables/experiments_summary.csv"
        if st.button("📊 Run Batch Benchmark (Quick)", key="bench_btn"):
            with st.spinner("Running batch evaluation..."):
                run_comparative_benchmark(quick_mode=True)
                st.success("Batch benchmark completed!")

        if csv_path.exists():
            df_bench = pd.read_csv(csv_path)
            fig_bar = plot_comparison_bar_chart(df_bench)
            st.plotly_chart(fig_bar, use_container_width=True)

            summary_tbl = df_bench.groupby("system_display_name")[[
                "rms_body_accel_mps2",
                "peak_body_accel_mps2",
                "rms_susp_travel_m",
                "rms_dynamic_tyre_load_n",
                "saturation_pct",
                "electrical_energy_j"
            ]].mean().reset_index()
            summary_tbl["rms_susp_travel_mm"] = summary_tbl["rms_susp_travel_m"] * 1000.0
            summary_tbl = summary_tbl.drop(columns=["rms_susp_travel_m"])
            st.dataframe(summary_tbl.style.format({
                "rms_body_accel_mps2": "{:.4f}",
                "peak_body_accel_mps2": "{:.4f}",
                "rms_susp_travel_mm": "{:.2f}",
                "rms_dynamic_tyre_load_n": "{:.1f}",
                "saturation_pct": "{:.1f}%",
                "electrical_energy_j": "{:.1f}"
            }), use_container_width=True)

            csv_b = df_bench.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Benchmark CSV", csv_b, "benchmark_results.csv", "text/csv")
        else:
            st.info("Benchmark summary CSV not yet generated.")

    # --------------------------------------------------------------------------
    # TAB 5: REAL MEASURED ROADS & MPC (BENCHMARK)
    # --------------------------------------------------------------------------
    with adv_tabs[4]:
        st.subheader("Benchmark: Real Measured Road Profiles (LiRA-CD) & Constrained MPC")
        st.markdown("""
        Evaluates the primary 3-way comparative study:
        1. **Passive Suspension** (Reference Baseline)
        2. **Fixed Nominal LQR** (Retained Benchmark)
        3. **Constrained Discrete-Time MPC** (Primary Proposed Comparison)
        
        Using real physical profilometer data from the Danish Road Directorate P79 Laser Survey (LiRA-CD dataset).
        """)

        real_csv = PROJECT_ROOT / "results/tables/real_road_benchmark_summary.csv"

        if st.button("🚀 Re-Run Real Road Benchmark (Passive vs LQR vs MPC)", key="real_road_btn"):
            with st.spinner("Executing real-road simulations on LiRA-CD Highway M13 and Urban CPH1..."):
                run_real_road_study()
                st.success("Real-road comparative benchmark completed!")

        if real_csv.exists():
            df_real = pd.read_csv(real_csv)
            st.markdown("### Measured Road Benchmark Summary")
            st.dataframe(
                df_real[[
                    "trace_key", "speed_kmh", "system_name",
                    "rms_body_accel_mps2", "peak_body_accel_mps2",
                    "rms_susp_travel_m", "rms_dynamic_tyre_load_n",
                    "electrical_energy_j"
                ]].style.format({
                    "rms_body_accel_mps2": "{:.4f}",
                    "peak_body_accel_mps2": "{:.4f}",
                    "rms_susp_travel_m": "{:.4f}",
                    "rms_dynamic_tyre_load_n": "{:.1f}",
                    "electrical_energy_j": "{:.1f}"
                }),
                use_container_width=True
            )

            # Show pre-generated publication figures
            fig_bar_file = PROJECT_ROOT / "results/figures/real_road_comparison_bar_chart.png"
            fig_ts_m13 = PROJECT_ROOT / "results/figures/real_road_time_series_lira_highway_m13.png"
            fig_ts_cph = PROJECT_ROOT / "results/figures/real_road_time_series_lira_urban_cph1.png"

            if fig_bar_file.exists():
                st.image(str(fig_bar_file), caption="Comparative Metrics Across Real Measured Roads", use_container_width=True)

            col_ts1, col_ts2 = st.columns(2)
            with col_ts1:
                if fig_ts_m13.exists():
                    st.image(str(fig_ts_m13), caption="Time Series: LiRA Highway M13", use_container_width=True)
            with col_ts2:
                if fig_ts_cph.exists():
                    st.image(str(fig_ts_cph), caption="Time Series: LiRA Urban CPH1", use_container_width=True)
        else:
            st.info("Real road benchmark CSV not yet generated. Click button above to run.")

        st.markdown("---")
        st.markdown("### Reference Paper Replication Study (Deshpande et al., 2014)")
        st.markdown("Compares **Passive vs Fixed LQR vs Constrained MPC vs DOSMC** on academic benchmark profiles and real measured roads.")

        ref_csv = PROJECT_ROOT / "results/tables/reference_study_comparison.csv"
        ref_fig = PROJECT_ROOT / "results/figures/reference_paper_benchmark.png"

        if ref_csv.exists():
            df_ref = pd.read_csv(ref_csv)
            st.dataframe(
                df_ref[[
                    "road_name", "controller_name", "rms_body_accel_mps2",
                    "accel_imprv_pct", "rms_dynamic_tyre_load_n", "electrical_energy_j"
                ]].style.format({
                    "rms_body_accel_mps2": "{:.4f}",
                    "accel_imprv_pct": "{:+.1f}%",
                    "rms_dynamic_tyre_load_n": "{:.1f}",
                    "electrical_energy_j": "{:.1f}"
                }),
                use_container_width=True
            )
        if ref_fig.exists():
            st.image(str(ref_fig), caption="Reference Paper Benchmark Response Comparison", use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 6: PARAMETER SENSITIVITY
    # --------------------------------------------------------------------------
    with adv_tabs[5]:
        st.subheader("Vehicle Parameter Robustness Study")
        st.markdown("Evaluates controller stability under payload mass variations, spring softening, and damper wear.")

        sens_csv = PROJECT_ROOT / "results/tables/sensitivity_analysis.csv"
        if st.button("🔬 Re-run Sensitivity Study", key="sens_btn"):
            with st.spinner("Running parameter sensitivity study..."):
                run_parameter_sensitivity_study()
                st.success("Sensitivity analysis completed!")

        if sens_csv.exists():
            df_sens = pd.read_csv(sens_csv)
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Bar(x=df_sens["variation_case"], y=df_sens["passive_rms_accel_mps2"], name="Passive Suspension", marker_color="#94a3b8"))
            fig_sens.add_trace(go.Bar(x=df_sens["variation_case"], y=df_sens["active_rms_accel_mps2"], name="Active LQR (Nominal)", marker_color="#2563eb"))
            fig_sens.update_layout(barmode="group", title="RMS Acceleration Across Parameter Variations", template="plotly_white")
            st.plotly_chart(fig_sens, use_container_width=True)
            st.dataframe(df_sens, use_container_width=True)
        else:
            st.info("Sensitivity data not found.")

    # --------------------------------------------------------------------------
    # TAB 7: UNILATERAL TYRE CONTACT
    # --------------------------------------------------------------------------
    with adv_tabs[6]:
        st.subheader("Unilateral Tyre Contact Dynamics (Tyre Lift-off)")
        st.markdown("Examines landing impact shock and loss-of-contact dynamics ($F_{tyre} \ge 0$).")

        uni_json = PROJECT_ROOT / "results/tables/unilateral_contact_metrics.json"
        if st.button("💥 Re-run Unilateral Contact Study", key="uni_btn"):
            with st.spinner("Simulating unilateral contact dynamics..."):
                run_unilateral_contact_comparison()
                st.success("Unilateral contact analysis finished!")

        if uni_json.exists():
            with open(uni_json, "r", encoding="utf-8") as f:
                u_data = json.load(f)
            lin_m = u_data["linear_model"]
            uni_m = u_data["unilateral_model"]
            uc1, uc2, uc3 = st.columns(3)
            uc1.metric("Linear Model Peak Accel", f"{lin_m['peak_body_accel_mps2']:.2f} m/s²")
            uc2.metric("Unilateral Peak Accel (Shock)", f"{uni_m['peak_body_accel_mps2']:.2f} m/s²")
            uc3.metric("Airborne Duration", f"{uni_m['contact_loss_duration_s']*1000.0:.0f} ms")
        else:
            st.info("Unilateral contact metrics not found.")

    # --------------------------------------------------------------------------
    # TAB 8: ASSUMPTIONS & LIMITATIONS
    # --------------------------------------------------------------------------
    with adv_tabs[7]:
        st.subheader("Academic Scope, Physical Assumptions, and Engineering Limitations")
        st.markdown("""
        1. **Quarter-Car Kinematic Model**: Represents single-corner heave dynamics. Pitch, roll, and load transfers during braking/cornering are omitted.
        2. **Sensor Perception Delay**: Rolling-window feature extraction requires a 0.5 s delay buffer.
        3. **Actuator Bandwidth & Saturation**: Saturation ($\pm F_{max}$) and slew-rate limits are dynamically enforced.
        4. **Tyre Road Contact**: Dynamic tyre deflection is modeled via radial stiffness $k_t$.
        5. **Prototype Disclaimer**: This software is an academic engineering simulation tool. It does not establish real-vehicle safety certification or automotive roadworthiness.
        """)

