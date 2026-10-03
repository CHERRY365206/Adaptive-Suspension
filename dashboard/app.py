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

from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel, SimulationResult
from models.actuator import ActuatorModel
from roads.road_dataset import create_scenario
from controllers.lqr import LQRConfig, LQRController
from controllers.controller_registry import get_nominal_configs, build_nominal_controllers
from controllers.adaptive_selector import DEFAULT_ROAD_TO_CONTROLLER_MAP
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
from dashboard.visualizations import (
    plot_simulation_time_series,
    plot_comparison_bar_chart,
    plot_pso_convergence_interactive,
    plot_feature_importance_interactive,
    plot_road_profile,
    plot_body_acceleration_comparison,
    plot_suspension_travel_comparison,
    plot_tyre_load_comparison,
    plot_actuator_force
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
# VEHICLE PRESETS CONFIGURATION
# ==============================================================================
VEHICLE_PRESETS = {
    "Sedan EV": {
        "ms": 450.0,
        "mu": 45.0,
        "ks": 28000.0,
        "cs": 1500.0,
        "kt": 190000.0,
        "max_travel": 0.08,
        "f_max": 1800.0,
        "slew_rate": 18000.0,
        "label": "Sedan EV (450 kg chassis, 1800 N actuator)"
    },
    "Compact EV": {
        "ms": 350.0,
        "mu": 35.0,
        "ks": 24000.0,
        "cs": 1300.0,
        "kt": 175000.0,
        "max_travel": 0.08,
        "f_max": 1500.0,
        "slew_rate": 15000.0,
        "label": "Compact EV (350 kg chassis, 1500 N actuator)"
    }
}

ROAD_OPTIONS = {
    "Smooth Highway": "smooth",
    "Speed Bump": "bumpy",
    "Rough Road": "rough"
}


# ==============================================================================
# HEADER & SCENARIO SELECTION
# ==============================================================================
st.markdown('<div class="simulator-title">🚗 ADAPTIVE EV SUSPENSION SIMULATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="simulator-subtitle">Simulate how an intelligent suspension responds to different driving conditions.</div>', unsafe_allow_html=True)

st.subheader("DRIVING SCENARIO")

col_veh, col_road, col_spd, col_dur = st.columns([1.3, 1.3, 1.2, 1.0])

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
        index=1,
        help="Road excitation surface (ISO Class A, Versine Speed Bump, or ISO Class C/D Rough)."
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

# Vehicle parameter assignment
if vehicle_choice in VEHICLE_PRESETS:
    preset = VEHICLE_PRESETS[vehicle_choice]
    ms_val = preset["ms"]
    mu_val = preset["mu"]
    ks_val = preset["ks"]
    cs_val = preset["cs"]
    kt_val = preset["kt"]
    max_travel_val = preset["max_travel"]
    fmax_val = preset["f_max"]
    slew_val = preset["slew_rate"]
else:
    # Custom EV parameters exposed in expander
    with st.expander("▼ Advanced Vehicle Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            ms_val = st.slider("Sprung Mass ms [kg]", 250.0, 700.0, 450.0, 10.0)
            mu_val = st.slider("Unsprung Mass mu [kg]", 25.0, 80.0, 45.0, 5.0)
        with c2:
            ks_val = st.slider("Suspension Stiffness ks [N/m]", 15000.0, 45000.0, 28000.0, 1000.0)
            cs_val = st.slider("Suspension Damping cs [Ns/m]", 600.0, 3000.0, 1500.0, 100.0)
        with c3:
            kt_val = st.slider("Tyre Stiffness kt [N/m]", 120000.0, 260000.0, 190000.0, 5000.0)
            fmax_val = st.slider("Maximum Actuator Force Fmax [N]", 500.0, 3000.0, 1800.0, 100.0)
            slew_val = st.slider("Actuator Slew Rate [N/s]", 5000.0, 35000.0, 18000.0, 1000.0)
        max_travel_val = 0.08


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
if run_simulation_btn:
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
            ground_truth_class=road_type_key,
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

        # STEP 7: Controller Selection & PSO-Tuned Gains
        controller_tag = DEFAULT_ROAD_TO_CONTROLLER_MAP.get(ai_detected_road, "balanced")
        controller_display = controller_tag.replace("_", " ").title() + " LQR"
        progress_box.write(f"✓ AI detected road: '{ai_detected_road.capitalize()}' ({ai_conf*100.0:.1f}% confidence) → Selecting {controller_display}...")

        # STEP 8: AI Adaptive Suspension Simulation on EXACT SAME road realization
        progress_box.write("✓ Simulating AI Adaptive Suspension with PSO-tuned gains and actuator constraints...")
        res_adaptive = suite.simulate_system(
            system_type="proposed_ai_pso",
            road_func=road_fn,
            ground_truth_class=road_type_key,
            duration_s=duration_s,
            dt=0.002
        )

        # STEP 9: Compute Performance Metrics
        progress_box.write("✓ Computing standardized engineering performance metrics...")
        metrics_passive = compute_simulation_metrics(res_passive, ground_truth_class=road_type_key)
        metrics_adaptive = compute_simulation_metrics(res_adaptive, ground_truth_class=road_type_key)

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
    st.subheader("🧠 AI DECISION")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Test Scenario", sim["road_choice"])
    with d2:
        st.metric("AI Detected Road", sim["ai_detected_road"].capitalize())
    with d3:
        st.metric("Detection Confidence", f"{sim['ai_conf'] * 100.0:.1f}%")
    with d4:
        st.metric("Controller Selected", sim["controller_display"])

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
        "🔬 Parameter Sensitivity",
        "💥 Unilateral Tyre Contact",
        "⚠️ Assumptions & Limitations"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: PHYSICAL MODEL & POLES
    # --------------------------------------------------------------------------
    with adv_tabs[0]:
        st.subheader("Quarter-Car 2-DOF Analytical State-Space Formulation")
        v_ref = VehicleParameters()
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
    # TAB 5: PARAMETER SENSITIVITY
    # --------------------------------------------------------------------------
    with adv_tabs[4]:
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
    # TAB 6: UNILATERAL TYRE CONTACT
    # --------------------------------------------------------------------------
    with adv_tabs[5]:
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
    # TAB 7: ASSUMPTIONS & LIMITATIONS
    # --------------------------------------------------------------------------
    with adv_tabs[6]:
        st.subheader("Academic Scope, Physical Assumptions, and Engineering Limitations")
        st.markdown("""
        1. **Quarter-Car Kinematic Model**: Represents single-corner heave dynamics. Pitch, roll, and load transfers during braking/cornering are omitted.
        2. **Sensor Perception Delay**: Rolling-window feature extraction requires a 0.5 s delay buffer.
        3. **Actuator Bandwidth & Saturation**: Saturation ($\pm F_{max}$) and slew-rate limits are dynamically enforced.
        4. **Tyre Road Contact**: Dynamic tyre deflection is modeled via radial stiffness $k_t$.
        5. **Prototype Disclaimer**: This software is an academic engineering simulation tool. It does not establish real-vehicle safety certification or automotive roadworthiness.
        """)
