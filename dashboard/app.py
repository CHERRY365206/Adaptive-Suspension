"""Streamlit Interactive Academic Engineering Dashboard for Active Suspension EV Simulation."""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from models.vehicle_parameters import VehicleParameters
from models.quarter_car import QuarterCarModel, SimulationResult
from models.actuator import ActuatorModel
from roads.smooth import SmoothRoadGenerator
from roads.bump import BumpRoadGenerator
from roads.rough import RoughRoadGenerator
from roads.road_dataset import create_scenario
from controllers.lqr import LQRConfig, LQRController
from controllers.controller_registry import get_nominal_configs, build_nominal_controllers
from controllers.adaptive_selector import DEFAULT_ROAD_TO_CONTROLLER_MAP
from ai.predict_road import RoadPredictor
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
    plot_feature_importance_interactive
)

st.set_page_config(
    page_title="AI-Assisted Active Suspension EV Simulation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .card { background-color: #F8FAFC; border-radius: 8px; padding: 1.2rem; border-left: 5px solid #2563EB; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("🎛️ Simulation Parameters")

st.sidebar.subheader("Quarter-Car EV Chassis")
ms_val = st.sidebar.slider("Sprung Mass $m_s$ [kg]", min_value=300.0, max_value=650.0, value=450.0, step=10.0)
mu_val = st.sidebar.slider("Unsprung Mass $m_u$ [kg]", min_value=30.0, max_value=70.0, value=45.0, step=5.0)
ks_val = st.sidebar.slider("Suspension Stiffness $k_s$ [N/m]", min_value=15000.0, max_value=40000.0, value=28000.0, step=1000.0)
cs_val = st.sidebar.slider("Suspension Damping $c_s$ [Ns/m]", min_value=800.0, max_value=2500.0, value=1500.0, step=100.0)
kt_val = st.sidebar.slider("Tyre Stiffness $k_t$ [N/m]", min_value=140000.0, max_value=240000.0, value=190000.0, step=5000.0)

st.sidebar.subheader("Active Actuator Constraints")
fmax_val = st.sidebar.slider("Max Force $F_{max}$ [N]", min_value=500.0, max_value=3000.0, value=1800.0, step=100.0)
slew_val = st.sidebar.slider("Slew Rate Limit [N/s]", min_value=5000.0, max_value=30000.0, value=18000.0, step=1000.0)

st.sidebar.subheader("Road Excitation")
road_type_select = st.sidebar.selectbox("Road Profile Type", ["bumpy", "rough", "smooth"])
speed_kmh_select = st.sidebar.slider("Vehicle Forward Speed [km/h]", min_value=20.0, max_value=110.0, value=50.0, step=5.0)
duration_select = st.sidebar.slider("Simulation Duration [s]", min_value=2.0, max_value=10.0, value=6.0, step=1.0)

# Instantiate vehicles and actuators from sidebar values
try:
    v_params = VehicleParameters(
        ms=ms_val,
        mu=mu_val,
        ks=ks_val,
        cs=cs_val,
        kt=kt_val,
        max_travel=0.08,
        g=9.81
    )
    actuator = ActuatorModel(
        max_force=fmax_val,
        min_force=-fmax_val,
        slew_rate_limit=slew_val
    )
except ValueError as e:
    st.sidebar.error(f"Parameter Error: {e}")
    st.stop()


# ==============================================================================
# MAIN TABS INTERFACE
# ==============================================================================
st.markdown('<div class="main-title">AI-Assisted Adaptive Active Suspension System for Electric Vehicles</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Quarter-Car Dynamics • Intelligent Road Classification • PSO LQR Optimization • Benchmark Suite</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "📖 Overview & Architecture",
    "⚙️ Physical Model & Poles",
    "🛣️ Road Profile Generator",
    "🧠 AI Road Classifier",
    "🎛️ LQR Controller Design",
    "🐝 PSO Optimization",
    "📈 Dynamic Simulation",
    "📊 6-System Benchmark",
    "🔬 Sensitivity Analysis",
    "💥 Unilateral Contact",
    "⚠️ Assumptions & Limitations"
])


# ------------------------------------------------------------------------------
# TAB 1: OVERVIEW & ARCHITECTURE
# ------------------------------------------------------------------------------
with tabs[0]:
    st.header("Project Architecture and Mathematical Framework")
    st.markdown("""
    This platform implements a complete, mathematically validated active suspension simulation for electric vehicles.
    It combines classical optimal control (LQR with state-input cross-coupling for vertical acceleration minimization)
    with machine learning (rolling-window Random Forest road classifier) and swarm intelligence (offline PSO controller tuning).
    """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Physical Quarter-Car Model")
        st.markdown(r"""
        The 2-DOF quarter-car model represents one corner of an electric passenger vehicle:
        - **Sprung Mass ($m_s$):** Quarter vehicle chassis + EV high-voltage battery payload.
        - **Unsprung Mass ($m_u$):** Wheel, hub assembly, and brake disk.
        - **Suspension ($k_s, c_s$):** Parallel coil spring and passive viscous damper.
        - **Active Actuator ($u$):** Electromagnetic or electro-hydraulic linear force generator.
        - **Tyre ($k_t$):** Radial stiffness acting on ground elevation $z_r(t)$.

        **Equations of Motion:**
        $$m_s \ddot{z}_s = -k_s(z_s - z_u) - c_s(\dot{z}_s - \dot{z}_u) + u$$
        $$m_u \ddot{z}_u = k_s(z_s - z_u) + c_s(\dot{z}_s - \dot{z}_u) - k_t(z_u - z_r) - u$$
        """)

    with col2:
        st.subheader("System Data Flow & Control Architecture")
        st.markdown(r"""
        ```
        [ Road Surface zr(s) ] ---> [ Quarter-Car Dynamics ]
                                        |  (Sensor Accelerations & Deflection)
                                        v
                            [ Rolling Window (0.5s) ]
                                        |  (Time & Spectral Features)
                                        v
                            [ Random Forest Classifier ]
                                        |  (Road Class: Smooth/Bumpy/Rough)
                                        v
                            [ Adaptive Selector (Dwell & Bumpless) ]
                                        |  (Switches among 3 PSO-tuned LQRs)
                                        v
                            [ Actuator Saturation & Slew Limiter ]
                                        |  (Applied Force u_applied)
                                        +---------------------------------> Closed-Loop Feedback
        ```
        """)

    st.info("💡 **Academic Reproducibility Notice:** All metrics, figures, and charts rendered in this dashboard are generated strictly from first-principles simulations and saved model artifacts—zero values are fabricated or hardcoded.")


# ------------------------------------------------------------------------------
# TAB 2: PHYSICAL MODEL & POLES
# ------------------------------------------------------------------------------
with tabs[1]:
    st.header("Quarter-Car EV Dynamics and Open-Loop Characteristics")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("State-Space Formulation")
        st.markdown(r"""
        With state vector $x = [z_s, \dot{z}_s, z_u, \dot{z}_u]^T$, the continuous system is:
        $$\dot{x} = A x + B u + E z_r$$
        """)
        qc = QuarterCarModel(v_params)
        st.write("**Matrix A (System Dynamics):**")
        st.dataframe(pd.DataFrame(qc.A, columns=["zs", "zs_dot", "zu", "zu_dot"]))
        st.write("**Matrix B (Actuator Input):**", qc.B.flatten())
        st.write("**Matrix E (Road Disturbance):**", qc.E.flatten())

    with col2:
        st.subheader("Natural Frequencies & Static Equilibrium")
        f_sprung = v_params.sprung_natural_frequency_hz
        f_unsprung = v_params.unsprung_natural_frequency_hz
        f_static = v_params.static_tyre_load
        zeta = v_params.damping_ratio_sprung

        st.metric("Sprung Mass Bounce Frequency", f"{f_sprung:.2f} Hz")
        st.metric("Wheel-Hop Resonance Frequency", f"{f_unsprung:.2f} Hz")
        st.metric("Static Normal Tyre Force", f"{f_static:.1f} N")
        st.metric("Passive Sprung Damping Ratio", f"{zeta:.3f}")

        # Open-loop poles
        ol_poles = np.linalg.eigvals(qc.A)
        st.write("**Open-Loop Eigenvalues:**")
        for p in ol_poles:
            st.write(f"- Real: `{np.real(p):.3f}`, Imag: `{np.imag(p):.3f} rad/s`")


# ------------------------------------------------------------------------------
# TAB 3: ROAD PROFILE GENERATOR
# ------------------------------------------------------------------------------
with tabs[2]:
    st.header("Synthetic Road Profiles (ISO 8608 Standards & Versine Humps)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("ISO Class A Smooth")
        st.caption("High-grade asphalt highway with micro-roughness")
    with col2:
        st.subheader("Speed Bump")
        st.caption("C1-continuous versine hump (no impulse artifacts)")
    with col3:
        st.subheader("ISO Class C/D Rough")
        st.caption("Degraded asphalt or cobblestone stochastic surface")

    speed_mps = speed_kmh_select / 3.6
    t_plot = np.linspace(0, duration_select, 1000)

    # Generate road
    sc_obj = create_scenario(scenario_id=1, road_type=road_type_select, speed_kmh=speed_kmh_select, split="test", duration_s=duration_select)
    zr_plot = sc_obj.road_profile.elevation_time(t_plot, speed_mps)

    fig_road = go.Figure()
    fig_road.add_trace(go.Scatter(x=t_plot, y=zr_plot * 1000.0, name="Elevation", line=dict(color="#1f77b4", width=2)))
    fig_road.update_layout(
        title=f"Road Surface Profile vs Time: {sc_obj.description}",
        xaxis_title="Time [s]",
        yaxis_title="Road Elevation [mm]",
        template="plotly_white"
    )
    st.plotly_chart(fig_road, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: AI ROAD CLASSIFIER
# ------------------------------------------------------------------------------
with tabs[3]:
    st.header("Intelligent Road Condition Classification Pipeline")
    st.markdown("""
    The Random Forest classifier receives **simulated sensor outputs only** (chassis acceleration, wheel acceleration,
    suspension deflection, and relative velocity). It has **zero direct access** to true road labels or future data.
    """)

    metrics_file = PROJECT_ROOT / "results/tables/classifier_metrics.json"
    model_file = PROJECT_ROOT / "artifacts/models/road_classifier.joblib"

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button("🚀 Train Road Classifier (Quick)"):
            with st.spinner("Simulating multi-scenario data and training Random Forest..."):
                train_metrics = train_road_classifier(quick_mode=True)
                st.success("Classifier training complete!")

    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            m = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Accuracy", f"{m['accuracy'] * 100.0:.1f}%")
        col2.metric("Macro Precision", f"{m['macro_precision'] * 100.0:.1f}%")
        col3.metric("Macro Recall", f"{m['macro_recall'] * 100.0:.1f}%")
        col4.metric("Macro F1-Score", f"{m['macro_f1'] * 100.0:.1f}%")

        col_cm, col_imp = st.columns([1, 1])
        with col_cm:
            st.subheader("Confusion Matrix (Held-out Scenarios)")
            cm = m["confusion_matrix"]
            cm_df = pd.DataFrame(cm, index=["Smooth", "Bumpy", "Rough"], columns=["Smooth", "Bumpy", "Rough"])
            st.dataframe(cm_df.style.background_gradient(cmap="Blues"))

        with col_imp:
            st.subheader("Top Predictive Sensor Features")
            fig_imp = plot_feature_importance_interactive(m.get("feature_importances", {}))
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("ℹ️ Classifier model not yet trained. Click the button above to train on independent scenarios.")


# ------------------------------------------------------------------------------
# TAB 5: LQR CONTROLLER DESIGN
# ------------------------------------------------------------------------------
with tabs[4]:
    st.header("LQR Formulations and State-Input Cross-Coupling")
    st.markdown(r"""
    Minimizing vertical body acceleration $a_s = \ddot{z}_s$ requires accounting for the fact that
    body acceleration depends on **both** the state $x$ and the actuator force $u$:
    $$a_s = C_a x + D_a u, \quad \text{where } C_a = [-k_s/m_s, -c_s/m_s, k_s/m_s, c_s/m_s], \quad D_a = 1/m_s$$
    The general quadratic cost functional introduces cross-coupling $N$:
    $$J = \int_0^\infty (x^T Q x + 2 x^T N u + u^T R u) dt$$
    """)

    nom_controllers = build_nominal_controllers(v_params)
    for name, ctrl in nom_controllers.items():
        with st.expander(f"Controller: {name.replace('_', ' ').title()} LQR", expanded=(name == "balanced")):
            cfg = ctrl.config
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accel Penalty $q_1$", f"{cfg.q1:.1e}")
            c2.metric("Travel Penalty $q_2$", f"{cfg.q2:.1e}")
            c3.metric("Tyre Deflection $q_3$", f"{cfg.q3:.1e}")
            c4.metric("Effort Penalty $r_0$", f"{cfg.r0:.1e}")

            st.write("**Optimal State-Feedback Gain Vector $K$:**")
            st.code(f"K = {ctrl.K.flatten().round(2).tolist()}")
            st.write("**Closed-Loop Poles (Re(lambda) < 0):**")
            st.write([f"{complex(e):.2f}" for e in ctrl.eigenvalues])


# ------------------------------------------------------------------------------
# TAB 6: PSO OPTIMIZATION
# ------------------------------------------------------------------------------
with tabs[5]:
    st.header("Particle Swarm Optimization (PSO) of Controller Weights")
    st.markdown("""
    PSO optimizes the 4 diagonal LQR parameters in logarithmic space ($q_1, q_2, q_3, r_0$)
    on isolated training scenarios.
    """)

    pso_res_file = PROJECT_ROOT / "artifacts/optimization/pso_results.json"

    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        if st.button("🐝 Run Offline PSO Tuning (Quick)"):
            with st.spinner("Executing swarm optimization on training scenarios..."):
                pso_data = run_pso_tuning(quick_mode=True)
                st.success("PSO optimization finished!")

    if pso_res_file.exists():
        with open(pso_res_file, "r", encoding="utf-8") as f:
            p_data = json.load(f)

        fig_pso = plot_pso_convergence_interactive(p_data)
        st.plotly_chart(fig_pso, use_container_width=True)

        for mode, data in p_data.items():
            with st.expander(f"Tuned {mode.capitalize()} LQR Weights"):
                st.write(f"**Best Objective Cost:** `{data['best_cost']:.4f}`")
                st.write(f"**Tuned Gain K:** `{np.array(data['gain_K']).round(2).tolist()}`")
    else:
        st.info("ℹ️ PSO tuning artifacts not yet generated. Run PSO Tuning using the button above.")


# ------------------------------------------------------------------------------
# TAB 7: DYNAMIC SIMULATION
# ------------------------------------------------------------------------------
with tabs[6]:
    st.header("Live Dynamic Simulation Execution")
    system_choice = st.selectbox(
        "Select Suspension Configuration to Simulate",
        [
            ("proposed_ai_pso", "Proposed AI + PSO Adaptive LQR"),
            ("passive", "Passive Suspension (Actuator Off)"),
            ("fixed_nominal_lqr", "Fixed Nominal LQR (Balanced)"),
            ("fixed_pso_lqr", "Fixed PSO-Optimized LQR"),
            ("adaptive_no_pso", "Adaptive LQR (Heuristic / No PSO)"),
            ("oracle_adaptive", "Oracle Adaptive LQR (Reference)")
        ],
        format_func=lambda x: x[1]
    )[0]

    suite = SuspensionBenchmarkSuite(vehicle_params=v_params, actuator=actuator)
    sc = create_scenario(scenario_id=555, road_type=road_type_select, speed_kmh=speed_kmh_select, split="test", duration_s=duration_select)
    road_fn = sc.road_profile.get_time_callable(sc.speed_mps)

    with st.spinner("Simulating quarter-car dynamics..."):
        sim_res = suite.simulate_system(
            system_type=system_choice,
            road_func=road_fn,
            ground_truth_class=sc.road_type,
            duration_s=duration_select,
            dt=0.002
        )
        sim_metrics = compute_simulation_metrics(sim_res, ground_truth_class=sc.road_type)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("RMS Body Accel", f"{sim_metrics['rms_body_accel_mps2']:.3f} m/s²")
    m2.metric("Peak Body Accel", f"{sim_metrics['peak_body_accel_mps2']:.3f} m/s²")
    m3.metric("RMS Travel", f"{sim_metrics['rms_susp_travel_m']*1000.0:.1f} mm")
    m4.metric("RMS Tyre Load", f"{sim_metrics['rms_dynamic_tyre_load_n']:.0f} N")
    m5.metric("Actuator Saturation", f"{sim_metrics['saturation_pct']:.1f}%")
    m6.metric("Min Tyre Contact", f"{sim_metrics['min_total_tyre_load_n']:.0f} N")

    fig_sim = plot_simulation_time_series(sim_res, title=f"Simulation: {SYSTEM_DISPLAY_NAMES[system_choice]} on {sc.description}")
    st.plotly_chart(fig_sim, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 8: 6-SYSTEM BENCHMARK
# ------------------------------------------------------------------------------
with tabs[7]:
    st.header("Comparative Benchmark Suite Across All 6 Systems")
    st.markdown("""
    Evaluates all five systems plus the idealized oracle reference across the official
    held-out scenario matrix using identical road profiles, vehicle masses, and tolerances.
    """)

    csv_path = PROJECT_ROOT / "results/tables/experiments_summary.csv"

    if st.button("📊 Run Full Benchmark Comparison (Quick)"):
        with st.spinner("Running batch evaluation across held-out scenarios..."):
            bench_res = run_comparative_benchmark(quick_mode=True)
            st.success("Benchmark completed and results saved!")

    if csv_path.exists():
        df_bench = pd.read_csv(csv_path)

        st.subheader("Benchmark Comparison Bar Chart")
        fig_bar = plot_comparison_bar_chart(df_bench)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Aggregated Performance Summary Table")
        summary_table = df_bench.groupby("system_display_name")[[
            "rms_body_accel_mps2",
            "peak_body_accel_mps2",
            "rms_susp_travel_m",
            "rms_dynamic_tyre_load_n",
            "saturation_pct",
            "electrical_energy_j"
        ]].mean().reset_index()

        summary_table["rms_susp_travel_mm"] = summary_table["rms_susp_travel_m"] * 1000.0
        summary_table = summary_table.drop(columns=["rms_susp_travel_m"])

        st.dataframe(summary_table.style.format({
            "rms_body_accel_mps2": "{:.3f}",
            "peak_body_accel_mps2": "{:.3f}",
            "rms_susp_travel_mm": "{:.1f}",
            "rms_dynamic_tyre_load_n": "{:.0f}",
            "saturation_pct": "{:.1f}%",
            "electrical_energy_j": "{:.1f}"
        }))

        # CSV Download Button
        csv_bytes = df_bench.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Detailed Benchmark Data (CSV)",
            data=csv_bytes,
            file_name="active_suspension_benchmark_results.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ Benchmark results not yet computed. Click the button above to run the batch experiment.")


# ------------------------------------------------------------------------------
# TAB 9: SENSITIVITY ANALYSIS
# ------------------------------------------------------------------------------
with tabs[8]:
    st.header("Vehicle Parameter Sensitivity & Robustness Analysis")
    st.markdown("""
    Evaluates whether the active suspension maintains its vibration isolation advantages
    when actual physical EV parameters deviate from the nominal model used for control design.
    Simulated variations include **battery payload mass changes (±20%)**, **damper degradation (-30%)**,
    **spring softening (-10%)**, and **tyre pressure fluctuations (±15%)**.
    """)

    sens_csv = PROJECT_ROOT / "results/tables/sensitivity_analysis.csv"

    if st.button("🔬 Run Sensitivity Analysis Study"):
        with st.spinner("Simulating parameter variation matrix..."):
            df_sens = run_parameter_sensitivity_study()
            st.success("Sensitivity analysis completed!")

    if sens_csv.exists():
        df_sens = pd.read_csv(sens_csv)

        # Plot sensitivity bar chart
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Bar(
            x=df_sens["variation_case"],
            y=df_sens["passive_rms_accel_mps2"],
            name="Passive Suspension",
            marker_color="#7f7f7f"
        ))
        fig_sens.add_trace(go.Bar(
            x=df_sens["variation_case"],
            y=df_sens["active_rms_accel_mps2"],
            name="Active LQR (Nominal Controller)",
            marker_color="#1f77b4"
        ))
        fig_sens.update_layout(
            barmode="group",
            title="RMS Body Acceleration Across Physical Parameter Variations",
            xaxis_title="Parameter Perturbation Case",
            yaxis_title="RMS Acceleration [m/s²]",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        st.subheader("Parameter Sensitivity Data Table")
        st.dataframe(df_sens.style.format({
            "passive_rms_accel_mps2": "{:.3f}",
            "active_rms_accel_mps2": "{:.3f}",
            "accel_reduction_pct": "{:.1f}%",
            "passive_peak_travel_mm": "{:.1f}",
            "active_peak_travel_mm": "{:.1f}",
            "active_rms_force_n": "{:.0f}"
        }))
    else:
        st.info("ℹ️ Sensitivity study not yet run. Click the button above to execute.")


# ------------------------------------------------------------------------------
# TAB 10: UNILATERAL CONTACT DYNAMICS
# ------------------------------------------------------------------------------
with tabs[9]:
    st.header("Unilateral Tyre Contact Dynamics (Tyre Lift-off & Landing Impact)")
    st.markdown("""
    In the standard linear suspension model, the tyre is modeled as a bilateral spring that can pull
    downward on the vehicle wheel even during severe lift-off. In the **unilateral contact model**,
    the tyre separates from the ground when normal load drops to zero ($F_{tyre} \ge 0$), causing
    the wheel to remain airborne longer and land with an **impact shock**.
    """)

    uni_json = PROJECT_ROOT / "results/tables/unilateral_contact_metrics.json"

    if st.button("💥 Run Unilateral Tyre Contact Study"):
        with st.spinner("Simulating severe bump with unilateral contact dynamics..."):
            uni_metrics = run_unilateral_contact_comparison()
            st.success("Unilateral contact simulation completed!")

    if uni_json.exists():
        with open(uni_json, "r", encoding="utf-8") as f:
            u_data = json.load(f)

        lin_m = u_data["linear_model"]
        uni_m = u_data["unilateral_model"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Linear Model Peak Accel", f"{lin_m['peak_body_accel_mps2']:.2f} m/s²")
        c2.metric("Unilateral Peak Accel (Shock)", f"{uni_m['peak_body_accel_mps2']:.2f} m/s²",
                  delta=f"{uni_m['peak_body_accel_mps2'] - lin_m['peak_body_accel_mps2']:+.2f} m/s²", delta_color="inverse")
        c3.metric("Contact Loss Duration", f"{uni_m['contact_loss_duration_s'] * 1000.0:.0f} ms")

        st.subheader("Key Scientific Observations:")
        st.markdown(f"""
        1. **Impact Shock**: When the tyre lifts off and lands back on the road surface, the unilateral contact model generates a higher peak acceleration ({uni_m['peak_body_accel_mps2']:.2f} m/s² vs {lin_m['peak_body_accel_mps2']:.2f} m/s² in the linear model).
        2. **Airborne Duration**: Tyre contact was completely lost for **{uni_m['contact_loss_duration_s'] * 1000.0:.0f} ms** ({uni_m['contact_loss_pct']:.1f}% of simulation).
        3. **Constraint Enforcement**: In the linear approximation, estimated normal load dropped to `{lin_m['min_total_tyre_load_n']:.0f} N` (tensile adhesion), which is physically impossible on actual asphalt.
        """)
    else:
        st.info("ℹ️ Unilateral contact study not yet run. Click the button above to execute.")


# ------------------------------------------------------------------------------
# TAB 11: ASSUMPTIONS & LIMITATIONS
# ------------------------------------------------------------------------------
with tabs[10]:
    st.header("Academic Scope, Physical Assumptions, and Engineering Limitations")
    st.markdown("""
    ### 1. Quarter-Car Dynamics Simplification
    - The 2-DOF quarter-car model isolates heave (vertical body motion) and wheel-hop.
    - Pitch, roll, yaw, longitudinal load transfer (under acceleration/braking), and warp kinematics are omitted.
    - Results represent single-corner performance and should not be equated to full-vehicle road tests.

    ### 2. Sensor and Road Perception
    - Sensor signals are simulated from the state-space ODEs without simulated electrical EMI noise or high-frequency drift.
    - Real-world road perception suffers from window lag (0.5 s delay in feature extraction) and preview uncertainty.

    ### 3. Actuator Constraints & Linearity
    - LQR assumes an unconstrained linear plant during gain synthesis.
    - In practice, severe road shocks cause force saturation ($F_{max} = 1800$ N) and slew-rate clamping, introducing nonlinear clipping.

    ### 4. Tyre Contact Dynamics
    - The tyre is modeled as a linear spring ($k_t = 190,000$ N/m). Under severe rebounds, estimated normal load can become negative, indicating loss of road contact.

    ### 5. Academic Prototype Disclaimer
    - **This software is an educational and simulation research prototype.**
    - No claims of vehicle safety, compliance with FMVSS/ECE automotive standards, or roadworthiness are made.
    """)
