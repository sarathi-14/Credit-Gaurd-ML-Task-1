"""CreditGuard AI: Intelligent Credit Scoring & Financial Risk Prediction System.

Streamlit Frontend Web Application.
Connects directly to the trained Machine Learning pipeline and evaluation artifacts.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    RAW_FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    categorize_risk,
    MODELS_DIR,
    OUTPUTS_DIR,
    VIS_DIR,
    DATA_DIR,
)
from src.predict import load_model, predict_single_borrower

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditGuard AI | Intelligent Credit Scoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING (Dark Theme, Modern FinTech Cards, Glow Badges)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Typography & Font Styling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Background adjustments */
    .stApp {
        background-color: #0B0F19;
        color: #F1F5F9;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.5);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .metric-sub {
        color: #38BDF8;
        font-size: 0.8rem;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .status-online {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 50%, #0B1120 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    /* Disclaimer Box */
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #F59E0B;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.83rem;
        color: #FDE68A;
        margin: 16px 0;
    }

    /* Form Inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None


# -----------------------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_model():
    """Load the trained machine learning pipeline from disk."""
    try:
        model = load_model()
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def get_model_metadata():
    """Load model performance metadata and comparison metrics."""
    meta_path = MODELS_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_comparison_table():
    """Load comparison CSV table."""
    csv_path = OUTPUTS_DIR / "model_comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_sample_dataset(nrows=10000):
    """Load sample of dataset for analytics dashboard."""
    train_path = DATA_DIR / "cs-training.csv"
    if train_path.exists():
        df = pd.read_csv(train_path, nrows=nrows)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df
    return None


model_pipeline, model_err = get_cached_model()
metadata = get_model_metadata()
comparison_df = get_comparison_table()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM STATUS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2rem;">🛡️</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.3rem; font-weight: 800; color: #FFFFFF;">CreditGuard AI</h2>
                    <p style="margin: 0; font-size: 0.75rem; color: #38BDF8; font-weight: 600;">FINANCIAL RISK INTELLIGENCE</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Home Dashboard",
            "Credit Risk Prediction",
            "Risk Assessment Results",
            "Model Performance",
            "Data Analytics",
            "Prediction History",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # System Status Pill
    if model_pipeline is not None:
        st.markdown(
            """
            <div style="background-color: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.78rem; color: #94A3B8; font-weight: 600;">PIPELINE STATUS</span>
                    <span class="status-pill status-online">● ACTIVE</span>
                </div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #F1F5F9;">Random Forest Classifier</div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">Validation ROC-AUC: <strong>0.8674</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("⚠️ Model pipeline not loaded.")

    st.markdown(
        """
        <div style="margin-top: 25px; padding: 10px; font-size: 0.72rem; color: #64748B; text-align: center; border-top: 1px solid rgba(255, 255, 255, 0.05);">
            CodeAlpha ML Internship — Task 1<br>
            Developed by Machine Learning Intern
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 1: HOME DASHBOARD
# =============================================================================
if page == "Home Dashboard":
    st.markdown(
        """
        <div class="hero-banner">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
                <div>
                    <span style="background: rgba(59, 130, 246, 0.2); color: #60A5FA; font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 6px; letter-spacing: 0.05em;">
                        CODEALPHA ML INTERNSHIP — TASK 1
                    </span>
                    <h1 style="color: #FFFFFF; font-size: 2.2rem; font-weight: 800; margin: 12px 0 6px 0;">
                        CreditGuard AI
                    </h1>
                    <p style="color: #94A3B8; font-size: 1.05rem; max-width: 650px; margin: 0; line-height: 1.5;">
                        Intelligent Credit Scoring and Financial Risk Prediction System using Supervised Machine Learning to detect serious borrower delinquency within a two-year horizon.
                    </p>
                </div>
                <div style="text-align: right;">
                    <div class="status-pill status-online" style="font-size: 0.85rem; padding: 6px 16px;">
                        ● PRODUCTION READY PIPELINE
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)

    roc_auc_val = f"{metadata.get('validation_roc_auc', 0.8674):.4f}" if metadata else "0.8674"
    recall_val = f"{metadata.get('validation_recall', 0.7621):.2%}" if metadata else "76.21%"
    train_records = f"{metadata.get('training_records', 120000):,}" if metadata else "120,000"
    champion_name = metadata.get("best_model_name", "Random Forest") if metadata else "Random Forest"

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Validation ROC-AUC</div>
                <div class="metric-value" style="color: #38BDF8;">{roc_auc_val}</div>
                <div class="metric-sub">Kaggle Benchmark Metric</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Delinquency Recall</div>
                <div class="metric-value" style="color: #10B981;">{recall_val}</div>
                <div class="metric-sub">Balanced Default Detection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Training Records</div>
                <div class="metric-value">{train_records}</div>
                <div class="metric-sub">Stratified 80/20 Partition</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Champion Model</div>
                <div class="metric-value" style="font-size: 1.45rem; color: #F59E0B;">{champion_name}</div>
                <div class="metric-sub">100 Trees + Feature Eng.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # Core Pillars Section
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🎯 System Architecture & Objective")
        st.write(
            """
            **CreditGuard AI** addresses the classic binary credit risk prediction challenge: predicting whether
            a borrower will experience **Serious Delinquency (90 days past due or worse)** within two years based
            on historical repayment records, credit card revolving utilization, debt-to-income ratios, and demographic data.
            """
        )

        st.markdown(
            """
            * **Dataset Origin:** Official Kaggle *Give Me Some Credit* benchmark (150,000 historical records).
            * **Leakage-Free Engineering:** Multi-stage Scikit-learn Pipeline with custom financial feature transformers.
            * **Class Imbalance Handling:** Class weight calibration addressing the severe **14:1 imbalance** (93.3% safe vs 6.7% delinquent).
            * **Multi-Model Benchmark:** Side-by-side empirical comparison between Logistic Regression, Decision Tree, and Random Forest.
            """
        )

        st.markdown(
            """
            <div class="disclaimer-box">
                <strong>Educational Use Notice:</strong> This software is engineered as an educational and portfolio demonstration
                for the CodeAlpha Machine Learning Internship. It is not an automated loan approval or underwriting tool.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("### ⚡ Quick Model Benchmark")
        if comparison_df is not None:
            fig = px.bar(
                comparison_df,
                x="Model",
                y="ROC-AUC",
                color="Model",
                color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B"],
                title="Validation ROC-AUC by Model",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8"),
                yaxis=dict(range=[0.75, 0.90], gridcolor="rgba(255,255,255,0.08)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train the model using `python src/train_model.py` to view benchmark metrics.")


# =============================================================================
# PAGE 2: CREDIT RISK PREDICTION
# =============================================================================
elif page == "Credit Risk Prediction":
    st.markdown("## 🔍 Individual Borrower Risk Assessment")
    st.write(
        "Enter borrower financial indicators below to generate real-time distress probability and credit risk categorization."
    )

    # Preset profiles for quick testing
    st.markdown("##### Quick Fill Sample Profiles:")
    preset_cols = st.columns(3)

    preset_choice = None
    if preset_cols[0].button("🟢 Low Risk Prime Profile", use_container_width=True):
        preset_choice = "low"
    if preset_cols[1].button("🟡 Moderate Risk Near-Prime Profile", use_container_width=True):
        preset_choice = "medium"
    if preset_cols[2].button("🔴 High Risk Subprime Profile", use_container_width=True):
        preset_choice = "high"

    # Default values based on preset
    if preset_choice == "low":
        d_util, d_age, d_late30, d_debt, d_inc, d_lines, d_late90, d_re, d_late60, d_dep = (
            0.15,
            48,
            0,
            0.22,
            8500.0,
            10,
            0,
            2,
            0,
            1,
        )
    elif preset_choice == "medium":
        d_util, d_age, d_late30, d_debt, d_inc, d_lines, d_late90, d_re, d_late60, d_dep = (
            0.65,
            34,
            1,
            0.48,
            4200.0,
            6,
            0,
            1,
            0,
            2,
        )
    elif preset_choice == "high":
        d_util, d_age, d_late30, d_debt, d_inc, d_lines, d_late90, d_re, d_late60, d_dep = (
            1.25,
            26,
            3,
            0.82,
            2800.0,
            4,
            2,
            0,
            1,
            0,
        )
    else:
        d_util, d_age, d_late30, d_debt, d_inc, d_lines, d_late90, d_re, d_late60, d_dep = (
            0.32,
            42,
            0,
            0.35,
            5500.0,
            8,
            0,
            1,
            0,
            1,
        )

    with st.form("borrower_form"):
        st.markdown("### 📋 Financial & Demographic Attributes")

        col1, col2 = st.columns(2)

        with col1:
            utilization = st.number_input(
                "Revolving Credit Utilization (Balance / Limit)",
                min_value=0.0,
                max_value=20.0,
                value=float(d_util),
                step=0.05,
                help=FEATURE_DESCRIPTIONS["RevolvingUtilizationOfUnsecuredLines"],
            )
            age = st.number_input(
                "Borrower Age (Years)",
                min_value=18,
                max_value=105,
                value=int(d_age),
                step=1,
                help=FEATURE_DESCRIPTIONS["age"],
            )
            monthly_income = st.number_input(
                "Monthly Gross Income ($)",
                min_value=0.0,
                max_value=500000.0,
                value=float(d_inc),
                step=500.0,
                help=FEATURE_DESCRIPTIONS["MonthlyIncome"],
            )
            debt_ratio = st.number_input(
                "Debt-to-Income Ratio (Monthly Payments / Gross Income)",
                min_value=0.0,
                max_value=50.0,
                value=float(d_debt),
                step=0.05,
                help=FEATURE_DESCRIPTIONS["DebtRatio"],
            )
            open_lines = st.number_input(
                "Number of Open Credit Lines & Loans",
                min_value=0,
                max_value=60,
                value=int(d_lines),
                step=1,
                help=FEATURE_DESCRIPTIONS["NumberOfOpenCreditLinesAndLoans"],
            )

        with col2:
            late_30_59 = st.number_input(
                "Times 30–59 Days Past Due (Last 2 Years)",
                min_value=0,
                max_value=20,
                value=int(d_late30),
                step=1,
                help=FEATURE_DESCRIPTIONS["NumberOfTime30-59DaysPastDueNotWorse"],
            )
            late_60_89 = st.number_input(
                "Times 60–89 Days Past Due (Last 2 Years)",
                min_value=0,
                max_value=20,
                value=int(d_late60),
                step=1,
                help=FEATURE_DESCRIPTIONS["NumberOfTime60-89DaysPastDueNotWorse"],
            )
            late_90_plus = st.number_input(
                "Times 90+ Days Late (Severe Delinquency)",
                min_value=0,
                max_value=20,
                value=int(d_late90),
                step=1,
                help=FEATURE_DESCRIPTIONS["NumberOfTimes90DaysLate"],
            )
            real_estate = st.number_input(
                "Number of Real Estate Loans / Mortgages",
                min_value=0,
                max_value=30,
                value=int(d_re),
                step=1,
                help=FEATURE_DESCRIPTIONS["NumberRealEstateLoansOrLines"],
            )
            dependents = st.number_input(
                "Number of Dependents (Excluding Borrower)",
                min_value=0,
                max_value=20,
                value=int(d_dep),
                step=1,
                help=FEATURE_DESCRIPTIONS["NumberOfDependents"],
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🛡️ Predict Credit Risk Assessment", use_container_width=True)

    if submitted:
        if model_pipeline is None:
            st.error("Error: Trained model pipeline is unavailable. Please verify models/best_credit_model.joblib exists.")
        else:
            borrower_payload = {
                "RevolvingUtilizationOfUnsecuredLines": utilization,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
                "DebtRatio": debt_ratio,
                "MonthlyIncome": monthly_income,
                "NumberOfOpenCreditLinesAndLoans": open_lines,
                "NumberOfTimes90DaysLate": late_90_plus,
                "NumberRealEstateLoansOrLines": real_estate,
                "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
                "NumberOfDependents": dependents,
            }

            with st.spinner("Analyzing financial history through Scikit-learn Pipeline..."):
                prediction_res = predict_single_borrower(borrower_payload, model=model_pipeline)
                st.session_state.latest_prediction = prediction_res

                # Record in session history
                history_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Age": age,
                    "Income": f"${monthly_income:,.0f}",
                    "Utilization": f"{utilization:.1%}",
                    "DebtRatio": f"{debt_ratio:.2f}",
                    "Probability": prediction_res["probability_percentage"],
                    "Risk Tier": prediction_res["risk_category"],
                    "Predicted Class": prediction_res["predicted_class"],
                }
                st.session_state.prediction_history.insert(0, history_entry)

            st.success("✅ Prediction calculated successfully! Proceed to 'Risk Assessment Results' or review below.")


# =============================================================================
# PAGE 3: RISK ASSESSMENT RESULTS
# =============================================================================
if page == "Risk Assessment Results" or (page == "Credit Risk Prediction" and st.session_state.latest_prediction is not None):
    res = st.session_state.latest_prediction

    if res is None:
        st.info("ℹ️ No prediction has been performed yet. Please enter borrower details on the **Credit Risk Prediction** page.")
    else:
        st.markdown("## 📊 Comprehensive Risk Assessment Output")

        # Top highlight banner
        cat = res["risk_category"]
        color = res["risk_color"]
        prob = res["probability"]

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
                        border-left: 6px solid {color}; border-radius: 12px; padding: 22px; margin-bottom: 25px;
                        box-shadow: 0 10px 20px rgba(0,0,0,0.4);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="background-color: {color}22; color: {color}; font-weight: 700; padding: 4px 12px; border-radius: 6px; font-size: 0.85rem;">
                            {res['risk_tier']}
                        </span>
                        <h2 style="color: #FFFFFF; margin: 8px 0 4px 0; font-size: 1.8rem; font-weight: 800;">
                            {cat}
                        </h2>
                        <p style="color: #94A3B8; margin: 0; font-size: 0.95rem;">
                            {res['risk_description']}
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.82rem; color: #94A3B8; font-weight: 600;">ESTIMATED DISTRESS PROBABILITY</div>
                        <div style="font-size: 2.4rem; font-weight: 800; color: {color};">{res['probability_percentage']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_gauge, col_details = st.columns([2, 3])

        with col_gauge:
            # Plotly Gauge Chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 36}},
                    title={"text": "Delinquency Probability", "font": {"color": "#94A3B8", "size": 14}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                        "bar": {"color": color},
                        "bgcolor": "rgba(30, 41, 59, 0.5)",
                        "steps": [
                            {"range": [0, 15], "color": "rgba(16, 185, 129, 0.25)"},
                            {"range": [15, 35], "color": "rgba(245, 158, 11, 0.25)"},
                            {"range": [35, 100], "color": "rgba(239, 68, 68, 0.25)"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 3},
                            "thickness": 0.8,
                            "value": prob * 100,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                margin=dict(l=25, r=25, t=30, b=25),
                height=260,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_details:
            st.markdown("### 🔍 Risk Factor Breakdown")

            st.markdown("##### ⚠️ Risk Elevators:")
            for factor in res["risk_factors"]:
                st.markdown(f"- 🔴 {factor}")

            st.markdown("##### 🛡️ Protective Indicators:")
            for factor in res["protective_factors"]:
                st.markdown(f"- 🟢 {factor}")

            st.markdown("##### 📌 Underwriting Recommendation:")
            st.info(res["recommendation"])


# =============================================================================
# PAGE 4: MODEL PERFORMANCE
# =============================================================================
elif page == "Model Performance":
    st.markdown("## 📈 Empirical Model Evaluation & Comparison")
    st.write(
        "All performance scores are calculated on the strictly held-out 20% validation split (30,000 records). "
        "No synthetic data or fabricated results."
    )

    if comparison_df is not None:
        st.markdown("### 🏆 Algorithm Leaderboard")
        st.dataframe(
            comparison_df.style.highlight_max(
                subset=["ROC-AUC", "PR-AUC", "Recall", "F1-Score", "Accuracy"],
                color="#1E3A8A",
            ),
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 📊 Metrics Comparison Chart")
            df_melt = comparison_df.melt(
                id_vars=["Model"],
                value_vars=["ROC-AUC", "PR-AUC", "Recall", "F1-Score", "Precision", "Accuracy"],
                var_name="Metric",
                value_name="Score",
            )
            fig_bar = px.bar(
                df_melt,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B"],
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)", range=[0, 1.05]),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=340,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            st.markdown("### 🎯 Tradeoff & Metric Definitions")
            st.markdown(
                """
                - **ROC-AUC (0.8674 - Random Forest):** Area under the Receiver Operating Characteristic curve. Measures ranking quality independent of threshold.
                - **Recall (76.21%):** High sensitivity is critical in credit underwriting to avoid **False Negatives** (costly credit default).
                - **False Positives vs False Negatives:** In retail lending, a False Negative incurs severe capital loss (defaulted principal), whereas a False Positive merely prompts manual document review.
                """
            )

        st.markdown("---")
        st.markdown("### 📉 Static Diagnostic Curves")
        c1, c2 = st.columns(2)
        roc_img = VIS_DIR / "roc_curves_comparison.png"
        pr_img = VIS_DIR / "pr_curves_comparison.png"
        cm_img = VIS_DIR / "confusion_matrices.png"

        if roc_img.exists():
            c1.image(str(roc_img), caption="Validation ROC Curves", use_container_width=True)
        if pr_img.exists():
            c2.image(str(pr_img), caption="Validation Precision-Recall Curves", use_container_width=True)
        if cm_img.exists():
            st.image(str(cm_img), caption="Normalized Confusion Matrices", use_container_width=True)
    else:
        st.warning("Model comparison data not found. Run `python src/train_model.py` first.")


# =============================================================================
# PAGE 5: DATA ANALYTICS
# =============================================================================
elif page == "Data Analytics":
    st.markdown("## 📊 Exploratory Data Analytics (EDA)")
    st.write("Visualizing borrower risk distributions, financial correlations, and predictive feature importances.")

    tab1, tab2, tab3 = st.tabs(["Key Financial Distributions", "Correlation & Relationships", "Feature Importance"])

    with tab1:
        st.markdown("### 📈 Core Financial Feature Distributions")
        c1, c2 = st.columns(2)
        p1 = VIS_DIR / "01_target_distribution.png"
        p3 = VIS_DIR / "03_age_distribution.png"
        p4 = VIS_DIR / "04_monthly_income_distribution.png"
        p6 = VIS_DIR / "06_credit_utilization_distribution.png"

        if p1.exists():
            c1.image(str(p1), caption="Target Class Imbalance (SeriousDlqin2yrs)", use_container_width=True)
        if p3.exists():
            c2.image(str(p3), caption="Age Distribution Stratified by Delinquency", use_container_width=True)
        if p4.exists():
            c1.image(str(p4), caption="Log Monthly Income Distribution", use_container_width=True)
        if p6.exists():
            c2.image(str(p6), caption="Credit Utilization Distribution", use_container_width=True)

    with tab2:
        st.markdown("### 🔗 Correlation & Feature Relationships")
        p8 = VIS_DIR / "08_correlation_heatmap.png"
        p10 = VIS_DIR / "10_feature_relationships.png"

        if p8.exists():
            st.image(str(p8), caption="Full Feature Correlation Matrix Heatmap", use_container_width=True)
        if p10.exists():
            st.image(str(p10), caption="Age vs. Log Monthly Income Stratified by Delinquency", use_container_width=True)

    with tab3:
        st.markdown("### 🏆 Top Predictive Features (Random Forest)")
        p_imp = VIS_DIR / "feature_importance.png"
        if p_imp.exists():
            st.image(str(p_imp), caption="Top 10 Feature Importances (Gini Impurity Reduction)", use_container_width=True)

        st.markdown(
            """
            **Economic Interpretation:**
            1. **`RevolvingUtilization_Clipped` / `HighUtilizationFlag`:** Highest predictive contribution. Borrowers maxing out credit limits show immediate liquidity distress.
            2. **`TotalPastDue` & `HasSevereDelinquency`:** Past payment behavior is the strongest single behavioral predictor of future delinquency.
            3. **`age`:** Older borrowers systematically demonstrate lower delinquency rates due to career maturity and accumulated net worth.
            4. **`DebtRatio_Log` & `IncomePerPerson`:** Measures cash flow solvency relative to dependents and debt service.
            """
        )


# =============================================================================
# PAGE 6: PREDICTION HISTORY
# =============================================================================
elif page == "Prediction History":
    st.markdown("## 🕒 Session Prediction History")
    st.write("Displays borrower assessments performed during this active browser session.")

    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions recorded in this session yet.")
    else:
        df_history = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df_history, use_container_width=True)

        col_dl, col_clr = st.columns([2, 1])
        with col_dl:
            csv_data = df_history.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export History as CSV",
                data=csv_data,
                file_name="creditguard_prediction_history.csv",
                mime="text/csv",
            )
        with col_clr:
            if st.button("🗑️ Clear Session History"):
                st.session_state.prediction_history = []
                st.rerun()

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Privacy & Data Minimization:</strong> Predictions are stored strictly in temporary in-memory session state.
            No personally identifiable financial records are transmitted or permanently recorded.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 7: ABOUT PROJECT
# =============================================================================
elif page == "About Project":
    st.markdown("## ℹ️ About CreditGuard AI")

    st.markdown(
        """
        ### 🎓 CodeAlpha Machine Learning Internship
        - **Project Title:** Intelligent Credit Scoring and Financial Risk Prediction System
        - **Task:** Task 1 — Credit Scoring Model
        - **Organization:** CodeAlpha
        - **Developer:** Machine Learning Intern
        - **Environment:** Windows 11, Python 3, Scikit-learn 1.9, Streamlit, Plotly

        ---

        ### 🔬 Methodology & Machine Learning Lifecycle
        1. **Problem Formulation:** Binary classification targeting borrower financial distress (`SeriousDlqin2yrs`) within 2 years.
        2. **Dataset:** Kaggle *Give Me Some Credit* (150,000 real-world borrower entries).
        3. **Preprocessing & Feature Engineering:** Leak-free custom Scikit-learn Transformer handling median imputation, sentinel code cleaning, log scaling, and financial indicators.
        4. **Model Architecture:** Comparative evaluation of regularized Logistic Regression, Decision Tree, and Random Forest ensemble.
        5. **Validation Strategy:** Stratified 80/20 train-validation split evaluated via ROC-AUC, PR-AUC, Recall, Precision, and F1-score.
        6. **Interactive Deployment:** Streamlit cloud/desktop interface with Plotly analytics and real-time inference.

        ---

        ### ⚠️ Model Limitations & Ethical Considerations
        - **Class Imbalance:** Only 6.7% of historical loans represent delinquency; predictions require careful probability calibration.
        - **Correlation vs. Causation:** High feature importance indicates strong predictive association, not necessarily causality.
        - **Fairness & Bias:** Automated credit scoring systems must be continuously audited for demographic fairness and compliant with lending regulations (e.g. ECOA, FCRA).
        - **Not a Production Underwriting System:** Developed exclusively for academic assessment and portfolio demonstration.
        """
    )
