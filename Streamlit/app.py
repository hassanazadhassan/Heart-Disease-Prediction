"""
Streamlit interface for the Heart Disease Risk Predictor.
All ML logic lives in helper.py - this file is UI only.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import helper

# --------------------------------------------------------------------------
# Shared Plotly styling — every chart uses this so the chart frame is always
# a clean white card instead of inheriting a dark theme around a white plot
# area (paper_bgcolor is the outer frame, plot_bgcolor is the inner grid —
# both need to be set, or you get a mismatched box like the earlier bug).
# --------------------------------------------------------------------------

def styled_layout(**overrides):
    base = dict(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#1f2937", family="sans-serif", size=13),
        margin=dict(l=10, r=20, t=20, b=10),
        transition=dict(duration=0),  # avoid a fade-in mid-transition looking "washed out"
    )
    base.update(overrides)
    return base


def styled_axis(**overrides):
    """Default axis config with solid, readable tick labels - merge extra
    options on top (e.g. gridcolor, range, title)."""
    title = overrides.pop("title", None)
    base = dict(
        tickfont=dict(color="#1f2937", size=12),
        color="#1f2937",
    )
    if title is not None:
        # Newer Plotly versions removed the old top-level `titlefont` axis
        # property in favor of nesting the font under `title`.
        base["title"] = dict(text=title, font=dict(color="#1f2937"))
    base.update(overrides)
    return base


CHART_CONFIG = {"displayModeBar": False}

# --------------------------------------------------------------------------
# Page config + global styling
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="\u2764\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}

    :root {
        --brand-1: #7f1d1d;
        --brand-2: #dc2626;
        --brand-3: #f97316;
        --ink: #1f2937;
        --muted: #6b7280;
        --card-bg: #ffffff;
        --page-bg: #f7f7fb;
    }

    .stApp {
        background: linear-gradient(180deg, #fff5f5 0%, var(--page-bg) 320px);
    }

    .hero {
        background: linear-gradient(120deg, var(--brand-1), var(--brand-2) 55%, var(--brand-3));
        border-radius: 20px;
        padding: 2.1rem 2.4rem;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 30px rgba(127, 29, 29, 0.25);
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 1.02rem;
        max-width: 700px;
    }
    .mode-pill {
        display: inline-block;
        margin-top: 0.9rem;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        font-size: 0.85rem;
        font-weight: 600;
        backdrop-filter: blur(4px);
    }

    .card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 2px 14px rgba(17, 24, 39, 0.06);
        border: 1px solid rgba(17, 24, 39, 0.05);
        margin-bottom: 1.1rem;
    }
    .card h3 {
        margin-top: 0;
        color: var(--ink);
    }

    .section-title {
        font-weight: 700;
        color: var(--ink);
        font-size: 1.05rem;
        margin: 0.2rem 0 0.7rem 0;
        border-left: 4px solid var(--brand-2);
        padding-left: 0.6rem;
    }

    .result-banner {
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-banner.safe { background: linear-gradient(120deg, #16a34a, #22c55e); }
    .result-banner.warn { background: linear-gradient(120deg, #d97706, #f59e0b); }
    .result-banner.danger { background: linear-gradient(120deg, #b91c1c, #ef4444); }
    .result-banner .big { font-size: 1.7rem; font-weight: 800; margin-bottom: 0.15rem; }
    .result-banner .small { font-size: 0.95rem; opacity: 0.92; }

    .metric-card {
        background: var(--card-bg);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(17,24,39,0.06);
        box-shadow: 0 2px 10px rgba(17,24,39,0.05);
    }
    .metric-card .label { color: var(--muted); font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em;}
    .metric-card .value { color: var(--ink); font-size: 1.6rem; font-weight: 800; }
    .metric-card .value.best { color: var(--brand-2); }

    .footnote { color: var(--muted); font-size: 0.85rem; }

    div[data-testid="stForm"] {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        border: 1px solid rgba(17,24,39,0.05);
        box-shadow: 0 2px 14px rgba(17,24,39,0.06);
    }

    .stButton>button, .stFormSubmitButton>button {
        background: linear-gradient(120deg, var(--brand-1), var(--brand-2));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        opacity: 0.92;
    }

    /* Force widget labels/captions to a readable dark color no matter the
       active theme (dark-mode browsers/OS were rendering these invisible
       white-on-white against our light card background). */
    div[data-testid="stForm"] label p,
    div[data-testid="stForm"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stForm"] .stCaption,
    div[data-testid="stForm"] small,
    div[data-testid="stForm"] [data-testid="stCaptionContainer"] p {
        color: var(--ink) !important;
        opacity: 1 !important;
    }

    /* Number inputs: force a white box with dark text (previously the text
       color was forced dark but the box itself stayed dark -> invisible). */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        caret-color: var(--ink) !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #f3f4f6 !important;
        color: var(--ink) !important;
        border-color: #d1d5db !important;
    }

    /* Select boxes: the closed box */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] * {
        color: var(--ink) !important;
    }

    /* Select boxes: the dropdown list popup renders in a portal outside the
       form, so it needs its own (unscoped) rule. */
    ul[role="listbox"],
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    ul[role="listbox"] li,
    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] * {
        background-color: #ffffff !important;
        color: var(--ink) !important;
    }
    ul[role="listbox"] li:hover,
    div[data-baseweb="popover"] li:hover {
        background-color: #fee2e2 !important;
    }

    /* Plotly charts: force every bit of text (axis ticks, titles, bar labels)
       to be fully solid - Plotly's default tick-label fade-in transition can
       otherwise get caught mid-render and look washed out/half-invisible. */
    .js-plotly-plot .plotly text {
        fill: #1f2937 !important;
        fill-opacity: 1 !important;
        opacity: 1 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### \u2764\ufe0f Heart Disease Risk")
    st.caption("Machine learning demo built on the UCI Cleveland Heart Disease dataset.")

    st.markdown("---")
    st.markdown("**Prediction mode**")
    mode_label = st.radio(
        "Choose what the model should predict:",
        options=["Binary (No Disease vs. Disease)", "Multiclass (Severity 0\u20134)"],
        index=0,
        label_visibility="collapsed",
    )
    model_type = "binary" if mode_label.startswith("Binary") else "multiclass"

    st.markdown("---")
    page = st.radio(
        "Section",
        options=["\U0001F52C Predict", "\U0001F4CA Model Performance", "\u2139\ufe0f About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Model status**")
    for key, info in helper.model_status().items():
        if info["connected"]:
            st.markdown(f"\u2705 {info['label']} connected")
        else:
            st.markdown(f"\u274c {info['label']} not found")
    with st.expander("Details"):
        for key, info in helper.model_status().items():
            st.caption(f"{info['label']}: `{info['path']}`")

    st.markdown("---")
    st.caption(
        "This tool is for educational purposes only and is **not** a substitute "
        "for professional medical advice."
    )

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------

mode_pill_text = "Binary mode \u00b7 No Disease vs. Disease" if model_type == "binary" else "Multiclass mode \u00b7 Severity levels 0\u20134"

st.markdown(
    f"""
    <div class="hero">
        <h1>Heart Disease Risk Predictor</h1>
        <p>Enter a patient's clinical measurements to estimate heart disease risk using
        models trained on the UCI Cleveland Heart Disease dataset.</p>
        <span class="mode-pill">{mode_pill_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# PAGE: PREDICT
# ==========================================================================

if page.endswith("Predict"):

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

        with st.form("prediction_form"):
            defaults = helper.default_form_values()
            values = {}

            c1, c2 = st.columns(2)
            num_items = list(helper.NUMERIC_FIELDS.items())
            for i, (key, field) in enumerate(num_items):
                target_col = c1 if i % 2 == 0 else c2
                with target_col:
                    if field.step == int(field.step):
                        values[key] = st.number_input(
                            field.label, min_value=int(field.min_value), max_value=int(field.max_value),
                            value=int(field.default), step=int(field.step), help=field.help,
                        )
                    else:
                        values[key] = st.number_input(
                            field.label, min_value=float(field.min_value), max_value=float(field.max_value),
                            value=float(field.default), step=float(field.step), help=field.help, format="%.1f",
                        )
                    if field.normal_range:
                        st.caption(field.normal_range)

            st.markdown("&nbsp;", unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            cat_items = list(helper.CATEGORICAL_FIELDS.items())
            for i, (key, field) in enumerate(cat_items):
                target_col = c3 if i % 2 == 0 else c4
                with target_col:
                    option_values = list(field.options.keys())
                    option_labels = [field.options[v] for v in option_values]
                    default_idx = option_values.index(field.default)
                    chosen_label = st.selectbox(field.label, options=option_labels, index=default_idx, help=field.help)
                    values[key] = option_values[option_labels.index(chosen_label)]

            submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

        if not submitted:
            st.markdown(
                """
                <div class="card">
                    <p style="color:#6b7280; margin-bottom:0;">
                    Fill in the patient's information on the left and click
                    <b>Predict Risk</b> to see the model's estimate here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            try:
                model = helper.load_model(model_type)
            except helper.ModelNotFoundError as e:
                st.error(str(e))
                st.info(
                    f"Expected path for **{model_type}**: `{helper.MODEL_PATHS[model_type]}`\n\n"
                    "Check that the filename in `helper.MODEL_PATHS` matches your `.pkl` "
                    "file exactly (case-sensitive, including `.pkl`), and that the file "
                    "sits inside the `models/` subfolder next to `app.py` — not the "
                    "project root. See **README.md** for the full steps."
                )
            else:
                result = helper.predict(model_type, model, values)

                if model_type == "binary":
                    banner_class = "safe" if result["prediction"] == 0 else "danger"
                else:
                    banner_class = ["safe", "warn", "warn", "danger", "danger"][result["prediction"]]

                st.markdown(
                    f"""
                    <div class="result-banner {banner_class}">
                        <div class="big">{result['label']}</div>
                        <div class="small">Model: {mode_label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if result["probabilities"]:
                    proba_df = pd.DataFrame(
                        {"Class": list(result["probabilities"].keys()),
                         "Probability": list(result["probabilities"].values())}
                    ).sort_values("Probability", ascending=True)

                    fig = go.Figure(go.Bar(
                        x=proba_df["Probability"],
                        y=proba_df["Class"],
                        orientation="h",
                        marker=dict(color=proba_df["Probability"], colorscale="Reds"),
                        text=[f"{p:.1%}" for p in proba_df["Probability"]],
                        textposition="outside",
                        textfont=dict(color="#1f2937"),
                    ))
                    fig.update_layout(**styled_layout(
                        height=280,
                        margin=dict(l=10, r=45, t=10, b=10),
                        xaxis=styled_axis(range=[0, 1], tickformat=".0%", title="Predicted probability",
                                          gridcolor="#f3f4f6", zeroline=False),
                        yaxis=styled_axis(title=""),
                    ))
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
                    st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("View submitted patient data"):
                    st.dataframe(result["input"], use_container_width=True, hide_index=True)

# ==========================================================================
# PAGE: MODEL PERFORMANCE
# ==========================================================================

elif page.endswith("Model Performance"):

    metrics = helper.MODEL_METRICS[model_type]
    class_report = helper.CLASS_REPORTS[model_type]
    top_features = helper.TOP_FEATURES[model_type]

    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    best_model = max(metrics.items(), key=lambda kv: kv[1]["accuracy"])[0]

    cols = st.columns(len(metrics))
    for col, (name, m) in zip(cols, metrics.items()):
        is_best = name == best_model
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{name}</div>
                    <div class="value {'best' if is_best else ''}">{m['accuracy']:.1%}</div>
                    <div class="footnote">Macro F1: {m['macro_f1']:.3f} &middot; {m['params']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    comp_df = pd.DataFrame(
        [{"Model": name, "Accuracy": m["accuracy"], "Macro F1": m["macro_f1"], "Weighted F1": m["weighted_f1"]}
         for name, m in metrics.items()]
    )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Accuracy", x=comp_df["Model"], y=comp_df["Accuracy"], marker_color="#dc2626"))
        fig.add_trace(go.Bar(name="Weighted F1", x=comp_df["Model"], y=comp_df["Weighted F1"], marker_color="#f97316"))
        fig.update_layout(**styled_layout(
            barmode="group",
            height=380,
            yaxis=styled_axis(tickformat=".0%", title="Score", range=[0, 1], gridcolor="#f3f4f6", zeroline=False),
            xaxis=styled_axis(gridcolor="#ffffff"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#1f2937")),
            margin=dict(l=10, r=10, t=30, b=10),
        ))
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><h3>Comparison Table</h3>', unsafe_allow_html=True)
        display_df = comp_df.copy()
        display_df["Accuracy"] = display_df["Accuracy"].map(lambda v: f"{v:.2%}")
        display_df["Macro F1"] = display_df["Macro F1"].map(lambda v: f"{v:.3f}")
        display_df["Weighted F1"] = display_df["Weighted F1"].map(lambda v: f"{v:.3f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Class-Level Performance (Best Model)</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(class_report["rows"])
    report_df.columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Top Predictive Features</div>', unsafe_allow_html=True)
    feat_cols = st.columns(len(top_features))
    for col, (name, feats) in zip(feat_cols, top_features.items()):
        with col:
            st.markdown(f'<div class="card"><b>{name}</b>', unsafe_allow_html=True)
            feat_df = pd.DataFrame(feats, columns=["Feature", "Importance"])
            fig = go.Figure(go.Bar(
                x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
                marker_color="#7f1d1d",
            ))
            fig.update_layout(**styled_layout(
                height=230,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=styled_axis(autorange="reversed", title=""),
                xaxis=styled_axis(title="", gridcolor="#f3f4f6", zeroline=False),
            ))
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
            st.markdown('</div>', unsafe_allow_html=True)

    if model_type == "multiclass":
        st.info(
            "Severity classes 2-4 have very few examples in the dataset (7, 7 and 3 "
            "patients respectively), which is why per-class performance is much lower "
            "than the binary model. This is an expected effect of class imbalance, not "
            "necessarily a modeling mistake."
        )

# ==========================================================================
# PAGE: ABOUT
# ==========================================================================

else:
    st.markdown(
        """
        <div class="card">
            <h3>About this project</h3>
            <p>
            This app predicts heart disease risk from patient clinical measurements,
            using models trained on the
            <a href="https://archive.ics.uci.edu/dataset/45/heart+disease" target="_blank">
            UCI Cleveland Heart Disease dataset</a> (303 patients, 13 clinical features).
            </p>
            <p>
            Two prediction targets are supported, switchable from the sidebar:
            </p>
            <ul>
                <li><b>Binary</b> &mdash; No Disease (0) vs. Disease Present (1&ndash;4 collapsed to 1)</li>
                <li><b>Multiclass</b> &mdash; Severity levels 0 (none) through 4 (most severe)</li>
            </ul>
            <p>
            Both pipelines apply the same preprocessing (standard scaling for numeric
            features, one-hot encoding for nominal categorical features) and SMOTE
            oversampling to address class imbalance, before being trained and tuned via
            grid search across Logistic Regression, KNN, Decision Tree, Random Forest,
            and XGBoost. Logistic Regression was selected as the final model for both
            tasks after hyperparameter tuning.
            </p>
        </div>
        <div class="card">
            <h3>Project structure</h3>
            <ul>
                <li><code>app.py</code> &mdash; Streamlit interface (this file's sibling)</li>
                <li><code>helper.py</code> &mdash; feature schema, preprocessing, model loading, static metrics</li>
                <li><code>models/binary_model.pkl</code> &mdash; trained binary pipeline</li>
                <li><code>models/multiclass_model.pkl</code> &mdash; trained multiclass pipeline</li>
            </ul>
        </div>
        <div class="card">
            <h3>Disclaimer</h3>
            <p style="margin-bottom:0;">
            This tool is a machine learning demo for educational/portfolio purposes.
            It is <b>not</b> a medical device and should never be used for real clinical
            decision-making. Always consult a qualified healthcare professional.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
