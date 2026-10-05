import html
import pandas as pd
import plotly.express as px
import streamlit as st
from config import (
    COLOR_ALERT,
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_CARD_TEXT,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    COLOR_SURFACE,
    COLOR_SURFACE_LIGHT,
    COLOR_SURFACE_ALT,
    COLOR_TEXT,
    COLOR_TEXT_DARK,
    COLOR_WHITE,
    COLOR_TEXT_MUTED,
    CHART_PALETTE,
)
def _chart_style(fig):
    fig.update_layout(
        font={"family": "Arial", "color": COLOR_TEXT},
        paper_bgcolor=COLOR_BACKGROUND,
        plot_bgcolor=COLOR_BACKGROUND,
        margin={"l": 45, "r": 25, "t": 30, "b": 45},
        hoverlabel={
            "bgcolor": COLOR_BACKGROUND,
            "font": {"color": COLOR_TEXT, "family": "Arial"},
            "bordercolor": COLOR_BORDER,
        },
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont={"color": COLOR_TEXT_MUTED, "family": "Arial"},
        title_font={"color": COLOR_TEXT_MUTED, "family": "Arial"},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLOR_BORDER,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont={"color": COLOR_TEXT_MUTED, "family": "Arial"},
        title_font={"color": COLOR_TEXT_MUTED, "family": "Arial"},
    )
    return fig
def _kpi(title, value, description, accent):
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="height:3px;background:{accent};margin:-1rem -1rem 0 -1rem;
                        border-radius:8px 8px 0 0;"></div>
            <div style="padding-top:8px;">
                <div style="font-size:11px;font-weight:700;letter-spacing:.08em;
                            color:{COLOR_TEXT_MUTED};">{title}</div>
                <div style="font-size:25px;font-weight:700;line-height:1.25;
                            color:{COLOR_TEXT};margin-top:5px;">{value}</div>
                <div style="font-size:11px;color:{COLOR_TEXT_MUTED};margin-top:5px;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
def _table(df):
    if df.empty:
        st.info("No diagnostic data available.")
        return
    columns = list(df.columns)
    header_cells = "".join(
        f'<th style="background:{COLOR_CARD};color:{COLOR_CARD_TEXT};'
        f'border-right:1px solid {COLOR_BORDER};border-bottom:1px solid {COLOR_BORDER};'
        f'padding:9px 10px;font-size:12px;font-weight:700;text-align:left;'
        f'white-space:nowrap;">{html.escape(str(column))}</th>'
        for column in columns
    )
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for index, column in enumerate(columns):
            value = row[column]
            if pd.isna(value):
                display_value = ""
            else:
                display_value = str(value)
            alignment = "left" if index == 0 else "right"
            cells.append(
                f'<td style="background:{COLOR_SURFACE_LIGHT};color:{COLOR_TEXT};'
                f'border-right:1px solid {COLOR_BORDER};border-bottom:1px solid {COLOR_BORDER};'
                f'padding:8px 10px;font-size:12px;text-align:{alignment};'
                f'white-space:nowrap;">{html.escape(display_value)}</td>'
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    table_html = f"""
    <div style="width:100%;overflow-x:auto;border:1px solid {COLOR_BORDER};
                border-radius:8px;background:{COLOR_SURFACE_LIGHT};">
        <table style="width:max-content;min-width:100%;border-collapse:separate;
                      border-spacing:0;overflow:hidden;">
            <thead>
                <tr>{header_cells}</tr>
            </thead>
            <tbody>
                {"".join(body_rows)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
@st.cache_data(show_spinner=False)
def _build_custom_evaluation(customers, transactions, campaigns):
    """Build a leakage-safe historical evaluation cohort for custom data."""
    from src.analytics.custom_model_evaluation import build_custom_model_evaluation
    return build_custom_model_evaluation(
        customers,
        transactions,
        campaigns,
    )
def _prepare_evaluation(customer_data, data_mode):
    """Prepare the evaluation cohort for the active dataset."""
    if data_mode == "Custom Data":
        custom_data = st.session_state.get("custom_data")
        if not isinstance(custom_data, dict):
            return pd.DataFrame()
        required_custom_data = {
            "customers",
            "transactions",
            "campaigns",
        }
        if not required_custom_data.issubset(custom_data):
            return pd.DataFrame()
        return _build_custom_evaluation(
            custom_data["customers"],
            custom_data["transactions"],
            custom_data["campaigns"],
        )
    required = {
        "customer_id",
        "observation_date",
        "churn",
        "churn_probability",
        "customer_segment",
    }
    missing = required - set(customer_data.columns)
    if missing:
        return pd.DataFrame()
    evaluation = customer_data.copy()
    evaluation["customer_id"] = evaluation["customer_id"].astype(str).str.strip()
    evaluation["observation_date"] = pd.to_datetime(
        evaluation["observation_date"], errors="coerce", dayfirst=True
    )
    evaluation["churn"] = pd.to_numeric(evaluation["churn"], errors="coerce")
    evaluation["churn_probability"] = pd.to_numeric(
        evaluation["churn_probability"], errors="coerce"
    )
    evaluation = evaluation.dropna(
        subset=["observation_date", "churn", "churn_probability", "customer_segment"]
    ).copy()
    evaluation["churn"] = evaluation["churn"].astype(int)
    return evaluation
def _run_threshold_backend(y_true, y_prob):
    from src.analytics.churn_threshold import calculate_threshold_sensitivity
    return calculate_threshold_sensitivity(y_true, y_prob)
def _run_segment_backend(y_true, y_pred, segments, y_prob):
    from src.analytics.segment_model_audit import calculate_segment_model_audit
    return calculate_segment_model_audit(
        y_true=y_true,
        y_pred=y_pred,
        segments=segments,
        y_prob=y_prob,
    )
def _run_calibration_backend(y_true, y_prob):
    from src.analytics.model_calibration import (
        calculate_calibration_reliability,
        calculate_calibration_summary,
    )
    return {
        "reliability": calculate_calibration_reliability(y_true, y_prob),
        "summary": calculate_calibration_summary(y_true, y_prob),
    }
def _normalise_threshold_output(result):
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, pd.DataFrame):
                return item.copy()
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if isinstance(result, dict):
        for key in ("results", "threshold_results", "data"):
            if isinstance(result.get(key), pd.DataFrame):
                return result[key].copy()
    raise TypeError("Threshold backend did not return a DataFrame-compatible result.")
def _normalise_segment_output(result):
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, pd.DataFrame):
                return item.copy()
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if isinstance(result, dict):
        for key in ("results", "segment_results", "data"):
            if isinstance(result.get(key), pd.DataFrame):
                return result[key].copy()
    raise TypeError("Segment-audit backend did not return a DataFrame-compatible result.")
def _normalise_calibration_output(result):
    if not isinstance(result, dict):
        raise TypeError("Calibration backend did not return a dictionary result.")
    reliability = result.get("reliability")
    summary = result.get("summary")
    if not isinstance(reliability, pd.DataFrame):
        raise TypeError("Calibration backend reliability output must be a DataFrame.")
    if not isinstance(summary, dict):
        raise TypeError("Calibration backend summary output must be a dictionary.")
    return reliability.copy(), dict(summary)
def _find_column(df, candidates):
    lower = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None
def render_model_diagnostics_page(customer_analytics, customers, data_mode):
    """Render Page 1: Model Diagnostics & Threshold Analysis."""
    del customers  # Kept in the page signature for routing compatibility.
    active_dataset = st.session_state.get("active_dataset", "Demo Dataset")
    badge_color = COLOR_PRIMARY if active_dataset == "Custom Data" else COLOR_TERTIARY
    left, right = st.columns([5.5, 1.5], vertical_alignment="center")
    with left:
        st.markdown(
            '<div style="font-size:31px;font-weight:750;letter-spacing:-.03em;">'
            "Model Diagnostics &amp; Threshold Analysis</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:13px;color:'
            f'{COLOR_TEXT_MUTED};margin-top:4px;">'
            "Understand how the churn model behaves across decision thresholds and RFM segments."
            "</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div style="text-align:right;margin-top:4px;">'
            f'<span style="display:inline-block;padding:5px 10px;border-radius:14px;'
            f'background:{COLOR_BACKGROUND};border:1px solid {COLOR_BORDER};'
            f'color:{badge_color};font-size:11px;font-weight:700;">'
            f'{active_dataset.upper()}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div style="height:1px;background:{COLOR_BORDER};margin:16px 0 20px;"></div>',
        unsafe_allow_html=True,
    )
    evaluation = _prepare_evaluation(customer_analytics, data_mode)
    if evaluation.empty:
        st.info(
            "Model diagnostics require a complete observed 90-day evaluation window. "
            "Custom data must contain enough transaction history to build a historical "
            "evaluation cohort and observe the following 90 days."
        )
        return
    y_true = evaluation["churn"].astype(int).to_numpy()
    y_prob = evaluation["churn_probability"].clip(0, 1).to_numpy()
    st.subheader("Threshold Sensitivity")
    st.caption(
        "Classification trade-offs across the tested 20%–80% probability thresholds. "
        "The existing 50% threshold is shown as the baseline, not as a universally optimal threshold."
    )
    threshold_data = _normalise_threshold_output(_run_threshold_backend(y_true, y_prob))
    threshold_col = _find_column(threshold_data, ["threshold"])
    if threshold_col is None:
        raise ValueError("Threshold backend output is missing the threshold column.")
    metric_columns = {
        "Precision": _find_column(threshold_data, ["precision"]),
        "Recall": _find_column(threshold_data, ["recall"]),
        "F1": _find_column(threshold_data, ["f1", "f1_score"]),
    }
    metric_columns = {label: col for label, col in metric_columns.items() if col}
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    baseline = threshold_data.loc[
        (pd.to_numeric(threshold_data[threshold_col], errors="coerce") - 0.50).abs().idxmin()
    ]
    with k1:
        _kpi("BASELINE THRESHOLD", "50%", "Existing operating threshold", COLOR_PRIMARY)
    with k2:
        _kpi("BASELINE PRECISION", f"{float(baseline[metric_columns['Precision']]):.1%}", "At 50% threshold", COLOR_SECONDARY)

    with k3:
        _kpi("BASELINE RECALL", f"{float(baseline[metric_columns['Recall']]):.1%}", "At 50% threshold", COLOR_TERTIARY)
    with k4:
        _kpi("BASELINE F1", f"{float(baseline[metric_columns['F1']]):.1%}", "At 50% threshold", COLOR_ALERT)
    chart_data = threshold_data[[threshold_col] + list(metric_columns.values())].copy()
    chart_data.columns = ["Threshold"] + list(metric_columns.keys())
    chart_data = chart_data.melt(
        id_vars="Threshold", var_name="Metric", value_name="Score"
    )
    fig = px.line(

        chart_data,

        x="Threshold",
        y="Score",
        color="Metric",
        markers=True,
        color_discrete_sequence=CHART_PALETTE[:3],
    )
    fig = _chart_style(fig)
    fig.update_layout(
        xaxis_title="Churn Probability Threshold",
        yaxis_title="Score",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
        height=410,
        legend_title=None,
    )
    fig.add_vline(
        x=0.50,
        line_dash="dash",
        line_color=COLOR_BORDER,
        annotation_text="Baseline 50%",
        annotation_position="top right",
    )
    st.plotly_chart(fig, width="stretch")
    display_threshold = threshold_data.copy()
    if threshold_col:
        display_threshold[threshold_col] = pd.to_numeric(
            display_threshold[threshold_col], errors="coerce"
        ).map(lambda value: f"{value:.0%}")
    for col in display_threshold.columns:
        if col.lower() in {"accuracy", "precision", "recall", "f1", "f1_score", "predicted churn rate", "predicted_churn_rate"}:

            display_threshold[col] = pd.to_numeric(display_threshold[col], errors="coerce").map(
                lambda value: f"{value:.1%}"
            )
    _table(display_threshold)
    st.divider()
    st.subheader("Model Calibration & Reliability")
    st.caption(
        "Measures whether predicted churn probabilities broadly match observed churn "
        "frequencies across probability bands. Lower calibration error and Brier score "
        "indicate closer agreement."
    )
    calibration_result = _run_calibration_backend(y_true, y_prob)
    calibration_data, calibration_summary = _normalise_calibration_output(
        calibration_result
    )
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        _kpi(
            "BRIER SCORE",
            f"{calibration_summary['brier_score']:.3f}",
            "Lower is better",
            COLOR_PRIMARY,
        )
    with c2:
        _kpi(
            "MEAN PREDICTED",
            f"{calibration_summary['mean_predicted_probability']:.1%}",
            "Average predicted churn probability",
            COLOR_SECONDARY,
        )
    with c3:
        _kpi(
            "OBSERVED CHURN",
            f"{calibration_summary['observed_churn_rate']:.1%}",
            "Observed evaluation outcome",
            COLOR_TERTIARY,
        )
    with c4:
        _kpi(
            "WEIGHTED CALIBRATION GAP",
            f"{calibration_summary['weighted_mean_absolute_calibration_gap']:.1%}",
            "Weighted absolute gap across populated bins",
            COLOR_ALERT,
        )

    plot_data = calibration_data.dropna(
        subset=["mean_predicted_probability", "observed_churn_rate"]
    ).copy()

    if plot_data.empty:
        st.info("No populated probability bins are available for the reliability diagram.")
    else:
        plot_data["customers"] = pd.to_numeric(
            plot_data["customers"], errors="coerce"
        ).fillna(0)

        fig = px.scatter(
            plot_data,
            x="mean_predicted_probability",
            y="observed_churn_rate",
            size="customers",
            text="probability_bin",
            hover_data={
                "mean_predicted_probability": ":.1%",
                "observed_churn_rate": ":.1%",
                "customers": True,
                "calibration_gap": ":.1%",
            },
        )
        fig = _chart_style(fig)
        fig.update_traces(textposition="top center")
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"dash": "dash", "color": COLOR_BORDER},
        )
        fig.update_layout(
            xaxis_title="Mean Predicted Churn Probability",
            yaxis_title="Observed Churn Rate",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    display_calibration = calibration_data.copy()
    for col in (
        "mean_predicted_probability",
        "observed_churn_rate",
        "calibration_gap",
        "absolute_calibration_gap",
    ):
        if col in display_calibration.columns:
            display_calibration[col] = pd.to_numeric(
                display_calibration[col], errors="coerce"
            ).map(lambda value: "" if pd.isna(value) else f"{value:.1%}")

    if "customers" in display_calibration.columns:
        display_calibration["customers"] = pd.to_numeric(
            display_calibration["customers"], errors="coerce"
        ).fillna(0).astype(int).map(lambda value: f"{value:,}")

    _table(display_calibration)

    with st.expander("Calibration methodology & interpretation"):
        st.markdown(
            "- **Brier score:** mean squared error between predicted churn probability and the observed binary outcome; lower values indicate better probabilistic accuracy.\n"
            "- **Calibration gap:** observed churn rate minus mean predicted probability within each probability bin. Positive values indicate underprediction; negative values indicate overprediction.\n"
            "- **Weighted mean absolute calibration gap:** average absolute bin-level calibration error weighted by the number of observations in each populated bin.\n"
            "- The reliability diagram compares observed churn frequency with predicted probability. The dashed diagonal represents perfect calibration.\n"
            "- Calibration is a diagnostic of probability reliability; it does not establish that changing a probability threshold will improve retention outcomes."
        )

    st.divider()
    st.subheader("Segment-Level Performance Audit")

    st.caption(

        "Compares classification behaviour across RFM customer segments at the default 50% threshold. "

        "Small segment samples should be interpreted cautiously."

    )
    if "customer_segment" not in evaluation.columns:

        st.info("Customer segment information is unavailable for the model audit.")

        return
    segment_pred = (y_prob >= 0.50).astype(int)

    segment_data = _normalise_segment_output(

        _run_segment_backend(

            y_true,

            segment_pred,

            evaluation["customer_segment"].astype(str).to_numpy(),

            y_prob,

        )

    )
    support_col = _find_column(segment_data, ["sample_count", "support", "count", "n"])
    segment_col = _find_column(segment_data, ["segment", "customer_segment"])
    accuracy_col = _find_column(segment_data, ["accuracy"])
    precision_col = _find_column(segment_data, ["precision"])
    recall_col = _find_column(segment_data, ["recall"])
    f1_col = _find_column(segment_data, ["f1", "f1_score"])
    if segment_col is None:
        segment_col = segment_data.columns[0]
    if support_col:
        segment_data = segment_data.sort_values(support_col, ascending=False)
    k1, k2, k3 = st.columns(3, gap="medium")
    with k1:
        _kpi("SEGMENTS AUDITED", f"{len(segment_data):,}", "RFM segments with evaluation data", COLOR_PRIMARY)
    with k2:
        largest_support = int(pd.to_numeric(segment_data[support_col], errors="coerce").max()) if support_col else 0
        _kpi("LARGEST SUPPORT", f"{largest_support:,}", "Observations in the largest segment", COLOR_SECONDARY)
    with k3:
        _kpi("EVALUATION OBSERVATIONS", f"{len(evaluation):,}", "Observed churn outcomes", COLOR_TERTIARY)
    display_segment = segment_data.copy()
    for col in display_segment.columns:
        if col.lower() in {"accuracy", "precision", "recall", "f1", "f1_score", "observed churn rate", "observed_churn_rate"}:

            display_segment[col] = pd.to_numeric(display_segment[col], errors="coerce").map(
                lambda value: f"{value:.1%}"
            )
        elif col.lower() in {"brier score", "brier_score"}:
            display_segment[col] = pd.to_numeric(display_segment[col], errors="coerce").map(
                lambda value: f"{value:.3f}"
            )
    _table(display_segment)
    chart_metrics = [
        ("Precision", precision_col),
        ("Recall", recall_col),
        ("F1", f1_col),
    ]
    chart_metrics = [(label, col) for label, col in chart_metrics if col]
    if chart_metrics:
        audit_chart = segment_data[[segment_col] + [col for _, col in chart_metrics]].copy()
        audit_chart.columns = ["Segment"] + [label for label, _ in chart_metrics]
        audit_chart = audit_chart.melt(
            id_vars="Segment", var_name="Metric", value_name="Score"
        )
        fig = px.bar(
            audit_chart,
            x="Score",
            y="Segment",
            color="Metric",
            barmode="group",
            orientation="h",
            color_discrete_sequence=CHART_PALETTE[:3],
        )
        fig = _chart_style(fig)

        fig.update_layout(
            xaxis_title="Score",
            yaxis_title=None,
            xaxis_tickformat=".0%",
            xaxis_range=[0, 1],
            height=430,
            legend_title=None,
        )
        st.plotly_chart(fig, width="stretch")
    with st.expander("Methodology & interpretation"):
        st.markdown(
            "- Threshold sensitivity reuses the existing model probabilities and held-out labels; it does not retrain the model.\n"

            "- The 50% threshold is the established baseline and is shown for reference, not as an optimization result.\n"

            "- Segment-level results describe model behaviour on the supplied evaluation data. They are diagnostic observations, not causal segment effects.\n"

            "- For custom data, the evaluation cohort is reconstructed at a historical observation date so the following 90-day churn outcome can be observed from the uploaded transaction history.\n"

            "- Custom-data probabilities are generated by applying the persisted Random Forest to features rebuilt at that historical observation date; the observed churn label is derived independently from subsequent transactions."

        )
