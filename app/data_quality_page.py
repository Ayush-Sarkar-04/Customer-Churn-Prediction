import streamlit as st
import pandas as pd

from config import (
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    COLOR_SURFACE_ALT,
    COLOR_TEXT,
    COLOR_TEXT_DARK,
    COLOR_TEXT_MUTED,
    COLOR_CARD,
    COLOR_CARD_TEXT,
    COLOR_CARD_MUTED,
)



from src.analytics.data_quality import (
    build_data_quality_report,
    get_quality_summary,
)


def render_data_quality_page(
    customers_df,
    transactions_df,
    campaigns_df,
):
    """Render the Data Quality dashboard page."""

    st.markdown(
        f"""
        <div style="
            background:{COLOR_CARD};
            border:1px solid {COLOR_BORDER};
            border-radius:10px;
            padding:22px 26px;
            margin-bottom:18px;
        ">
            <div style="
                color:{COLOR_CARD_MUTED};
                font-size:9px;
                font-weight:800;
                letter-spacing:.14em;
                text-transform:uppercase;
            ">DATA GOVERNANCE</div>
            <div style="
                color:{COLOR_CARD_TEXT};
                font-size:30px;
                font-weight:800;
                letter-spacing:-.03em;
                margin-top:4px;
            ">Data Quality</div>
            <div style="
                color:{COLOR_CARD_MUTED};
                font-size:11px;
                line-height:1.45;
                margin-top:6px;
            ">
                Validate schema, data types, dates, identifiers, missing values,
                duplicates and cross-file customer references.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    report = build_data_quality_report(
        customers_df,
        transactions_df,
        campaigns_df,
    )

    quality = get_quality_summary(report)

    # Neutral status card instead of the disconnected green banner.
    status_valid = quality["overall_status"] == "VALID"
    status_text = "DATASET VALID" if status_valid else "DATASET REJECTED"
    status_detail = (
        "All configured data-quality checks passed."
        if status_valid
        else "One or more configured checks require attention."
    )
    status_accent = COLOR_SECONDARY if status_valid else COLOR_PRIMARY

    st.markdown(
        f"""
        <div style="
            background:{COLOR_BACKGROUND};
            border:1px solid {COLOR_BORDER};
            border-left:4px solid {status_accent};
            border-radius:8px;
            padding:14px 17px;
            margin-bottom:16px;
        ">
            <div style="
                color:{COLOR_TEXT};
                font-size:13px;
                font-weight:750;
            ">{status_text}</div>
            <div style="
                color:{COLOR_TEXT_MUTED};
                font-size:10px;
                margin-top:3px;
            ">{status_detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Files Checked", 3)

    with col2:
        total_rows = (
            report["Customers"]["row_count"]
            + report["Transactions"]["row_count"]
            + report["Campaigns"]["row_count"]
        )
        st.metric("Total Rows", f"{total_rows:,}")

    with col3:
        st.metric("Checks Passed", quality["passed_checks"])

    with col4:
        st.metric("Checks Failed", quality["failed_checks"])

    st.divider()

    st.subheader("Dataset Overview")

    overview = pd.DataFrame(
        [
            {
                "Dataset": "Customers",
                "Rows": report["Customers"]["row_count"],
                "Columns": report["Customers"]["column_count"],
            },
            {
                "Dataset": "Transactions",
                "Rows": report["Transactions"]["row_count"],
                "Columns": report["Transactions"]["column_count"],
            },
            {
                "Dataset": "Campaigns",
                "Rows": report["Campaigns"]["row_count"],
                "Columns": report["Campaigns"]["column_count"],
            },
        ]
    )

    st.markdown(
            '<div class="app-table-wrap">'
            + overview.to_html(
                index=False,
                classes="app-table",
                border=0,
                escape=True,
            )
            + '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Validation Checks")

    summary = quality["summary"].copy()

    display_summary = summary.rename(
        columns={
            "dataset": "Dataset",
            "check": "Check",
            "status": "Status",
        }
    )

    st.markdown(
            '<div class="app-table-wrap">'
            + display_summary.to_html(
                index=False,
                classes="app-table",
                border=0,
                escape=True,
            )
            + '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if quality["failed_checks"] > 0:
        st.subheader("Validation Issues")

        for dataset_name, dataset_report in report.items():

            if dataset_name == "Cross-file References":
                checks = dataset_report["checks"]

                for check_name, result in checks.items():
                    if not result["valid"]:
                        for error in result["errors"]:
                            st.error(
                                f"{dataset_name} → "
                                f"{check_name}: {error}"
                            )

                continue

            checks = dataset_report["checks"]

            for check_name, result in checks.items():

                if result["valid"]:
                    continue

                if "errors" in result:
                    for error in result["errors"]:
                        st.error(
                            f"{dataset_name} → "
                            f"{check_name}: {error}"
                        )

                if "missing" in result:
                    missing = result.get("missing", [])

                    if missing:
                        st.error(
                            f"{dataset_name} → "
                            f"{check_name}: "
                            f"Missing columns: {missing}"
                        )

                if "extra" in result:
                    extra = result.get("extra", [])

                    if extra:
                        st.error(
                            f"{dataset_name} → "
                            f"{check_name}: "
                            f"Unexpected columns: {extra}"
                        )
