import streamlit as st
import pandas as pd

from src.analytics.data_quality import (
    build_data_quality_report,
    get_quality_summary,
)


def render_data_quality_page(
    customers_df,
    transactions_df,
    campaigns_df,
):
    """
    Render the Data Quality dashboard page.
    """

    st.title("Data Quality")

    # Build complete quality report
    report = build_data_quality_report(
        customers_df,
        transactions_df,
        campaigns_df,
    )

    quality = get_quality_summary(report)

    # ---------------------------------------------------------
    # Overall status
    # ---------------------------------------------------------
    if quality["overall_status"] == "VALID":
        st.success("✓ DATASET VALID")
    else:
        st.error("✕ DATASET REJECTED")

    # ---------------------------------------------------------
    # Summary metrics
    # ---------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Files Checked",
            3,
        )

    with col2:
        total_rows = (
            report["Customers"]["row_count"]
            + report["Transactions"]["row_count"]
            + report["Campaigns"]["row_count"]
        )

        st.metric(
            "Total Rows",
            f"{total_rows:,}",
        )

    with col3:
        st.metric(
            "Checks Passed",
            quality["passed_checks"],
        )

    with col4:
        st.metric(
            "Checks Failed",
            quality["failed_checks"],
        )

    st.divider()

    # ---------------------------------------------------------
    # Dataset overview
    # ---------------------------------------------------------
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

    st.dataframe(
        overview,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # ---------------------------------------------------------
    # Validation results
    # ---------------------------------------------------------
    st.subheader("Validation Checks")

    summary = quality["summary"].copy()

    display_summary = summary.rename(
        columns={
            "dataset": "Dataset",
            "check": "Check",
            "status": "Status",
        }
    )

    st.dataframe(
        display_summary,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # ---------------------------------------------------------
    # Failed checks / errors
    # ---------------------------------------------------------
    if quality["failed_checks"] > 0:

        st.subheader("Validation Issues")

        for dataset_name, dataset_report in report.items():

            # Cross-file reference checks
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

            # Dataset-level checks
            checks = dataset_report["checks"]

            for check_name, result in checks.items():

                if result["valid"]:
                    continue

                # Validators that return standard errors
                if "errors" in result:

                    for error in result["errors"]:
                        st.error(
                            f"{dataset_name} → "
                            f"{check_name}: {error}"
                        )

                # Schema validator returns missing / extra
                if "missing" in result:

                    missing = result.get(
                        "missing",
                        []
                    )

                    if missing:
                        st.error(
                            f"{dataset_name} → "
                            f"{check_name}: "
                            f"Missing columns: {missing}"
                        )

                if "extra" in result:

                    extra = result.get(
                        "extra",
                        []
                    )

                    if extra:
                        st.error(
                            f"{dataset_name} → "
                            f"{check_name}: "
                            f"Unexpected columns: {extra}"
                        )

    else:

        st.info(
            "All configured data-quality checks passed."
        )