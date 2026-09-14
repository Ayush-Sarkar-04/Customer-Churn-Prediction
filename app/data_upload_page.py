import streamlit as st
from textwrap import dedent

from config import (
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE_ALT,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_CARD,
    COLOR_CARD_TEXT,
    COLOR_CARD_MUTED,
)

from src.data.templates import (
    template_csv_bytes,
    get_template_filename,
)
from src.data.upload import (
    load_uploaded_datasets,
)


# =========================================================
# SESSION STATE
# =========================================================

def initialize_dataset_state():
    """Initialize the application's dataset state."""
    if "active_dataset" not in st.session_state:
        st.session_state.active_dataset = "Demo Dataset"

    if "custom_data" not in st.session_state:
        st.session_state.custom_data = None

    if "upload_validation" not in st.session_state:
        st.session_state.upload_validation = None

    if "data_source_mode" not in st.session_state:
        st.session_state.data_source_mode = "Demo Dataset"

    # Custom Data is active only when validated custom data is actually
    # present. This prevents a stale session-state marker from claiming
    # Custom Data while the application is really using the demo dataset.
    custom_data = st.session_state.get("custom_data")
    if st.session_state.get("active_dataset") == "Custom Data":
        if not isinstance(custom_data, dict) or not all(
            custom_data.get(key) is not None
            for key in ("customers", "transactions", "campaigns")
        ):
            st.session_state.active_dataset = "Demo Dataset"


# =========================================================
# RESET DATASET
# =========================================================

def reset_to_demo_dataset():
    """Remove the active custom dataset and return to demo data."""
    st.session_state.active_dataset = "Demo Dataset"
    st.session_state.custom_data = None
    st.session_state.upload_validation = None
    st.session_state.data_source_mode = "Demo Dataset"


# =========================================================
# TEMPLATE DOWNLOAD
# =========================================================

def render_template_downloads():
    """Render blank CSV template download controls."""

    st.subheader("Download Templates")
    st.caption(
        "Use the predefined templates to prepare custom datasets "
        "with the required schemas."
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    templates = [
        ("customers", "Customers", col1),
        ("transactions", "Transactions", col2),
        ("campaigns", "Campaigns", col3),
    ]

    for dataset_name, label, column in templates:
        with column:
            st.markdown(
                dedent(f"""
                <div style="
                    background:{COLOR_CARD};
                    border:1px solid {COLOR_BORDER};
                    border-radius:8px;
                    padding:15px 16px 13px 16px;
                    margin-bottom:7px;
                ">
                    <div style="
                        color:{COLOR_CARD_TEXT};
                        font-size:13px;
                        font-weight:700;
                    ">{label}</div>
                    <div style="
                        color:{COLOR_CARD_MUTED};
                        font-size:10px;
                        margin-top:4px;
                    ">Required CSV structure</div>
                </div>
                """),
                unsafe_allow_html=True,
            )

            st.download_button(
                label=f"Download {label} Template",
                data=template_csv_bytes(dataset_name),
                file_name=get_template_filename(dataset_name),
                mime="text/csv",
                width="stretch",
                key=f"download_{dataset_name}_template",
            )


# =========================================================
# CUSTOM UPLOAD
# =========================================================

def render_custom_upload():
    """Render the three-file custom dataset upload interface."""

    st.subheader("Upload Files")
    st.caption(
        "All three files must pass validation before they become active. "
        "Maximum file size: 50 MB per file."
    )

    customers_file = st.file_uploader(
        "Customers",
        type=["csv"],
        key="customers_upload",
        help="Upload the Customers CSV file. Maximum size: 50 MB.",
    )

    transactions_file = st.file_uploader(
        "Transactions",
        type=["csv"],
        key="transactions_upload",
        help="Upload the Transactions CSV file. Maximum size: 50 MB.",
    )

    campaigns_file = st.file_uploader(
        "Campaigns",
        type=["csv"],
        key="campaigns_upload",
        help="Upload the Campaigns CSV file. Maximum size: 50 MB.",
    )

    files_ready = all(
        [
            customers_file is not None,
            transactions_file is not None,
            campaigns_file is not None,
        ]
    )

    if st.button(
        "Validate & Load Data",
        width="stretch",
        disabled=not files_ready,
        key="validate_custom_data",
    ):
        uploaded_files = {
            "customers": customers_file,
            "transactions": transactions_file,
            "campaigns": campaigns_file,
        }

        with st.spinner("Validating uploaded data..."):
            result = load_uploaded_datasets(uploaded_files)

        st.session_state.upload_validation = result

        if result["valid"]:
            st.session_state.custom_data = result["data"]
            st.session_state.active_dataset = "Custom Data"

            st.success(
                "Custom dataset validated successfully and is now active."
            )
        else:
            # Failed validation never becomes an active data source.
            st.session_state.custom_data = None
            st.session_state.active_dataset = "Demo Dataset"

            st.error(
                "Custom dataset was rejected. "
                "Resolve the validation issues below."
            )


# =========================================================
# VALIDATION RESULTS
# =========================================================

def render_validation_results():
    """Display the latest upload validation result."""

    result = st.session_state.get("upload_validation")

    if not result:
        return

    st.divider()
    st.subheader("Validation Result")

    if result["valid"]:
        st.markdown("**DATASET VALID**")

        data = result["data"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Customers", f"{len(data['customers']):,}")

        with col2:
            st.metric("Transactions", f"{len(data['transactions']):,}")

        with col3:
            st.metric("Campaigns", f"{len(data['campaigns']):,}")

    else:
        st.markdown("**DATASET REJECTED**")
        st.markdown("Validation issues:")

        for error in result.get("errors", []):
            st.write(f"- {error}")


# =========================================================
# ACTIVE DATASET
# =========================================================

def render_active_dataset():
    """Display the currently selected dataset state."""

    active_dataset = st.session_state.get(
        "active_dataset",
        "Demo Dataset",
    )

    custom_data = st.session_state.get("custom_data")

    if active_dataset == "Custom Data" and custom_data:
        customers = len(custom_data["customers"])
        transactions = len(custom_data["transactions"])
        campaigns = len(custom_data["campaigns"])

        st.markdown(
            dedent(f"""
            <div style="
                background:{COLOR_PRIMARY};
                border:1px solid {COLOR_PRIMARY};
                border-left:4px solid {COLOR_PRIMARY};
                border-radius:8px;
                padding:16px 18px;
                margin-top:18px;
            ">
                <div style="
                    color:{COLOR_TEXT};
                    font-size:9px;
                    font-weight:800;
                    letter-spacing:.12em;
                ">ACTIVE DATASET</div>
                <div style="
                    color:{COLOR_TEXT};
                    font-size:17px;
                    font-weight:750;
                    margin-top:4px;
                ">CUSTOM DATA</div>
                <div style="
                    color:{COLOR_TEXT};
                    font-size:10px;
                    margin-top:4px;
                ">
                    {customers:,} customers · {transactions:,} transactions ·
                    {campaigns:,} campaigns
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        if st.button(
            "Reset to Demo Dataset",
            width="stretch",
            key="reset_demo_dataset",
        ):
            reset_to_demo_dataset()
            st.rerun()

    else:
        st.markdown(
            dedent(f"""
            <div style="
                background:{COLOR_PRIMARY};
                border:1px solid {COLOR_PRIMARY};
                border-left:4px solid {COLOR_PRIMARY};
                border-radius:8px;
                padding:16px 18px;
                margin-top:18px;
            ">
                <div style="
                    color:{COLOR_TEXT};
                    font-size:9px;
                    font-weight:800;
                    letter-spacing:.12em;
                ">ACTIVE DATASET</div>
                <div style="
                    color:{COLOR_TEXT};
                    font-size:17px;
                    font-weight:750;
                    margin-top:4px;
                ">DEMO DATASET</div>
                <div style="
                    color:{COLOR_TEXT};
                    font-size:10px;
                    margin-top:4px;
                ">
                    The fixed dataset included with the project is currently active.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )


# =========================================================
# MAIN PAGE
# =========================================================

def render_data_upload_page():
    """Render the Data & Upload control center."""

    initialize_dataset_state()

    active_dataset = st.session_state.get("active_dataset", "Demo Dataset")
    custom_data = st.session_state.get("custom_data")
    custom_is_active = (
        active_dataset == "Custom Data"
        and isinstance(custom_data, dict)
        and all(custom_data.get(key) is not None for key in (
            "customers", "transactions", "campaigns"
        ))
    )

    # The visible source selector is deliberately separate from the active
    # dataset. Selecting Custom Data only opens the upload workflow; it does
    # not activate Custom Data until validation succeeds.
    if custom_is_active:
        st.session_state.data_source_mode = "Custom Data"
    elif "data_source_mode" not in st.session_state:
        st.session_state.data_source_mode = "Demo Dataset"

    source_mode = st.session_state.get(
        "data_source_mode",
        "Custom Data" if custom_is_active else "Demo Dataset",
    )

    # ---------------------------------------------------------
    # PAGE HEADER
    # ---------------------------------------------------------

    st.title("Data & Upload")
    st.caption(
        "Select the dataset powering the application, prepare custom CSVs, "
        "validate them, and switch the active data source."
    )

    # ---------------------------------------------------------
    # DATASET SOURCE
    # ---------------------------------------------------------

    with st.container(border=True):
        st.markdown("**Dataset Source**")
        st.caption("Choose the data source used by the application.")

        col1, col2 = st.columns(2, gap="small")

        with col1:
            if st.button(
                "Demo Dataset",
                width="stretch",
                type="primary" if source_mode == "Demo Dataset" else "secondary",
                key="source_demo_dataset",
            ):
                reset_to_demo_dataset()
                st.rerun()

        with col2:
            if st.button(
                "Custom Data",
                width="stretch",
                type="primary" if source_mode == "Custom Data" else "secondary",
                key="source_custom_data",
            ):
                st.session_state.data_source_mode = "Custom Data"
                st.rerun()

    # ---------------------------------------------------------
    # ACTIVE DATA SOURCE
    # ---------------------------------------------------------

    render_active_dataset()

    # ---------------------------------------------------------
    # CUSTOM DATA WORKFLOW
    # ---------------------------------------------------------

    if source_mode == "Custom Data":
        render_template_downloads()
        st.divider()
        render_custom_upload()
        render_validation_results()
    else:
        st.markdown(
            "Use the Custom Data button above to upload and validate your "
            "own Customers, Transactions, and Campaigns CSV files."
        )
