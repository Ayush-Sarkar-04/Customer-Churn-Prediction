import streamlit as st
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
    """
    Initialize the application's dataset state.
    """
    if "active_dataset" not in st.session_state:
        st.session_state.active_dataset = "Demo Dataset"
    if "custom_data" not in st.session_state:
        st.session_state.custom_data = None
    if "upload_validation" not in st.session_state:
        st.session_state.upload_validation = None
# =========================================================
# RESET DATASET
# =========================================================
def reset_to_demo_dataset():
    """
    Remove the active custom dataset and return to demo data.
    """
    st.session_state.active_dataset = "Demo Dataset"
    st.session_state.custom_data = None
    st.session_state.upload_validation = None
# =========================================================
# TEMPLATE DOWNLOAD
# =========================================================
def render_template_downloads():
    """
    Render blank CSV template download controls.
    """
    st.subheader("CSV Templates")
    st.caption(
        "Download the required templates before preparing custom data."
    )
    col1, col2, col3 = st.columns(3)
    templates = [
        ("customers", "Customers", col1),
        ("transactions", "Transactions", col2),
        ("campaigns", "Campaigns", col3),
    ]
    for dataset_name, label, column in templates:
        with column:
            st.markdown(f"**{label}**")

            st.download_button(
                label=f"Download {label} Template",
                data=template_csv_bytes(dataset_name),
                file_name=get_template_filename(dataset_name),
                mime="text/csv",
                use_container_width=True,
                key=f"download_{dataset_name}_template",
            )
# =========================================================
# CUSTOM UPLOAD
# =========================================================
def render_custom_upload():
    """
    Render the three-file custom dataset upload interface.
    """
    st.subheader("Upload Custom Dataset")
    st.caption(
        "Upload Customers, Transactions and Campaigns CSV files. "
        "All three files must pass validation before they become active."
    )
    customers_file = st.file_uploader(
        "Customers",
        type=["csv"],
        key="customers_upload",
        help="Upload the Customers CSV file.",
    )
    transactions_file = st.file_uploader(
        "Transactions",
        type=["csv"],
        key="transactions_upload",
        help="Upload the Transactions CSV file.",
    )
    campaigns_file = st.file_uploader(
        "Campaigns",
        type=["csv"],
        key="campaigns_upload",
        help="Upload the Campaigns CSV file.",
    )
    files_ready = all(
        [
            customers_file is not None,
            transactions_file is not None,
            campaigns_file is not None,
        ]
    )
    st.divider()
    if st.button(
        "Validate & Load Data",
        use_container_width=True,
        disabled=not files_ready,
        key="validate_custom_data",
    ):
        uploaded_files = {
            "customers": customers_file,
            "transactions": transactions_file,
            "campaigns": campaigns_file,
        }
        with st.spinner("Validating uploaded data..."):
            result = load_uploaded_datasets(
                uploaded_files
            )
        st.session_state.upload_validation = result
        if result["valid"]:
            st.session_state.custom_data = result["data"]
            st.session_state.active_dataset = "Custom Data"
            st.success(
                "Custom dataset validated successfully and is now active."
            )
        else:
            # Never replace an existing valid dataset with invalid data.
            st.session_state.custom_data = None
            st.error(
                "Custom dataset was rejected. "
                "Resolve the validation issues below."
            )
# =========================================================
# VALIDATION RESULTS
# =========================================================
def render_validation_results():
    """
    Display the latest upload validation result.
    """
    result = st.session_state.get(
        "upload_validation"
    )
    if not result:
        return
    st.divider()
    st.subheader("Validation Result")
    if result["valid"]:
        st.markdown("**DATASET VALID**")
        data = result["data"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Customers",
                f"{len(data['customers']):,}",
            )
        with col2:
            st.metric(
                "Transactions",
                f"{len(data['transactions']):,}",
            )
        with col3:
            st.metric(
                "Campaigns",
                f"{len(data['campaigns']):,}",
            )
    else:
        st.markdown("**DATASET REJECTED**")
        st.markdown("Validation issues:")
        for error in result.get("errors", []):
            st.write(f"- {error}")
# =========================================================
# ACTIVE DATASET
# =========================================================
def render_active_dataset():
    """
    Display the currently selected dataset state.
    """
    st.divider()
    st.subheader("Active Dataset")
    active_dataset = st.session_state.get(
        "active_dataset",
        "Demo Dataset",
    )
    if active_dataset == "Custom Data":
        st.markdown("**CUSTOM DATA**")
        custom_data = st.session_state.get(
            "custom_data"
        )
        if custom_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Customers",
                    f"{len(custom_data['customers']):,}",
                )
            with col2:
                st.metric(
                    "Transactions",
                    f"{len(custom_data['transactions']):,}",
                )
            with col3:
                st.metric(
                    "Campaigns",
                    f"{len(custom_data['campaigns']):,}",
                )
        if st.button(
            "Reset to Demo Dataset",
            use_container_width=True,
            key="reset_demo_dataset",
        ):
            reset_to_demo_dataset()
            st.rerun()

    else:
        st.markdown("**DEMO DATASET**")
        st.caption(
            "The application is currently using the project's fixed demo data."
        )
# =========================================================
# MAIN PAGE
# =========================================================
def render_data_upload_page():
    """
    Render the complete Data & Upload page.
    """
    initialize_dataset_state()
    st.title("Data & Upload")
    st.caption(
        "Choose the dataset used by the customer intelligence system."
    )
    st.divider()
    # -----------------------------------------------------
    # DATA SOURCE
    # -----------------------------------------------------
    active_dataset = st.session_state.get(
        "active_dataset",
        "Demo Dataset",
    )
    st.subheader("Data Source")
    selected_source = st.radio(
        "Dataset",
        [
            "Demo Dataset",
            "Custom Data",
        ],
        index=(
            0
            if active_dataset == "Demo Dataset"
            else 1
        ),
        horizontal=True,
        key="dataset_source_selector",
    )
    # -----------------------------------------------------
    # DEMO DATASET
    # -----------------------------------------------------
    if selected_source == "Demo Dataset":
        if active_dataset == "Custom Data":
            reset_to_demo_dataset()
        st.markdown("**Demo Dataset**")
        st.caption(
            "Use the fixed dataset included with the project."
        )
    # -----------------------------------------------------
    # CUSTOM DATA
    # -----------------------------------------------------
    else:
        render_template_downloads()
        st.divider()
        render_custom_upload()
        render_validation_results()
    # -----------------------------------------------------
    # ACTIVE DATASET
    # -----------------------------------------------------
    render_active_dataset()