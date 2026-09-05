"""
app.py
------
Purpose:
    Streamlit Web Application entry point for the AI Data Analyst Agent.
    Provides a clean dashboard for uploading CSV datasets, inspecting summaries,
    asking natural language questions, validating generated code, executing in sandbox,
    and rendering automated charts.
"""

from typing import Optional, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import load_csv, get_dataset_summary, format_value
from agent import DataAnalystAgent
from validator import clean_code, validate_code
from sandbox import execute_pandas_code
from charts import generate_chart


@st.cache_data(show_spinner=False)
def load_cached_csv(file_buffer: Any) -> pd.DataFrame:
    """
    Cached wrapper around load_csv to prevent re-reading CSV on Streamlit reruns.
    """
    return load_csv(file_buffer)


def setup_page_config() -> None:
    """Configures Streamlit page parameters, title, icon, and layout."""
    st.set_page_config(
        page_title="AI Data Analyst Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_header() -> None:
    """Renders the main page title, badge, and application overview."""
    st.title("📊 AI Data Analyst Agent")
    st.markdown(
        """
        Welcome to **AI Data Analyst Agent**! Upload any CSV dataset to view automated summary metrics,
        inspect columns and data types, and ask natural language questions about your data.
        """
    )
    st.divider()


def render_sidebar() -> Optional[Any]:
    """
    Renders the sidebar controls including project description,
    file uploader, and upload status badge.

    Returns:
        Optional[Any]: Streamlit UploadedFile object if a file is uploaded, else None.
    """
    with st.sidebar:
        st.header("⚙️ Control Panel")
        st.caption("AI Data Analyst Agent v1.0")
        
        st.markdown(
            """
            ### About
            This agent translates plain-English questions into executable Pandas operations,
            safely processes your data, and generates relevant visualizations.
            """
        )
        
        st.divider()
        st.subheader("📁 Data Source")
        
        uploaded_file = st.file_uploader(
            label="Upload a CSV dataset",
            type=["csv"],
            help="Select a CSV file from your computer to analyze."
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Loaded: `{uploaded_file.name}`")
        else:
            st.info("👆 Please upload a CSV file to get started.")
            
        st.divider()
        st.markdown(
            """
            **Tech Stack:**
            - Python 3.10+
            - Streamlit
            - Pandas & Matplotlib
            - Google Gemini API
            """
        )
        
        return uploaded_file


def render_metrics(summary: Dict[str, Any]) -> None:
    """
    Renders high-level KPI cards for the uploaded dataset.

    Args:
        summary (Dict[str, Any]): Summary dictionary returned by get_dataset_summary().
    """
    st.subheader("📈 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Rows",
            value=f"{summary['num_rows']:,}"
        )
    with col2:
        st.metric(
            label="Total Columns",
            value=f"{summary['num_cols']:,}"
        )
    with col3:
        st.metric(
            label="Missing Values",
            value=f"{summary['total_missing']:,}",
            delta="Clean" if summary['total_missing'] == 0 else f"{summary['total_missing']} missing",
            delta_color="normal" if summary['total_missing'] == 0 else "inverse"
        )
    with col4:
        st.metric(
            label="Duplicate Rows",
            value=f"{summary['duplicate_rows']:,}",
            delta="Unique" if summary['duplicate_rows'] == 0 else f"{summary['duplicate_rows']} duplicates",
            delta_color="normal" if summary['duplicate_rows'] == 0 else "inverse"
        )


def render_dataset_preview(summary: Dict[str, Any]) -> None:
    """
    Renders interactive dataset preview tabs including sample rows,
    column metadata schema table, and raw column list.

    Args:
        summary (Dict[str, Any]): Dataset metadata summary.
    """
    tab_preview, tab_schema, tab_cols = st.tabs([
        "🔍 Data Preview (First 10 Rows)",
        "📋 Column Schema & Missing Values",
        "🏷️ Column Names List"
    ])
    
    with tab_preview:
        st.markdown("Showing the first **10 rows** of the dataset:")
        st.dataframe(summary["head"], use_container_width=True)
        
    with tab_schema:
        st.markdown("Detailed breakdown of data types and null counts per column:")
        st.dataframe(summary["schema_df"], use_container_width=True, hide_index=True)
        
    with tab_cols:
        st.markdown(f"**All Columns ({len(summary['columns'])} total):**")
        st.write(summary["columns"])


def render_result(result: Any) -> None:
    """
    Renders execution output according to its data type.

    Args:
        result (Any): The calculated 'result' object from the sandbox.
    """
    st.subheader("💡 Answer")

    if isinstance(result, (int, float, str, bool, np.number, np.bool_)):
        st.metric(label="Result", value=format_value(result))
    elif isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True)
    elif isinstance(result, pd.Series):
        st.dataframe(result.to_frame(), use_container_width=True)
    elif isinstance(result, dict):
        st.json(result)
    elif isinstance(result, (list, tuple)):
        st.write(result)
    else:
        st.write(result)


def render_query_section(df: pd.DataFrame) -> None:
    """
    Renders the 'Ask a Question' natural language query section.
    Full generation -> AST validation -> Sandbox execution -> Auto-Visualization workflow.

    Args:
        df (pd.DataFrame): Target dataset DataFrame.
    """
    st.divider()
    st.subheader("💬 Ask a Question About Your Data")
    st.caption("Type a question in plain English (e.g., 'What are the top 5 highest sales regions?')")
    
    with st.container():
        user_query = st.text_input(
            label="Enter your question:",
            placeholder="e.g. Find the average price by product category and show the top 3",
            key="user_question_input"
        )
        
        ask_clicked = st.button("🚀 Ask Agent", type="primary", use_container_width=False)
        
        if ask_clicked:
            if not user_query.strip():
                st.warning("⚠️ Please enter a question before clicking Ask.")
                return

            with st.spinner("🧠 Analyzing data, executing query, and rendering visualization..."):
                try:
                    # 1. Generate code via Gemini
                    agent = DataAnalystAgent()
                    generated_code = agent.generate_code(df, user_query)
                    cleaned_code = clean_code(generated_code)

                    st.markdown(f"**Question:** {user_query}")

                    # 2. Validate AST code safety
                    is_valid, validation_msg = validate_code(cleaned_code)

                    if not is_valid:
                        st.error(f"❌ **Validation Error:** {validation_msg}")
                        with st.expander("🔍 View Rejected Code"):
                            st.code(cleaned_code, language="python")
                        return

                    # 3. Sandbox execution (with deep copy)
                    success, result_or_error = execute_pandas_code(cleaned_code, df)

                    if not success:
                        st.error(f"❌ **Execution Error:** {result_or_error}")
                    else:
                        # 4. Display result
                        render_result(result_or_error)

                        # 5. Automated Chart Generation
                        fig, chart_name = generate_chart(result_or_error)
                        if fig is not None:
                            st.divider()
                            st.subheader("📊 Visualization")
                            st.pyplot(fig)
                            st.caption(f"**Chart:** {chart_name}")
                            plt.close(fig)

                    # Keep generated code visible below answer & chart
                    with st.expander("💻 View Executed Pandas Code", expanded=False):
                        st.code(cleaned_code, language="python")
                        if success:
                            st.caption("Validation: ✅ Passed | Execution: ✅ Passed")
                        else:
                            st.caption("Validation: ✅ Passed | Execution: ❌ Failed")

                except ValueError as ve:
                    st.error(f"⚠️ **Configuration / Input Error:** {str(ve)}")
                except RuntimeError as re:
                    st.error(f"❌ **API Error:** {str(re)}")
                except Exception as e:
                    st.error(f"❌ **Unexpected Failure:** {str(e)}")


def main() -> None:
    """Main application layout and execution workflow."""
    setup_page_config()
    render_header()
    
    uploaded_file = render_sidebar()
    
    if uploaded_file is not None:
        try:
            # Cached CSV load
            df = load_cached_csv(uploaded_file)
            summary = get_dataset_summary(df)
            
            # Display dataset analytics UI
            render_metrics(summary)
            st.divider()
            render_dataset_preview(summary)
            
            # Display Question Input Section
            render_query_section(df)
            
        except ValueError as ve:
            st.error(f"❌ **CSV Error:** {str(ve)}")
        except Exception as e:
            st.error(f"❌ **Unexpected Error:** Unable to process file. Details: {str(e)}")
    else:
        st.info("👈 Upload a CSV file from the sidebar panel to view dataset statistics and ask questions.")


if __name__ == "__main__":
    main()
