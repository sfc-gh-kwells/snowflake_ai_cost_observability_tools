"""
Snowflake AI Cost Toolkit Dashboard

A comprehensive Streamlit dashboard for analyzing AI services costs and Cortex Analyst usage.
"""

import streamlit as st
import pandas as pd
import datetime

# Import all utility functions
from utils import (
    # Session management
    get_session,

    # Core Cortex Analyst functions
    fetch_semantic_model_paths,
    get_cortex_analyst_logs,
    verified_query_count,
    top_verified_queries,
    slowest_queries,
    latency_summary_by_semantic_model,
    user_activity_by_semantic_model,
    semantic_model_usage_summary,
    create_cortex_analyst_query_history,
    total_cost_by_semantic_model,
    cost_breakdown_by_semantic_model,

    # LLM Usage Dashboard functions
    get_ai_services_total_credits,
    get_llm_inference_summary,
    get_cortex_analyst_summary,
    get_document_ai_credits,
    get_cortex_functions_total_credits,
    get_credits_by_function,
    get_credits_by_model,
    get_credits_by_warehouse,
    get_cortex_functions_query_history,
    get_cortex_analyst_requests_by_day,
    get_document_processing_metrics,
    get_document_processing_by_day,
    get_cortex_search_total_credits,
    get_cortex_search_by_service,
    get_cortex_search_by_day
)

# Page configuration
st.set_page_config(layout="wide")

# Title and description
st.title("🏔️ Snowflake AI Cost Toolkit Dashboard")
st.markdown(
    "**Comprehensive analysis of Snowflake AI Services costs and Cortex Analyst usage**")
st.divider()

# Initialize session
session = get_session()

# =====================================================
# DATE RANGE SELECTION
# =====================================================
st.markdown("### 📅 Select Analysis Period")

# Date range buttons and inputs
col1, col2, col3, col4, col5 = st.columns(5)

# Initialize session state for dates
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime.datetime.now() - datetime.timedelta(days=30)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.datetime.now()

# Quick date range buttons
if col1.button('7 Days'):
    st.session_state.start_date = datetime.datetime.now() - datetime.timedelta(days=7)
    st.session_state.end_date = datetime.datetime.now()
if col2.button('30 Days'):
    st.session_state.start_date = datetime.datetime.now() - datetime.timedelta(days=30)
    st.session_state.end_date = datetime.datetime.now()
if col3.button('60 Days'):
    st.session_state.start_date = datetime.datetime.now() - datetime.timedelta(days=60)
    st.session_state.end_date = datetime.datetime.now()
if col4.button('90 Days'):
    st.session_state.start_date = datetime.datetime.now() - datetime.timedelta(days=90)
    st.session_state.end_date = datetime.datetime.now()
if col5.button('1 Year'):
    st.session_state.start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    st.session_state.end_date = datetime.datetime.now()

# Date input with current session state
date_range = st.date_input(
    "Custom Date Range",
    value=(st.session_state.start_date.date(),
           st.session_state.end_date.date()),
    max_value=datetime.date.today()
)

# Update session state if date range is changed
if len(date_range) == 2:
    st.session_state.start_date = datetime.datetime.combine(
        date_range[0], datetime.time.min)
    st.session_state.end_date = datetime.datetime.combine(
        date_range[1], datetime.time.max)

start_date = st.session_state.start_date
end_date = st.session_state.end_date

st.divider()

# =====================================================
# AI SERVICES OVERVIEW
# =====================================================
st.markdown("## 🤖 AI Services Overview")

# Cache the overview data


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_overview_metrics(start_date, end_date):
    try:
        ai_total = get_ai_services_total_credits(session, start_date, end_date)
        llm_summary = get_llm_inference_summary(session, start_date, end_date)
        ca_summary = get_cortex_analyst_summary(session, start_date, end_date)
        doc_ai = get_document_ai_credits(session, start_date, end_date)
        cortex_functions = get_cortex_functions_total_credits(
            session, start_date, end_date)
        search_credits = get_cortex_search_total_credits(
            session, start_date, end_date)

        return ai_total, llm_summary, ca_summary, doc_ai, cortex_functions, search_credits
    except Exception as e:
        st.error(f"Error fetching overview metrics: {e}")
        return None, None, None, None, None, None


ai_total, llm_summary, ca_summary, doc_ai, cortex_functions, search_credits = get_overview_metrics(
    start_date, end_date)

if ai_total is not None:
    # Top-level metrics
    col1, col2, col3 = st.columns(3)

    total_ai_credits = ai_total.iloc[0]['TOTAL_CREDITS'] if len(
        ai_total) > 0 and ai_total.iloc[0]['TOTAL_CREDITS'] is not None else 0
    llm_credits = llm_summary.iloc[0]['LLM_INFERENCE_CREDITS'] if len(
        llm_summary) > 0 and llm_summary.iloc[0]['LLM_INFERENCE_CREDITS'] is not None else 0
    llm_tokens = llm_summary.iloc[0]['LLM_INFERENCE_TOKENS'] if len(
        llm_summary) > 0 and llm_summary.iloc[0]['LLM_INFERENCE_TOKENS'] is not None else 0

    col1.metric("Total AI Services Credits", f"{total_ai_credits:,}")
    col2.metric("LLM Inference Credits", f"{llm_credits:,}")
    col3.metric("LLM Tokens Used", f"{llm_tokens:,}")

    # Service breakdown metrics
    st.markdown("### Service Breakdown")
    col1, col2, col3, col4 = st.columns(4)

    ca_credits = ca_summary.iloc[0]['CORTEX_ANALYST_CREDITS'] if len(
        ca_summary) > 0 and ca_summary.iloc[0]['CORTEX_ANALYST_CREDITS'] is not None else 0
    ca_messages = ca_summary.iloc[0]['NUMBER_MESSAGES'] if len(
        ca_summary) > 0 and ca_summary.iloc[0]['NUMBER_MESSAGES'] is not None else 0
    doc_credits = doc_ai.iloc[0]['TOTAL_CREDITS'] if len(
        doc_ai) > 0 and doc_ai.iloc[0]['TOTAL_CREDITS'] is not None else 0
    cf_credits = cortex_functions.iloc[0]['TOTAL_CREDITS'] if len(
        cortex_functions) > 0 and cortex_functions.iloc[0]['TOTAL_CREDITS'] is not None else 0
    cs_credits = search_credits.iloc[0]['CORTEX_SEARCH_CREDITS'] if len(
        search_credits) > 0 and search_credits.iloc[0]['CORTEX_SEARCH_CREDITS'] is not None else 0

    col1.metric("Cortex Analyst Credits", f"{ca_credits:,}")
    col2.metric("Document AI Credits", f"{doc_credits:,}")
    col3.metric("Cortex Functions Credits", f"{cf_credits:,}")
    col4.metric("Cortex Search Credits", f"{cs_credits:,}")

    # Service breakdown chart
    services_data = {
        'Service': ['Cortex Analyst', 'Document AI', 'Cortex Functions', 'Cortex Search'],
        'Credits': [ca_credits, doc_credits, cf_credits, cs_credits]
    }
    services_df = pd.DataFrame(services_data)
    # Filter out zero values
    services_df = services_df[services_df['Credits'] > 0]

    if not services_df.empty:
        st.markdown("#### AI Services Credit Distribution")
        services_chart = services_df.set_index('Service')
        st.bar_chart(services_chart['Credits'])

st.divider()

# =====================================================
# CORTEX FUNCTIONS ANALYSIS
# =====================================================
st.markdown("## 🧠 Cortex Functions Analysis")


@st.cache_data(ttl=300)
def get_cortex_functions_data(start_date, end_date):
    try:
        credits_by_function = get_credits_by_function(
            session, start_date, end_date)
        credits_by_model = get_credits_by_model(session, start_date, end_date)
        credits_by_warehouse = get_credits_by_warehouse(
            session, start_date, end_date)
        return credits_by_function, credits_by_model, credits_by_warehouse
    except Exception as e:
        st.error(f"Error fetching Cortex Functions data: {e}")
        return None, None, None


credits_by_function, credits_by_model, credits_by_warehouse = get_cortex_functions_data(
    start_date, end_date)

if credits_by_function is not None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Credits by Function")
        if not credits_by_function.empty:
            func_chart = credits_by_function.set_index('FUNCTION_NAME')
            st.bar_chart(func_chart['TOTAL_CREDITS'])
        else:
            st.info("No function data available")

    with col2:
        st.markdown("##### Credits by Model")
        if not credits_by_model.empty:
            model_chart = credits_by_model.set_index('MODEL_NAME')
            st.bar_chart(model_chart['TOTAL_CREDITS_USED'])
        else:
            st.info("No model data available")

    with col3:
        st.markdown("##### Credits by Warehouse")
        if not credits_by_warehouse.empty:
            wh_chart = credits_by_warehouse.set_index('WAREHOUSE_NAME')
            st.bar_chart(wh_chart['CORTEX_COMPLETE_CREDITS'])
        else:
            st.info("No warehouse data available")

st.divider()

# =====================================================
# CORTEX ANALYST DETAILED ANALYSIS
# =====================================================
st.markdown("## 🔍 Cortex Analyst Detailed Analysis")

# Fetch semantic models


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_semantic_models():
    try:
        return fetch_semantic_model_paths(session)
    except Exception as e:
        st.error(f"Error fetching semantic models: {e}")
        return pd.DataFrame()


semantic_models_df = get_semantic_models()

if not semantic_models_df.empty:
    st.markdown("### Available Semantic Models")
    st.dataframe(semantic_models_df, use_container_width=True)

    # Multi-select for semantic models
    selected_models = st.multiselect(
        "Select Semantic Models for Analysis",
        options=semantic_models_df[semantic_models_df['semantic_model_file'].notna(
        )]['semantic_model_file'].tolist(),
        default=semantic_models_df[semantic_models_df['semantic_model_file'].notna(
        )]['semantic_model_file'].tolist()[:3] if len(semantic_models_df) > 0 else []
    )

    if selected_models:
        st.markdown("### Cortex Analyst Usage Analysis")

        # Aggregate data from selected models
        all_logs = pd.DataFrame()

        for model_file in selected_models:
            try:
                logs = get_cortex_analyst_logs(session, model_file)
                all_logs = pd.concat([all_logs, logs], ignore_index=True)
            except Exception as e:
                st.warning(f"Could not fetch logs for {model_file}: {e}")

        if not all_logs.empty:
            # Analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📊 Summary", "⚡ Performance", "👥 User Activity", "💰 Costs"])

            with tab1:
                st.markdown("#### Query Type Analysis")
                vq_summary = verified_query_count(all_logs)
                st.dataframe(vq_summary, use_container_width=True)

                if not vq_summary.empty:
                    vq_chart = vq_summary.pivot(
                        index='SEMANTIC_MODEL_NAME', columns='QUERY_TYPE', values='request_count').fillna(0)
                    st.bar_chart(vq_chart)

                st.markdown("#### Top Verified Queries")
                top_vq = top_verified_queries(all_logs)
                st.dataframe(top_vq, use_container_width=True)

            with tab2:
                st.markdown("#### Latency Analysis")
                latency_stats = latency_summary_by_semantic_model(all_logs)
                st.dataframe(latency_stats, use_container_width=True)

                st.markdown("#### Slowest Queries")
                slow_queries = slowest_queries(all_logs, 10)
                st.dataframe(slow_queries, use_container_width=True)

            with tab3:
                st.markdown("#### User Activity by Semantic Model")
                user_activity = user_activity_by_semantic_model(all_logs)
                st.dataframe(user_activity, use_container_width=True)

                st.markdown("#### Semantic Model Usage Summary")
                model_summary = semantic_model_usage_summary(all_logs)
                st.dataframe(model_summary, use_container_width=True)

            with tab4:
                try:
                    # Try to create joined query history for cost analysis
                    ca_query_history = create_cortex_analyst_query_history(
                        session, all_logs)

                    if not ca_query_history.empty:
                        st.markdown("#### Total Cost by Semantic Model")
                        total_costs = total_cost_by_semantic_model(
                            ca_query_history)
                        st.dataframe(total_costs, use_container_width=True)

                        # Cost chart
                        if not total_costs.empty:
                            cost_chart = total_costs.set_index(
                                'SEMANTIC_MODEL_NAME')
                            st.bar_chart(cost_chart['TOTAL_CREDITS_WH_AND_CA'])

                        st.markdown("#### Detailed Cost Breakdown")
                        cost_breakdown = cost_breakdown_by_semantic_model(
                            ca_query_history)
                        st.dataframe(cost_breakdown, use_container_width=True)
                    else:
                        st.info(
                            "No cost data available. Ensure the SF_INTELLIGENCE_QUERY_HISTORY table is populated.")
                except Exception as e:
                    st.warning(f"Cost analysis unavailable: {e}")
                    st.info(
                        "To enable cost analysis, run the setup.sql script and populate the SF_INTELLIGENCE_QUERY_HISTORY table.")
        else:
            st.info(
                "No Cortex Analyst logs found for the selected models and date range.")
    else:
        st.info("Please select semantic models to analyze.")
else:
    st.info(
        "No semantic models found. Ensure you have Cortex Agents set up in your account.")

st.divider()

# =====================================================
# TIME SERIES ANALYSIS
# =====================================================
st.markdown("## 📈 Time Series Analysis")


@st.cache_data(ttl=300)
def get_time_series_data(start_date, end_date):
    try:
        ca_by_day = get_cortex_analyst_requests_by_day(
            session, start_date, end_date)
        doc_by_day = get_document_processing_by_day(
            session, start_date, end_date)
        search_by_day = get_cortex_search_by_day(session, start_date, end_date)
        return ca_by_day, doc_by_day, search_by_day
    except Exception as e:
        st.error(f"Error fetching time series data: {e}")
        return None, None, None


ca_by_day, doc_by_day, search_by_day = get_time_series_data(
    start_date, end_date)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Cortex Analyst Requests by Day")
    if ca_by_day is not None and not ca_by_day.empty:
        ca_chart = ca_by_day.set_index('DAY')
        st.line_chart(ca_chart['TOTAL_REQUEST_COUNT'])
    else:
        st.info("No daily Cortex Analyst data available")

with col2:
    st.markdown("#### Document Processing by Day")
    if doc_by_day is not None and not doc_by_day.empty:
        doc_chart = doc_by_day.set_index('DAY')
        st.line_chart(doc_chart['DAILY_CREDITS'])
    else:
        st.info("No daily document processing data available")

# Search by day (full width)
st.markdown("#### Cortex Search Credits by Day")
if search_by_day is not None and not search_by_day.empty:
    search_chart = search_by_day.set_index('DAY')
    st.line_chart(search_chart['DAILY_CREDITS'])
else:
    st.info("No daily Cortex Search data available")

st.divider()

# =====================================================
# DETAILED DATA EXPLORER
# =====================================================
st.markdown("## 🔎 Detailed Data Explorer")

explorer_tab1, explorer_tab2, explorer_tab3 = st.tabs(
    ["🛠️ Cortex Functions", "📄 Document Processing", "🔍 Search Services"])

with explorer_tab1:
    if credits_by_function is not None:
        st.markdown("### Cortex Functions Breakdown")
        st.dataframe(credits_by_function, use_container_width=True)

        st.markdown("### Credits by Model")
        st.dataframe(credits_by_model, use_container_width=True)

        st.markdown("### Credits by Warehouse")
        st.dataframe(credits_by_warehouse, use_container_width=True)

with explorer_tab2:
    doc_metrics = get_document_processing_metrics(
        session, start_date, end_date)
    if doc_metrics is not None and not doc_metrics.empty:
        st.markdown("### Document Processing Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Credits",
                    f"{doc_metrics.iloc[0]['TOTAL_CREDITS']:,.2f}")
        col2.metric("Total Pages", f"{doc_metrics.iloc[0]['TOTAL_PAGES']:,}")
        col3.metric("Total Documents",
                    f"{doc_metrics.iloc[0]['TOTAL_DOCUMENTS']:,}")

        if doc_by_day is not None and not doc_by_day.empty:
            st.markdown("### Daily Document Processing Details")
            st.dataframe(doc_by_day, use_container_width=True)

with explorer_tab3:
    search_by_service = get_cortex_search_by_service(
        session, start_date, end_date)
    if search_by_service is not None and not search_by_service.empty:
        st.markdown("### Cortex Search by Service")
        st.dataframe(search_by_service, use_container_width=True)

        service_chart = search_by_service.set_index('SERVICE_NAME')
        st.bar_chart(service_chart['TOTAL_CREDITS'])

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.caption("Snowflake AI Cost Toolkit")
with col2:
    st.caption("Version 1.0")
with col3:
    st.caption("August 2025")
