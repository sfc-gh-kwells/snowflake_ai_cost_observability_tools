import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

try:
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
except:
    from snowflake.snowpark import Session
    import os
    session = Session.builder.config('connection_name', os.getenv(
        "SNOWFLAKE_CONNECTION_NAME") or "MY_DEMO").create()

CREDIT_RATES_AGENT = {
    'claude-3-5-sonnet': {'input': 1.88, 'output': 9.41, 'cache_write': 0.0, 'cache_read': 0.0},
    'claude-3-7-sonnet': {'input': 1.88, 'output': 9.41, 'cache_write': 2.35, 'cache_read': 0.19},
    'claude-4-sonnet': {'input': 1.88, 'output': 9.41, 'cache_write': 2.35, 'cache_read': 0.19},
    'claude-haiku-4-54': {'input': 0.69, 'output': 3.45, 'cache_write': 0.87, 'cache_read': 0.07},
    'claude-4-5-sonnet': {'input': 2.07, 'output': 10.36, 'cache_write': 2.59, 'cache_read': 0.21},
    'openai-gpt-4.1': {'input': 1.38, 'output': 5.52, 'cache_write': 0.0, 'cache_read': 0.35},
    'openai-gpt-54': {'input': 0.86, 'output': 6.90, 'cache_write': 0.0, 'cache_read': 0.09},
}

CREDIT_RATES_SI = {
    'claude-3-5-sonnet': {'input': 2.51, 'output': 12.55, 'cache_write': 0.0, 'cache_read': 0.0},
    'claude-3-7-sonnet': {'input': 2.51, 'output': 12.55, 'cache_write': 3.14, 'cache_read': 0.25},
    'claude-4-sonnet': {'input': 2.51, 'output': 12.55, 'cache_write': 3.14, 'cache_read': 0.25},
    'claude-haiku-4-54': {'input': 0.92, 'output': 4.60, 'cache_write': 1.15, 'cache_read': 0.09},
    'claude-4-5-sonnet': {'input': 2.76, 'output': 13.81, 'cache_write': 3.45, 'cache_read': 0.28},
    'openai-gpt-4.1': {'input': 1.84, 'output': 7.36, 'cache_write': 0.0, 'cache_read': 0.46},
    'openai-gpt-54': {'input': 1.15, 'output': 9.21, 'cache_write': 0.0, 'cache_read': 0.12},
}


@st.cache_data(ttl=300)
def get_agents():
    try:
        session.sql("SHOW AGENTS IN ACCOUNT").collect()
        result = session.sql("""
            SELECT "name" as agent_name, "database_name" as db_name, "schema_name" as sch_name
            FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))
        """).to_pandas()
        return result
    except Exception as e:
        st.error(f"Error fetching agents: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_agent_events(db_name: str, agent_name: str, days_back: int = 30):
    try:
        query = f"""
            SELECT 
                DATE(TO_TIMESTAMP(TIMESTAMP)) AS event_date,
                RESOURCE_ATTRIBUTES:"snow.user.name"::STRING AS user_name,
                RECORD_ATTRIBUTES:"snow.ai.observability.object.name"::STRING AS agent_name,
                RECORD_ATTRIBUTES:"snow.ai.observability.agent.planning.model"::STRING AS model_name,
                RECORD_ATTRIBUTES:"snow.ai.observability.agent.planning.token_count.input"::NUMBER AS input_tokens,
                RECORD_ATTRIBUTES:"snow.ai.observability.agent.planning.token_count.output"::NUMBER AS output_tokens,
                RECORD_ATTRIBUTES:"snow.ai.observability.agent.planning.token_count.cache_creation"::NUMBER AS cache_write_tokens,
                RECORD_ATTRIBUTES:"snow.ai.observability.agent.planning.token_count.cache_read"::NUMBER AS cache_read_tokens,
                RECORD_ATTRIBUTES:"snow.ai.observability.agent.planning.token_count.total"::NUMBER AS total_tokens
            FROM TABLE(SNOWFLAKE.LOCAL.GET_AI_OBSERVABILITY_EVENTS(
                '{db_name}',
                'agents',
                '{agent_name}',
                'CORTEX AGENT'
            ))
            WHERE DATE(TO_TIMESTAMP(TIMESTAMP)) >= DATEADD(day, -{days_back}, CURRENT_DATE())
        """
        return session.sql(query).to_pandas()
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_analyst_query_costs(db_name: str, agent_name: str, days_back: int = 30):
    try:
        query = f"""
            WITH analyst_queries AS (
                SELECT DISTINCT
                    RECORD_ATTRIBUTES:"snow.ai.observability.agent.tool.sql_execution.query_id"::STRING AS query_id,
                    RESOURCE_ATTRIBUTES:"db.user"::STRING AS user_name,
                    RECORD_ATTRIBUTES:"snow.ai.observability.object.name"::STRING AS agent_name,
                    DATE(TO_TIMESTAMP(TIMESTAMP)) AS event_date
                FROM TABLE(SNOWFLAKE.LOCAL.GET_AI_OBSERVABILITY_EVENTS(
                    '{db_name}',
                    'agents',
                    '{agent_name}',
                    'CORTEX AGENT'
                ))
                WHERE RECORD:"name"::STRING = 'SqlExecution_CortexAnalyst'
                  AND RECORD_ATTRIBUTES:"snow.ai.observability.agent.tool.sql_execution.query_id" IS NOT NULL
                  AND DATE(TO_TIMESTAMP(TIMESTAMP)) >= DATEADD(day, -{days_back}, CURRENT_DATE())
            )
            SELECT 
                aq.query_id,
                aq.user_name,
                aq.agent_name,
                aq.event_date,
                qh.CREDITS_USED_CLOUD_SERVICES AS query_credits,
                qh.TOTAL_ELAPSED_TIME / 1000.0 AS elapsed_seconds,
                qh.WAREHOUSE_NAME,
                qh.WAREHOUSE_SIZE
            FROM analyst_queries aq
            LEFT JOIN SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY qh
                ON aq.query_id = qh.QUERY_ID
        """
        return session.sql(query).to_pandas()
    except Exception as e:
        return pd.DataFrame()


def calculate_credits(df: pd.DataFrame, pricing_type: str = "SI") -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    rates_table = CREDIT_RATES_SI if pricing_type == "SI" else CREDIT_RATES_AGENT

    df['INPUT_CREDITS'] = 0.0
    df['OUTPUT_CREDITS'] = 0.0
    df['CACHE_WRITE_CREDITS'] = 0.0
    df['CACHE_READ_CREDITS'] = 0.0

    for model, rates in rates_table.items():
        mask = df['MODEL_NAME'] == model
        df.loc[mask, 'INPUT_CREDITS'] = (
            df.loc[mask, 'INPUT_TOKENS'].fillna(0) / 1_000_000) * rates['input']
        df.loc[mask, 'OUTPUT_CREDITS'] = (
            df.loc[mask, 'OUTPUT_TOKENS'].fillna(0) / 1_000_000) * rates['output']
        df.loc[mask, 'CACHE_WRITE_CREDITS'] = (
            df.loc[mask, 'CACHE_WRITE_TOKENS'].fillna(0) / 1_000_000) * rates['cache_write']
        df.loc[mask, 'CACHE_READ_CREDITS'] = (
            df.loc[mask, 'CACHE_READ_TOKENS'].fillna(0) / 1_000_000) * rates['cache_read']

    df['TOKEN_CREDITS'] = df['INPUT_CREDITS'] + df['OUTPUT_CREDITS'] + \
        df['CACHE_WRITE_CREDITS'] + df['CACHE_READ_CREDITS']
    return df


def main():
    st.title("Cortex Agent Cost Analyzer")
    st.caption(
        "Analyze Snowflake Intelligence and Agent API costs by user and agent")

    with st.spinner("Fetching agents..."):
        agents_df = get_agents()

    if agents_df.empty:
        st.warning("No agents found in account or insufficient permissions.")
        return

    st.sidebar.header("Filters")

    pricing_type = st.sidebar.radio(
        "Pricing Table",
        options=["SI", "Agent"],
        index=0,
        format_func=lambda x: "Snowflake Intelligence (Table 6d)" if x == "SI" else "Cortex Agents API (Table 6e)",
        help="Select which credit rate table to use for cost calculations. Note: We are not looking at the app origin to differentiate the logs pricing. Please, verify against the account usage views."
    )

    time_range = st.sidebar.select_slider(
        "Time Range",
        options=[7, 14, 30, 60, 90],
        value=30,
        format_func=lambda x: f"Last {x} days"
    )

    selected_agents = st.sidebar.multiselect(
        "Select Agents",
        options=agents_df['AGENT_NAME'].tolist(),
        default=agents_df['AGENT_NAME'].tolist()
    )

    all_events = []
    all_query_costs = []
    progress = st.progress(0)

    for idx, row in agents_df.iterrows():
        if row['AGENT_NAME'] in selected_agents:
            events = get_agent_events(
                row['DB_NAME'], row['AGENT_NAME'], time_range)
            if not events.empty:
                all_events.append(events)
            query_costs = get_analyst_query_costs(
                row['DB_NAME'], row['AGENT_NAME'], time_range)
            if not query_costs.empty:
                all_query_costs.append(query_costs)
        progress.progress((idx + 1) / len(agents_df))

    progress.empty()

    if not all_events:
        st.info(
            f"No usage data found for selected agents in the past {time_range} days.")
        return

    combined_df = pd.concat(all_events, ignore_index=True)
    combined_df = calculate_credits(combined_df, pricing_type)

    query_costs_df = pd.concat(
        all_query_costs, ignore_index=True) if all_query_costs else pd.DataFrame()

    users = combined_df['USER_NAME'].dropna().unique().tolist()
    selected_users = st.sidebar.multiselect(
        "Select Users",
        options=users,
        default=users
    )

    filtered_df = combined_df[combined_df['USER_NAME'].isin(selected_users)]

    if filtered_df.empty:
        st.info("No data available for the selected filters.")
        return

    total_token_credits = filtered_df['TOKEN_CREDITS'].sum()
    total_query_credits = query_costs_df['QUERY_CREDITS'].sum(
    ) if not query_costs_df.empty else 0
    total_credits = total_token_credits + total_query_credits

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Credits", f"{total_credits:.4f}")
    col2.metric("Token Credits", f"{total_token_credits:.4f}")
    col3.metric("Query Credits", f"{total_query_credits:.6f}")
    col4.metric("Unique Agents", filtered_df['AGENT_NAME'].nunique())

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Summary", "By User", "By Agent", "Daily Trend", "Detailed Data"])

    with tab1:
        st.subheader("Cost Summary")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Overall Totals")
            summary_data = {
                "Category": ["Token Credits", "Query Credits (Cortex Analyst)", "Total Credits"],
                "Credits": [total_token_credits, total_query_credits, total_credits]
            }
            st.dataframe(
                pd.DataFrame(summary_data),
                column_config={
                    "Category": "Cost Type",
                    "Credits": st.column_config.NumberColumn("Credits", format="%.6f"),
                },
                hide_index=True,
                use_container_width=True
            )

        with col_b:
            st.markdown("#### Token Breakdown")
            token_breakdown = {
                "Type": ["Input", "Output", "Cache Write", "Cache Read"],
                "Credits": [
                    filtered_df['INPUT_CREDITS'].sum(),
                    filtered_df['OUTPUT_CREDITS'].sum(),
                    filtered_df['CACHE_WRITE_CREDITS'].sum(),
                    filtered_df['CACHE_READ_CREDITS'].sum()
                ]
            }
            st.dataframe(
                pd.DataFrame(token_breakdown),
                column_config={
                    "Type": "Token Type",
                    "Credits": st.column_config.NumberColumn("Credits", format="%.6f"),
                },
                hide_index=True,
                use_container_width=True
            )

        st.markdown("#### Credits by Agent (Token + Query)")
        agent_token_credits = filtered_df.groupby(
            'AGENT_NAME')['TOKEN_CREDITS'].sum().reset_index()
        agent_token_credits.columns = ['AGENT_NAME', 'TOKEN_CREDITS']

        if not query_costs_df.empty:
            agent_query_credits = query_costs_df.groupby(
                'AGENT_NAME')['QUERY_CREDITS'].sum().reset_index()
            agent_query_credits.columns = ['AGENT_NAME', 'QUERY_CREDITS']
            agent_combined = agent_token_credits.merge(
                agent_query_credits, on='AGENT_NAME', how='left')
            agent_combined['QUERY_CREDITS'] = agent_combined['QUERY_CREDITS'].fillna(
                0)
        else:
            agent_combined = agent_token_credits.copy()
            agent_combined['QUERY_CREDITS'] = 0.0

        agent_combined['TOTAL_CREDITS'] = agent_combined['TOKEN_CREDITS'] + \
            agent_combined['QUERY_CREDITS']
        agent_combined = agent_combined.sort_values(
            'TOTAL_CREDITS', ascending=False)

        st.bar_chart(agent_combined.set_index('AGENT_NAME')
                     [['TOKEN_CREDITS', 'QUERY_CREDITS']])
        st.dataframe(
            agent_combined,
            column_config={
                "AGENT_NAME": "Agent",
                "TOKEN_CREDITS": st.column_config.NumberColumn("Token Credits", format="%.6f"),
                "QUERY_CREDITS": st.column_config.NumberColumn("Query Credits", format="%.6f"),
                "TOTAL_CREDITS": st.column_config.NumberColumn("Total Credits", format="%.6f"),
            },
            hide_index=True,
            use_container_width=True
        )

        if not query_costs_df.empty:
            st.markdown("#### Cortex Analyst Query Details")
            query_summary = query_costs_df.groupby('AGENT_NAME').agg({
                'QUERY_ID': 'count',
                'QUERY_CREDITS': 'sum',
                'ELAPSED_SECONDS': 'sum'
            }).reset_index()
            query_summary.columns = [
                'AGENT_NAME', 'QUERY_COUNT', 'TOTAL_QUERY_CREDITS', 'TOTAL_ELAPSED_SECONDS']
            st.dataframe(
                query_summary,
                column_config={
                    "AGENT_NAME": "Agent",
                    "QUERY_COUNT": st.column_config.NumberColumn("Query Count", format="%d"),
                    "TOTAL_QUERY_CREDITS": st.column_config.NumberColumn("Query Credits", format="%.6f"),
                    "TOTAL_ELAPSED_SECONDS": st.column_config.NumberColumn("Total Time (s)", format="%.2f"),
                },
                hide_index=True,
                use_container_width=True
            )

    with tab2:
        st.subheader("Credits by User")
        user_summary = filtered_df.groupby('USER_NAME').agg({
            'TOKEN_CREDITS': 'sum',
            'INPUT_TOKENS': 'sum',
            'OUTPUT_TOKENS': 'sum',
            'AGENT_NAME': 'count'
        }).rename(columns={'AGENT_NAME': 'REQUEST_COUNT'}).reset_index()
        user_summary = user_summary.sort_values(
            'TOKEN_CREDITS', ascending=False)

        st.bar_chart(user_summary.set_index('USER_NAME')['TOKEN_CREDITS'])
        st.dataframe(
            user_summary,
            column_config={
                "USER_NAME": "User",
                "TOKEN_CREDITS": st.column_config.NumberColumn("Token Credits", format="%.4f"),
                "INPUT_TOKENS": st.column_config.NumberColumn("Input Tokens", format="%d"),
                "OUTPUT_TOKENS": st.column_config.NumberColumn("Output Tokens", format="%d"),
                "REQUEST_COUNT": st.column_config.NumberColumn("Requests", format="%d"),
            },
            hide_index=True,
            use_container_width=True
        )

    with tab3:
        st.subheader("Credits by Agent")
        agent_summary = filtered_df.groupby('AGENT_NAME').agg({
            'TOKEN_CREDITS': 'sum',
            'INPUT_TOKENS': 'sum',
            'OUTPUT_TOKENS': 'sum',
            'USER_NAME': 'nunique',
            'MODEL_NAME': 'first'
        }).rename(columns={'USER_NAME': 'UNIQUE_USERS'}).reset_index()
        agent_summary = agent_summary.sort_values(
            'TOKEN_CREDITS', ascending=False)

        st.bar_chart(agent_summary.set_index('AGENT_NAME')['TOKEN_CREDITS'])
        st.dataframe(
            agent_summary,
            column_config={
                "AGENT_NAME": "Agent",
                "TOKEN_CREDITS": st.column_config.NumberColumn("Token Credits", format="%.4f"),
                "INPUT_TOKENS": st.column_config.NumberColumn("Input Tokens", format="%d"),
                "OUTPUT_TOKENS": st.column_config.NumberColumn("Output Tokens", format="%d"),
                "UNIQUE_USERS": st.column_config.NumberColumn("Unique Users", format="%d"),
                "MODEL_NAME": "Model",
            },
            hide_index=True,
            use_container_width=True
        )

    with tab4:
        st.subheader("Daily Credit Usage Trend")
        daily_summary = filtered_df.groupby('EVENT_DATE').agg({
            'TOKEN_CREDITS': 'sum',
            'INPUT_CREDITS': 'sum',
            'OUTPUT_CREDITS': 'sum',
            'USER_NAME': 'count'
        }).rename(columns={'USER_NAME': 'REQUEST_COUNT'}).reset_index()
        daily_summary = daily_summary.sort_values('EVENT_DATE')

        st.line_chart(daily_summary.set_index('EVENT_DATE')[
                      ['TOKEN_CREDITS', 'INPUT_CREDITS', 'OUTPUT_CREDITS']])
        st.dataframe(
            daily_summary,
            column_config={
                "EVENT_DATE": "Date",
                "TOKEN_CREDITS": st.column_config.NumberColumn("Token Credits", format="%.4f"),
                "INPUT_CREDITS": st.column_config.NumberColumn("Input Credits", format="%.4f"),
                "OUTPUT_CREDITS": st.column_config.NumberColumn("Output Credits", format="%.4f"),
                "REQUEST_COUNT": st.column_config.NumberColumn("Requests", format="%d"),
            },
            hide_index=True,
            use_container_width=True
        )

    with tab5:
        st.subheader("Detailed Event Data")

        user_agent_summary = filtered_df.groupby(['USER_NAME', 'AGENT_NAME', 'EVENT_DATE']).agg({
            'TOKEN_CREDITS': 'sum',
            'INPUT_TOKENS': 'sum',
            'OUTPUT_TOKENS': 'sum',
            'MODEL_NAME': 'first'
        }).reset_index()
        user_agent_summary = user_agent_summary.sort_values(
            ['EVENT_DATE', 'TOKEN_CREDITS'], ascending=[False, False])

        st.dataframe(
            user_agent_summary,
            column_config={
                "USER_NAME": "User",
                "AGENT_NAME": "Agent",
                "EVENT_DATE": "Date",
                "TOKEN_CREDITS": st.column_config.NumberColumn("Token Credits", format="%.6f"),
                "INPUT_TOKENS": st.column_config.NumberColumn("Input Tokens", format="%d"),
                "OUTPUT_TOKENS": st.column_config.NumberColumn("Output Tokens", format="%d"),
                "MODEL_NAME": "Model",
            },
            hide_index=True,
            use_container_width=True
        )

    st.divider()
    with st.expander("Credit Rates Reference (per 1M tokens)"):
        st.markdown("**Table 6(e): Cortex Agents**")
        agent_rates_df = pd.DataFrame([
            {'Model': model, 'Input': rates['input'], 'Output': rates['output'],
             'Cache Write': rates['cache_write'], 'Cache Read': rates['cache_read']}
            for model, rates in CREDIT_RATES_AGENT.items()
        ])
        st.dataframe(agent_rates_df, hide_index=True, use_container_width=True)

        st.markdown("**Table 6(d): Snowflake Intelligence**")
        si_rates_df = pd.DataFrame([
            {'Model': model, 'Input': rates['input'], 'Output': rates['output'],
             'Cache Write': rates['cache_write'], 'Cache Read': rates['cache_read']}
            for model, rates in CREDIT_RATES_SI.items()
        ])
        st.dataframe(si_rates_df, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
