"""
Snowflake AI Cost Toolkit Utilities

This module contains utility functions for analyzing Cortex Analyst logs and costs.

Copyright 2024 Snowflake AI Cost Toolkit Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import os
import configparser
import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session


def create_local_session(config_file="config.env", connection_name="connections.my_example_connection"):
    """
    Create a Snowpark session from a local config file for development/testing.

    Parameters
    ----------
    config_file : str, optional
        Path to the config file containing connection parameters.
    connection_name : str, optional
        Name of the connection section in the config file.

    Returns
    -------
    snowflake.snowpark.Session
        A Snowpark session for local development.
    """

    # Read the config file
    config = configparser.ConfigParser()
    config.read(config_file)

    if connection_name not in config:
        raise ValueError(
            f"Connection '{connection_name}' not found in {config_file}")

    # Extract connection parameters and strip quotes - simplified approach
    conn_params = {
        "account": config[connection_name]["account"].strip('"'),
        "user": config[connection_name]["user"].strip('"'),
        "role": config[connection_name]["role"].strip('"'),
        "warehouse": config[connection_name]["warehouse"].strip('"'),
        "database": config[connection_name]["database"].strip('"'),
        "schema": config[connection_name]["schema"].strip('"')
    }

    # Add password if it exists - keep it simple
    if "password" in config[connection_name]:
        conn_params["password"] = config[connection_name]["password"].strip(
            '"')

    # Create session directly without complex auth logic
    session = Session.builder.configs(conn_params).create()
    print(f"✅ Local Snowpark session created successfully")
    print(f"   Account: {conn_params['account']}")
    print(f"   User: {conn_params['user']}")
    print(f"   Role: {conn_params['role']}")
    print(f"   Warehouse: {conn_params['warehouse']}")
    print(f"   Database: {conn_params['database']}")
    print(f"   Schema: {conn_params['schema']}")

    return session


def get_session(config_file="config.env", connection_name="connections.my_example_connection"):
    """
    Get a Snowpark session - either from active session (Snowflake environment) 
    or create a local session (development environment).

    Parameters
    ----------
    config_file : str, optional
        Path to the config file for local development.
    connection_name : str, optional
        Name of the connection section in the config file.

    Returns
    -------
    snowflake.snowpark.Session
        A Snowpark session.
    """

    try:
        # Try to get active session (works in Snowflake Notebooks/Streamlit)
        session = get_active_session()
        print("✅ Using active Snowflake session")
        return session
    except:
        # Fall back to local session for development
        print("🔧 No active session found, creating local session for development...")
        return create_local_session(config_file, connection_name)


def fetch_semantic_model_paths(session):
    """
    Fetch semantic model paths from all agents in the schema.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session

    Returns
    -------
    pandas.DataFrame
        DataFrame containing agent names, tool names, and semantic model file paths
    """
    try:
        # First get all agents - use collect() for SHOW commands
        agents_result = session.sql(
            "SHOW AGENTS IN SCHEMA snowflake_intelligence.agents").collect()

        if not agents_result:
            print("⚠️  No agents found in SNOWFLAKE_INTELLIGENCE.AGENTS schema")
            return pd.DataFrame()

        results = []
        # Extract agent names from the result rows - typically the 'name' field is at index 1
        agent_names = [row[1] for row in agents_result]

        for agent in agent_names:
            try:
                describe_sql = f'DESCRIBE AGENT SNOWFLAKE_INTELLIGENCE.AGENTS."{agent}"'
                describe_result = session.sql(describe_sql).collect()

                if not describe_result:
                    continue

                # The agent spec is typically at index 6 (7th column) in DESCRIBE output
                agent_spec_json = describe_result[0][6] if len(
                    describe_result[0]) > 6 else None

                if agent_spec_json:
                    try:
                        spec = json.loads(agent_spec_json)

                        if 'tool_resources' in spec:
                            for tool_name, tool_data in spec['tool_resources'].items():
                                semantic_file = tool_data.get(
                                    'semantic_model_file')
                                results.append({
                                    "agent_name": agent,
                                    "tool_name": tool_name,
                                    "semantic_model_file": semantic_file
                                })
                    except json.JSONDecodeError as je:
                        print(
                            f"⚠️  Could not parse agent spec JSON for {agent}: {je}")
                        continue
            except Exception as e:
                print(f"⚠️  Could not describe agent {agent}: {e}")
                continue

    except Exception as e:
        print(
            f"⚠️  Could not access SNOWFLAKE_INTELLIGENCE.AGENTS schema: {e}")
        print("This may be because:")
        print("  • No Cortex Agents are configured in your account")
        print("  • Your role doesn't have access to SNOWFLAKE_INTELLIGENCE schema")
        print("  • You need USAGE privilege on SNOWFLAKE_INTELLIGENCE.AGENTS schema")
        return pd.DataFrame()

    # Convert to DataFrame for display
    df_results = pd.DataFrame(results)
    return df_results


def get_cortex_analyst_logs(session, semantic_model_file):
    """
    Retrieve Cortex Analyst logs for a specific semantic model file.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    semantic_model_file : str
        Path to the semantic model file

    Returns
    -------
    pandas.DataFrame
        DataFrame containing Cortex Analyst logs with additional computed columns
    """
    cortex_analyst_log = session.sql(f'''SELECT 
          timestamp,
          request_id,
          semantic_model_name,
          tables_referenced,
          user_name,
          source,
          feedback,
          response_status_code,
          request_body:messages[0].content[0].text::STRING as user_question,
          response_body:response_metadata.analyst_latency_ms::NUMBER as latency_ms,
          generated_sql,
          response_body:response_metadata.analyst_orchestration_path::STRING as orchestration_path,
          response_body:response_metadata.question_category::STRING as question_category,
          response_body:message.content[1].confidence.verified_query_used.name::STRING as verified_query_name,
         response_body:message.content[1].confidence.verified_query_used.question::STRING as verified_query_question
        FROM TABLE(
          SNOWFLAKE.LOCAL.CORTEX_ANALYST_REQUESTS('FILE_ON_STAGE', '{semantic_model_file}'))''')

    df = cortex_analyst_log.to_pandas()
    df["QUERY_TYPE"] = df["ORCHESTRATION_PATH"].apply(
        lambda x: "Verified Query" if x == "vqr_fast_path" else "Non-Verified Query"
    )
    df['CORTEX_ANALYST_CREDITS'] = 67/1000

    return df


def verified_query_count(df):
    """
    Analyze verified vs non-verified query breakdown by semantic model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Cortex Analyst logs

    Returns
    -------
    pandas.DataFrame
        DataFrame with query type breakdown and percentages by semantic model
    """
    # Breakdown by semantic model
    semantic_model_summary = (
        df.groupby(['SEMANTIC_MODEL_NAME', 'QUERY_TYPE'])
        .size()
        .reset_index(name="request_count")
    )

    # Calculate percentages within each semantic model
    semantic_model_summary["percentage"] = (
        semantic_model_summary
        .groupby('SEMANTIC_MODEL_NAME')['request_count']
        .transform(lambda x: round((x * 100.0) / x.sum(), 2))
    )

    # Sort by semantic model and request count
    semantic_model_summary = semantic_model_summary.sort_values(
        ['SEMANTIC_MODEL_NAME', 'request_count'],
        ascending=[True, False]
    ).reset_index(drop=True)

    return semantic_model_summary


def top_verified_queries(df):
    """
    Get the top verified queries by frequency for each semantic model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Cortex Analyst logs

    Returns
    -------
    pandas.DataFrame
        DataFrame with top verified queries and their frequencies
    """
    # Breakdown by semantic model
    semantic_model_top = (
        df[df['QUERY_TYPE'] == 'Verified Query']
        .groupby(["SEMANTIC_MODEL_NAME", "VERIFIED_QUERY_NAME", "VERIFIED_QUERY_QUESTION"])
        .size()
        .reset_index(name="frequency")
    )

    # Calculate percentages within each semantic model
    semantic_model_top["percentage_of_verified_queries"] = (
        semantic_model_top
        .groupby('SEMANTIC_MODEL_NAME')['frequency']
        .transform(lambda x: round((x * 100.0) / x.sum(), 2))
    )

    # Sort by semantic model and frequency, take top 10 per model
    semantic_model_top = (
        semantic_model_top
        .sort_values(['SEMANTIC_MODEL_NAME', 'frequency'], ascending=[True, False])
        .groupby('SEMANTIC_MODEL_NAME')
        .head(10)
        .reset_index(drop=True)
    )

    return semantic_model_top


def slowest_queries(df, number=10):
    """
    Get the slowest queries by latency for each semantic model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Cortex Analyst logs
    number : int, optional
        Number of slowest queries to return per semantic model (default: 10)

    Returns
    -------
    pandas.DataFrame
        DataFrame with slowest queries and their latency information
    """
    # Filter out rows where latency_ms is null
    slow_queries = df[df["LATENCY_MS"].notnull()].copy()

    # Compute latency in seconds
    slow_queries["LATENCY_SECONDS"] = (
        slow_queries["LATENCY_MS"] / 1000.0).round(2)

    # Breakdown by semantic model - top slowest per model
    semantic_model_slow = slow_queries[[
        "SEMANTIC_MODEL_NAME",
        "USER_QUESTION",
        "LATENCY_SECONDS",
        "ORCHESTRATION_PATH",
        "QUESTION_CATEGORY"
    ]].sort_values(['SEMANTIC_MODEL_NAME', 'LATENCY_SECONDS'], ascending=[True, False])

    # Take top N slowest per semantic model
    semantic_model_slow = (
        semantic_model_slow
        .groupby('SEMANTIC_MODEL_NAME')
        .head(number)
        .reset_index(drop=True)
    )

    return semantic_model_slow


def latency_summary_by_semantic_model(df):
    """
    Provide latency statistics breakdown by semantic model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Cortex Analyst logs

    Returns
    -------
    pandas.DataFrame
        DataFrame with latency statistics by semantic model
    """
    # Filter out rows where latency_ms is null
    latency_data = df[df["LATENCY_MS"].notnull()].copy()

    # Compute latency in seconds
    latency_data["LATENCY_SECONDS"] = (
        latency_data["LATENCY_MS"] / 1000.0).round(2)

    # Stats by semantic model
    semantic_model_stats = (
        latency_data
        .groupby('SEMANTIC_MODEL_NAME')['LATENCY_SECONDS']
        .agg(['count', 'mean', 'median', 'min', 'max'])
        .round(2)
        .reset_index()
    )

    semantic_model_stats.columns = [
        'SEMANTIC_MODEL_NAME', 'query_count', 'avg_latency_seconds',
        'median_latency_seconds', 'min_latency_seconds', 'max_latency_seconds'
    ]

    return semantic_model_stats


def create_sf_intelligence_query_history(session, target_table="cortex_analytics.public.sf_intelligence_query_history"):
    """
    Create a table with Cortex Agent query history joined to query attribution.
    Keeps last 30 days of queries with compute credits > 0 and cleans query text for matching.
    Only creates the table if it doesn't exist to avoid expensive recreation.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session.
    target_table : str, optional
        Fully qualified name of the target table.
    """

    # Check if table exists first
    table_parts = target_table.split('.')
    table_name = table_parts[-1].upper()
    schema_name = table_parts[-2].upper() if len(table_parts) > 1 else 'PUBLIC'
    database_name = table_parts[-3].upper() if len(
        table_parts) > 2 else 'CORTEX_ANALYTICS'

    try:
        # Check if table exists
        result = session.sql(f"""
            SELECT COUNT(*) as table_count 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = '{schema_name}' 
            AND TABLE_NAME = '{table_name}' 
            AND TABLE_CATALOG = '{database_name}'
        """).collect()

        table_exists = result[0]['TABLE_COUNT'] > 0

        if table_exists:
            print(
                f"Table {target_table} already exists. Skipping creation to avoid expensive recreation.")
            print(
                "To refresh data, consider running: ALTER TABLE {target_table} DROP ALL ROWS; then re-run this function.")
            return

    except Exception as e:
        print(
            f"Could not check if table exists: {e}. Proceeding with creation...")

    query = f"""
    CREATE TABLE {target_table} AS
    SELECT 
        qh.query_id,
        qh.query_text,
        qh.start_time,
        qh.total_elapsed_time,
        qh.warehouse_name,
        qh.user_name,
        qah.credits_attributed_compute,
        -- Clean the generated SQL for matching
        TRIM(REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(qh.query_text, '--[^\\n]*\\n', '\\n'),
                '/\\*.*?\\*/', ' '
            ),
            '\\s+', ' '
        )) AS cleaned_query_text
    FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY qh
    JOIN SNOWFLAKE.ACCOUNT_USAGE.QUERY_ATTRIBUTION_HISTORY qah 
        ON qh.query_id = qah.query_id
    WHERE qh.query_tag = 'cortex-agent'
      AND qh.start_time >= DATEADD(DAY, -30, CURRENT_TIMESTAMP())
      AND qah.credits_attributed_compute > 0
    """

    session.sql(query).collect()
    print(f"✅ Table {target_table} created successfully.")


def refresh_sf_intelligence_query_history(session, target_table="cortex_analytics.public.sf_intelligence_query_history"):
    """
    Refresh the SF_INTELLIGENCE_QUERY_HISTORY table with the latest data.
    This function will truncate and reload the table with fresh data.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session.
    target_table : str, optional
        Fully qualified name of the target table.
    """

    print(f"🔄 Refreshing {target_table} with latest data...")

    # First, truncate the existing table
    try:
        session.sql(f"TRUNCATE TABLE {target_table}").collect()
        print(f"✅ Truncated existing data from {target_table}")
    except Exception as e:
        print(f"⚠️  Could not truncate table (may not exist): {e}")

    # Insert fresh data
    insert_query = f"""
    INSERT INTO {target_table}
    SELECT 
        qh.query_id,
        qh.query_text,
        qh.start_time,
        qh.total_elapsed_time,
        qh.warehouse_name,
        qh.user_name,
        qah.credits_attributed_compute,
        -- Clean the generated SQL for matching
        TRIM(REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(qh.query_text, '--[^\\n]*\\n', '\\n'),
                '/\\*.*?\\*/', ' '
            ),
            '\\s+', ' '
        )) AS cleaned_query_text
    FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY qh
    JOIN SNOWFLAKE.ACCOUNT_USAGE.QUERY_ATTRIBUTION_HISTORY qah 
        ON qh.query_id = qah.query_id
    WHERE qh.query_tag = 'cortex-agent'
      AND qh.start_time >= DATEADD(DAY, -30, CURRENT_TIMESTAMP())
      AND qah.credits_attributed_compute > 0
    """

    try:
        session.sql(insert_query).collect()
        print(
            f"✅ Table {target_table} refreshed successfully with latest data.")
    except Exception as e:
        print(f"❌ Error refreshing table: {e}")
        # If insert fails, try to recreate the table completely
        print("🔄 Attempting to recreate table...")
        session.sql(f"DROP TABLE IF EXISTS {target_table}").collect()
        create_sf_intelligence_query_history(session, target_table)


def user_activity_by_semantic_model(df):
    """
    Provide user activity analysis by semantic model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Cortex Analyst logs

    Returns
    -------
    pandas.DataFrame
        DataFrame with user activity broken down by semantic model
    """
    # User activity by semantic model
    user_model_activity = df.groupby(['USER_NAME', 'SEMANTIC_MODEL_NAME']).agg({
        'REQUEST_ID': 'count',
        'LATENCY_MS': 'mean',
        'CORTEX_ANALYST_CREDITS': 'sum'
    }).rename(columns={
        'REQUEST_ID': 'queries_count',
        'LATENCY_MS': 'avg_latency_ms',
        'CORTEX_ANALYST_CREDITS': 'total_credits'
    }).round(2).reset_index()

    # Add percentage of user's queries per model
    user_totals = user_model_activity.groupby(
        'USER_NAME')['queries_count'].sum()
    user_model_activity['percentage_of_user_queries'] = (
        user_model_activity.apply(
            lambda row: round(
                (row['queries_count'] / user_totals[row['USER_NAME']]) * 100, 2),
            axis=1
        )
    )

    # Sort by user name and query count
    user_model_activity = user_model_activity.sort_values(
        ['USER_NAME', 'queries_count'],
        ascending=[True, False]
    ).reset_index(drop=True)

    return user_model_activity


def semantic_model_usage_summary(df):
    """
    Provide overall semantic model usage summary.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Cortex Analyst logs

    Returns
    -------
    pandas.DataFrame
        DataFrame with usage statistics per semantic model
    """
    model_usage = df.groupby('SEMANTIC_MODEL_NAME').agg({
        'REQUEST_ID': 'count',
        'USER_NAME': 'nunique',
        'LATENCY_MS': 'mean',
        'CORTEX_ANALYST_CREDITS': 'sum'
    }).rename(columns={
        'REQUEST_ID': 'total_queries',
        'USER_NAME': 'unique_users',
        'LATENCY_MS': 'avg_latency_ms',
        'CORTEX_ANALYST_CREDITS': 'total_cortex_analyst_credits'
    }).round(2).reset_index()

    # Add percentage of total queries
    model_usage['percentage'] = round(
        (model_usage['total_queries'] /
         model_usage['total_queries'].sum()) * 100, 2
    )

    # Sort by total queries descending
    model_usage = model_usage.sort_values(
        'total_queries', ascending=False).reset_index(drop=True)

    return model_usage


def create_cortex_analyst_query_history(session, cortex_analyst_df, sf_intelligence_table="cortex_analytics.public.sf_intelligence_query_history"):
    """
    Create joined query history by merging Snowflake Intelligence query history 
    with Cortex Analyst logs on generated SQL.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    cortex_analyst_df : pandas.DataFrame
        DataFrame containing Cortex Analyst logs with GENERATED_SQL column
    sf_intelligence_table : str, optional
        Fully qualified name of the Snowflake Intelligence query history table

    Returns
    -------
    pandas.DataFrame
        Joined DataFrame with additional computed columns
    """
    import re

    # Load Snowflake Intelligence query history
    sql = f"SELECT * FROM {sf_intelligence_table}"
    pd_sfi_query_history = session.sql(sql).to_pandas()

    def normalize_sql(sql_text):
        """Normalize SQL text for better matching"""
        if pd.isna(sql_text) or sql_text is None:
            return ""

        # Convert to string and strip whitespace
        sql_str = str(sql_text).strip()

        # Handle corrupted/cleaned schema names - remove underscores and spaces from identifiers
        sql_str = re.sub(r'doc_ai_q\s*_db\.doc_ai_\s*chema',
                         'doc_ai_qs_db.doc_ai_schema', sql_str)
        sql_str = re.sub(r'co_branding_agreement\s+',
                         'co_branding_agreements ', sql_str)
        sql_str = re.sub(r'__co_branding_agreement\b',
                         '__co_branding_agreements', sql_str)

        # Fix common column name corruptions
        sql_str = re.sub(r'have_renewal_option\s*_value',
                         'have_renewal_options_value', sql_str)
        sql_str = re.sub(r'have_force_majeure\s*_value',
                         'have_force_majeure_value', sql_str)

        # Remove extra whitespace and normalize case
        sql_normalized = re.sub(r'\s+', ' ', sql_str.upper())

        # Remove trailing semicolons
        sql_normalized = sql_normalized.rstrip(';')

        # Remove common SQL variations that don't affect logic
        # Normalize AS clauses
        sql_normalized = re.sub(r'\bAS\s+(\w+)', r'AS \1', sql_normalized)

        return sql_normalized

    # Create normalized SQL columns for better matching
    pd_sfi_query_history_copy = pd_sfi_query_history.copy()
    cortex_analyst_copy = cortex_analyst_df.copy()

    pd_sfi_query_history_copy['NORMALIZED_SQL'] = pd_sfi_query_history_copy['CLEANED_QUERY_TEXT'].apply(
        normalize_sql)
    cortex_analyst_copy['NORMALIZED_SQL'] = cortex_analyst_copy['GENERATED_SQL'].apply(
        normalize_sql)

    # Remove empty normalized SQL entries
    pd_sfi_query_history_copy = pd_sfi_query_history_copy[
        pd_sfi_query_history_copy['NORMALIZED_SQL'] != ""]
    cortex_analyst_copy = cortex_analyst_copy[cortex_analyst_copy['NORMALIZED_SQL'] != ""]

    # Set indexes for joining on normalized SQL
    pd_sfi_query_history_indexed = pd_sfi_query_history_copy.set_index(
        "NORMALIZED_SQL")
    cortex_analyst_indexed = cortex_analyst_copy.set_index("NORMALIZED_SQL")

    # Debug: Print join statistics
    print(f"🔗 Join Debug Info:")
    print(
        f"   Query History records (after normalization): {len(pd_sfi_query_history_indexed)}")
    print(
        f"   Cortex Analyst records (after normalization): {len(cortex_analyst_indexed)}")
    print(
        f"   Unique normalized SQLs in Query History: {len(pd_sfi_query_history_indexed.index.unique())}")
    print(
        f"   Unique normalized SQLs in Cortex Logs: {len(cortex_analyst_indexed.index.unique())}")

    # Check for overlapping normalized SQL
    qh_sqls = set(pd_sfi_query_history_indexed.index)
    ca_sqls = set(cortex_analyst_indexed.index)
    overlap = qh_sqls.intersection(ca_sqls)
    print(f"   Overlapping normalized SQLs: {len(overlap)}")

    if len(overlap) > 0:
        print(f"   Sample overlapping SQL: {list(overlap)[0][:100]}...")

    # Join on normalized SQL
    ca_query_history = pd.merge(
        pd_sfi_query_history_indexed,
        cortex_analyst_indexed,
        left_index=True,
        right_index=True,
        how='inner'
    )

    # If no matches with normalized SQL, try fuzzy matching on first 100 characters
    if len(ca_query_history) == 0:
        print("   🔄 No exact matches found, trying fuzzy matching on SQL fragments...")

        # Create truncated SQL versions for fuzzy matching
        pd_sfi_query_history_copy['SQL_FRAGMENT'] = pd_sfi_query_history_copy['NORMALIZED_SQL'].apply(
            lambda x: x[:100] if x else "")
        cortex_analyst_copy['SQL_FRAGMENT'] = cortex_analyst_copy['NORMALIZED_SQL'].apply(
            lambda x: x[:100] if x else "")

        # Remove empty fragments
        pd_sfi_fuzzy = pd_sfi_query_history_copy[pd_sfi_query_history_copy['SQL_FRAGMENT'] != ""].set_index(
            'SQL_FRAGMENT')
        cortex_fuzzy = cortex_analyst_copy[cortex_analyst_copy['SQL_FRAGMENT'] != ""].set_index(
            'SQL_FRAGMENT')

        # Try fuzzy join
        ca_query_history = pd.merge(
            pd_sfi_fuzzy,
            cortex_fuzzy,
            left_index=True,
            right_index=True,
            how='inner'
        )

        if len(ca_query_history) > 0:
            print(
                f"   ✅ Found {len(ca_query_history)} matches using fuzzy SQL fragment matching!")
        else:
            # Try even more aggressive matching - just first 50 chars
            print("   🔄 Trying very aggressive matching (first 50 chars)...")
            pd_sfi_query_history_copy['SHORT_FRAGMENT'] = pd_sfi_query_history_copy['NORMALIZED_SQL'].apply(
                lambda x: x[:50] if x else "")
            cortex_analyst_copy['SHORT_FRAGMENT'] = cortex_analyst_copy['NORMALIZED_SQL'].apply(
                lambda x: x[:50] if x else "")

            pd_sfi_short = pd_sfi_query_history_copy[pd_sfi_query_history_copy['SHORT_FRAGMENT'] != ""].set_index(
                'SHORT_FRAGMENT')
            cortex_short = cortex_analyst_copy[cortex_analyst_copy['SHORT_FRAGMENT'] != ""].set_index(
                'SHORT_FRAGMENT')

            ca_query_history = pd.merge(
                pd_sfi_short,
                cortex_short,
                left_index=True,
                right_index=True,
                how='inner'
            )

            if len(ca_query_history) > 0:
                print(
                    f"   ✅ Found {len(ca_query_history)} matches using aggressive fragment matching!")
            else:
                print(
                    "   ⚠️  Still no matches found even with aggressive fuzzy matching")

    # Add computed columns
    ca_query_history['TOTAL_TIME'] = (
        ca_query_history["TOTAL_ELAPSED_TIME"] + ca_query_history["LATENCY_MS"]
    )

    ca_query_history['TOTAL_CREDITS_WH_AND_CA'] = (
        ca_query_history["CREDITS_ATTRIBUTED_COMPUTE"] +
        ca_query_history["CORTEX_ANALYST_CREDITS"]
    )

    # Reset index to make it a regular DataFrame
    ca_query_history = ca_query_history.reset_index()

    return ca_query_history


def write_logs_to_table(session, semantic_model_files, table_name="CORTEX_ANALYST_LOGS"):
    """
    Write Cortex Analyst logs for multiple semantic model files to a Snowflake table.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    semantic_model_files : list
        List of semantic model file paths
    table_name : str, optional
        Name of the target table (default: "CORTEX_ANALYST_LOGS")
    """
    for file in semantic_model_files:
        if file is not None:
            df = get_cortex_analyst_logs(session, file)

            # Push back into Snowflake (append to table)
            session.write_pandas(
                df,
                table_name=table_name,
                database=None,   # or specify your DB
                schema=None,     # or specify your schema
                auto_create_table=False,  # you already created the table
                overwrite=False
            )


def total_cost_by_semantic_model(ca_query_history_df):
    """
    Calculate total Cortex Analyst and warehouse costs by semantic model.

    Parameters
    ----------
    ca_query_history_df : pandas.DataFrame
        Joined DataFrame containing both Cortex Analyst logs and query attribution data
        (output from create_cortex_analyst_query_history function)

    Returns
    -------
    pandas.DataFrame
        DataFrame with total combined credits (Cortex Analyst + Warehouse) by semantic model
    """
    # Calculate total cost by semantic model
    total_cost_by_model = (
        ca_query_history_df[['SEMANTIC_MODEL_NAME', 'TOTAL_CREDITS_WH_AND_CA']]
        .groupby(['SEMANTIC_MODEL_NAME'])
        .sum()
        .round(4)
        .reset_index()
    )

    # Sort by total credits descending to show most expensive models first
    total_cost_by_model = total_cost_by_model.sort_values(
        'TOTAL_CREDITS_WH_AND_CA',
        ascending=False
    ).reset_index(drop=True)

    return total_cost_by_model


def cost_breakdown_by_semantic_model(ca_query_history_df):
    """
    Provide detailed cost breakdown showing separate Cortex Analyst and warehouse costs by semantic model.

    Parameters
    ----------
    ca_query_history_df : pandas.DataFrame
        Joined DataFrame containing both Cortex Analyst logs and query attribution data
        (output from create_cortex_analyst_query_history function)

    Returns
    -------
    pandas.DataFrame
        DataFrame with separate and total costs by semantic model
    """
    # Calculate detailed cost breakdown by semantic model
    cost_breakdown = ca_query_history_df.groupby('SEMANTIC_MODEL_NAME').agg({
        'CORTEX_ANALYST_CREDITS': 'sum',
        'CREDITS_ATTRIBUTED_COMPUTE': 'sum',
        'TOTAL_CREDITS_WH_AND_CA': 'sum',
        'REQUEST_ID': 'count'  # Number of queries
    }).rename(columns={
        'CORTEX_ANALYST_CREDITS': 'cortex_analyst_credits',
        'CREDITS_ATTRIBUTED_COMPUTE': 'warehouse_credits',
        'TOTAL_CREDITS_WH_AND_CA': 'total_credits',
        'REQUEST_ID': 'query_count'
    }).round(4).reset_index()

    # Calculate percentages
    total_all_credits = cost_breakdown['total_credits'].sum()
    cost_breakdown['percentage_of_total_cost'] = round(
        (cost_breakdown['total_credits'] / total_all_credits) * 100, 2
    )

    # Calculate average cost per query
    cost_breakdown['avg_credits_per_query'] = round(
        cost_breakdown['total_credits'] / cost_breakdown['query_count'], 4
    )

    # Sort by total credits descending
    cost_breakdown = cost_breakdown.sort_values(
        'total_credits',
        ascending=False
    ).reset_index(drop=True)

    return cost_breakdown


# =====================================================
# LLM Usage Dashboard Functions
# =====================================================
# These functions are extracted from the LLM Usage Dashboard
# Streamlit app for reusable analysis across applications

def get_ai_services_total_credits(session, start_date, end_date):
    """
    Get total AI Services credits used within a date range.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with total AI Services credits
    """
    sql = f"""
    SELECT ROUND(COALESCE(SUM(credits_used), 0), 2) AS total_credits 
    FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}' 
    AND SERVICE_TYPE = 'AI_SERVICES'
    """
    return session.sql(sql).to_pandas()


def get_ai_services_total_credits_by_days(session, days=30):
    """
    Convenience function: Get total AI Services credits used in the last N days.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    days : int, optional
        Number of days to look back (default: 30)

    Returns
    -------
    pandas.DataFrame
        DataFrame with total AI Services credits
    """
    sql = f"""
    SELECT ROUND(COALESCE(SUM(credits_used), 0), 2) AS total_credits 
    FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_HISTORY 
    WHERE start_time >= DATEADD(DAY, -{days}, CURRENT_TIMESTAMP()) 
    AND SERVICE_TYPE = 'AI_SERVICES'
    """
    return session.sql(sql).to_pandas()


def get_llm_inference_summary_by_days(session, days=7):
    """
    Get LLM inference summary for the last N days.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    days : int, optional
        Number of days to look back (default: 7)

    Returns
    -------
    pandas.DataFrame
        DataFrame with LLM inference credits and tokens
    """
    sql = f"""
    SELECT 
        ROUND(SUM(credits_used), 2) AS llm_inference_credits,
        ROUND(SUM(input_tokens + output_tokens), 0) AS llm_inference_tokens
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
    WHERE start_time >= DATEADD(DAY, -{days}, CURRENT_TIMESTAMP()) 
    AND function_name = 'COMPLETE'
    """
    return session.sql(sql).to_pandas()


def get_llm_inference_summary(session, start_date, end_date):
    """
    Get LLM inference credits and tokens summary for COMPLETE function.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with LLM inference credits and tokens
    """
    sql = f"""
    SELECT 
        ROUND(SUM(token_credits), 0) AS llm_inference_credits, 
        SUM(tokens) AS llm_inference_tokens 
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
    WHERE function_name = 'COMPLETE' 
    AND start_time BETWEEN '{start_date}' AND '{end_date}'
    """
    return session.sql(sql).to_pandas()


def get_cortex_analyst_summary_by_days(session, days=7):
    """
    Get Cortex Analyst summary for the last N days.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    days : int, optional
        Number of days to look back (default: 7)

    Returns
    -------
    pandas.DataFrame
        DataFrame with Cortex Analyst credits and message count
    """
    sql = f"""
    SELECT 
        ROUND(SUM(credits_attributed), 2) AS cortex_analyst_credits,
        COUNT(*) AS number_messages
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_ANALYST_USAGE_HISTORY 
    WHERE start_time >= DATEADD(DAY, -{days}, CURRENT_TIMESTAMP())
    """
    return session.sql(sql).to_pandas()


def get_cortex_analyst_summary(session, start_date, end_date):
    """
    Get Cortex Analyst credits and message count summary.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with Cortex Analyst credits and message count
    """
    sql = f"""
    SELECT 
        ROUND(SUM(credits), 0) AS cortex_analyst_credits, 
        SUM(request_count) AS number_messages 
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_ANALYST_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    """
    return session.sql(sql).to_pandas()


def get_document_ai_credits(session, start_date, end_date):
    """
    Get Document AI total credits used.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with Document AI credits
    """
    sql = f"""
    SELECT ROUND(SUM(credits_used), 2) AS total_credits 
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_DOCUMENT_PROCESSING_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    """
    return session.sql(sql).to_pandas()


def get_cortex_functions_total_credits(session, start_date, end_date):
    """
    Get total Cortex Functions credits across all functions.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with total Cortex Functions credits
    """
    sql = f"""
    SELECT ROUND(SUM(token_credits), 2) AS total_credits 
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    """
    return session.sql(sql).to_pandas()


def get_credits_by_function(session, start_date, end_date):
    """
    Get credits breakdown by Cortex function name.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with credits by function name
    """
    sql = f"""
    SELECT 
        DISTINCT(function_name), 
        ROUND(SUM(token_credits), 2) AS total_credits 
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}' 
    GROUP BY 1 
    ORDER BY 2 DESC
    """
    return session.sql(sql).to_pandas()


def get_credits_by_model(session, start_date, end_date, limit=10):
    """
    Get credits and tokens breakdown by model for COMPLETE function.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query
    limit : int, optional
        Maximum number of models to return (default: 10)

    Returns
    -------
    pandas.DataFrame
        DataFrame with credits and tokens by model
    """
    sql = f"""
    SELECT 
        model_name,
        SUM(token_credits) AS total_credits_used, 
        SUM(tokens) AS total_tokens_used 
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY 
    WHERE function_name = 'COMPLETE' 
    AND start_time BETWEEN '{start_date}' AND '{end_date}' 
    GROUP BY 1 
    ORDER BY 2 DESC 
    LIMIT {limit}
    """
    return session.sql(sql).to_pandas()


def get_credits_by_warehouse(session, start_date, end_date):
    """
    Get Cortex and compute credits breakdown by warehouse.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with Cortex and compute credits by warehouse
    """
    sql = f"""
    SELECT 
        warehouse_name,
        w.warehouse_id,  
        SUM(token_credits) AS cortex_complete_credits, 
        SUM(credits_used_compute) AS total_compute_credits 
    FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY AS w 
    JOIN SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY AS c 
        ON c.warehouse_id = w.warehouse_id 
    WHERE c.start_time BETWEEN '{start_date}' AND '{end_date}' 
    GROUP BY warehouse_name, w.warehouse_id 
    ORDER BY 3 DESC
    """
    return session.sql(sql).to_pandas()


def get_cortex_functions_query_history(session, start_date, end_date):
    """
    Get detailed Cortex functions query history with query details.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with detailed Cortex functions query history
    """
    sql = f"""
    SELECT 
        q.query_id, 
        model_name, 
        function_name, 
        tokens, 
        token_credits, 
        query_text, 
        user_name, 
        role_name, 
        total_elapsed_time 
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_QUERY_USAGE_HISTORY AS c 
    JOIN SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY AS q 
        ON c.query_id = q.query_id 
    WHERE q.start_time BETWEEN '{start_date}' AND '{end_date}'
    """
    return session.sql(sql).to_pandas()


def get_cortex_analyst_requests_by_day(session, start_date, end_date):
    """
    Get Cortex Analyst requests aggregated by day.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with daily Cortex Analyst request counts
    """
    sql = f"""
    SELECT 
        TO_DATE(cortex_analyst_usage_history.start_time) AS day,
        SUM(cortex_analyst_usage_history.request_count) AS total_request_count 
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_ANALYST_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}' 
    GROUP BY day 
    ORDER BY day
    """
    return session.sql(sql).to_pandas()


def get_document_processing_metrics(session, start_date, end_date):
    """
    Get Document Processing summary metrics.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with document processing metrics
    """
    sql = f"""
    SELECT 
        ROUND(SUM(credits_used), 2) AS total_credits,
        SUM(page_count) AS total_pages,
        SUM(document_count) AS total_documents
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_DOCUMENT_PROCESSING_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    """
    return session.sql(sql).to_pandas()


def get_document_processing_by_day(session, start_date, end_date):
    """
    Get Document Processing metrics aggregated by day.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with daily document processing metrics
    """
    sql = f"""
    SELECT 
        TO_DATE(start_time) AS day,
        ROUND(SUM(credits_used), 2) AS daily_credits,
        SUM(page_count) AS daily_pages,
        SUM(document_count) AS daily_documents
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_DOCUMENT_PROCESSING_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY TO_DATE(start_time)
    ORDER BY day
    """
    return session.sql(sql).to_pandas()


def get_cortex_search_total_credits(session, start_date, end_date):
    """
    Get total Cortex Search serving credits.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with total Cortex Search credits
    """
    sql = f"""
    SELECT ROUND(SUM(credits), 2) AS cortex_search_credits 
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_SERVING_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    """
    return session.sql(sql).to_pandas()


def get_cortex_search_by_service(session, start_date, end_date):
    """
    Get Cortex Search credits breakdown by service name.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with Cortex Search credits by service
    """
    sql = f"""
    SELECT 
        service_name, 
        SUM(credits) AS total_credits 
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_SERVING_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}' 
    GROUP BY service_name
    ORDER BY total_credits DESC
    """
    return session.sql(sql).to_pandas()


def get_cortex_search_by_day(session, start_date, end_date):
    """
    Get Cortex Search credits aggregated by day.

    Parameters
    ----------
    session : snowflake.snowpark.Session
        Active Snowpark session
    start_date : str or datetime
        Start date for the query
    end_date : str or datetime
        End date for the query

    Returns
    -------
    pandas.DataFrame
        DataFrame with daily Cortex Search credits
    """
    sql = f"""
    SELECT 
        TO_DATE(start_time) AS day,
        ROUND(SUM(credits), 2) AS daily_credits
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_SERVING_USAGE_HISTORY 
    WHERE start_time BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY TO_DATE(start_time)
    ORDER BY day
    """
    return session.sql(sql).to_pandas()
