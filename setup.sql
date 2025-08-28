-- =====================================================
-- Snowflake AI Cost Toolkit - Setup Script
-- =====================================================
-- This script sets up the necessary tables and schemas
-- for analyzing Cortex Analyst costs and performance.
-- 
-- Copyright 2024 Snowflake AI Cost Toolkit Contributors
-- 
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
-- 
--     http://www.apache.org/licenses/LICENSE-2.0
-- 
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
-- =====================================================

-- Create database and schema (optional - adjust as needed)
CREATE DATABASE IF NOT EXISTS CORTEX_ANALYTICS;
USE DATABASE CORTEX_ANALYTICS;
CREATE SCHEMA IF NOT EXISTS PUBLIC;
USE SCHEMA PUBLIC;

-- =====================================================
-- 1. Create Cortex Analyst Logs Table
-- =====================================================
-- This table stores processed Cortex Analyst logs with
-- additional computed fields for analysis.

CREATE OR REPLACE TABLE CORTEX_ANALYST_LOGS (
    TIMESTAMP                TIMESTAMP_NTZ,
    REQUEST_ID               STRING,
    SEMANTIC_MODEL_NAME      STRING,
    TABLES_REFERENCED        STRING,
    USER_NAME                STRING,
    SOURCE                   STRING,
    FEEDBACK                 STRING,
    RESPONSE_STATUS_CODE     INTEGER,
    USER_QUESTION            STRING,
    LATENCY_MS               NUMBER,
    GENERATED_SQL            STRING,
    ORCHESTRATION_PATH       STRING,
    QUESTION_CATEGORY        STRING,
    VERIFIED_QUERY_NAME      STRING,
    VERIFIED_QUERY_QUESTION  STRING,
    QUERY_TYPE               STRING,
    CORTEX_ANALYST_CREDITS   FLOAT
);

-- =====================================================
-- 2. Create Snowflake Intelligence Query History Table
-- =====================================================
-- This table stores cleaned query history data joined with
-- attribution data for cost analysis.

CREATE OR REPLACE TABLE SF_INTELLIGENCE_QUERY_HISTORY AS
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
            REGEXP_REPLACE(qh.query_text, '--[^\n]*\n', '\n'),
            '/\*.*?\*/', ' '
        ),
        '\s+', ' '
    )) AS cleaned_query_text
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY qh
JOIN SNOWFLAKE.ACCOUNT_USAGE.QUERY_ATTRIBUTION_HISTORY qah 
    ON qh.query_id = qah.query_id
WHERE qh.query_tag = 'cortex-agent'
  AND qh.start_time >= DATEADD(DAY, -30, CURRENT_TIMESTAMP())
  AND qah.credits_attributed_compute > 0;

-- =====================================================
-- 3. Create Summary Views for Common Analysis
-- =====================================================

-- View: Cost Summary by Semantic Model
CREATE OR REPLACE VIEW V_COST_SUMMARY_BY_SEMANTIC_MODEL AS
WITH joined_data AS (
    SELECT 
        cal.*,
        qh.credits_attributed_compute,
        qh.total_elapsed_time,
        (cal.cortex_analyst_credits + qh.credits_attributed_compute) AS total_credits_wh_and_ca
    FROM CORTEX_ANALYST_LOGS cal
    JOIN CORTEX_ANALYTICS.PUBLIC.SF_INTELLIGENCE_QUERY_HISTORY qh
        ON TRIM(cal.generated_sql) = qh.cleaned_query_text
)
SELECT 
    semantic_model_name,
    COUNT(*) AS query_count,
    COUNT(DISTINCT user_name) AS unique_users,
    SUM(cortex_analyst_credits) AS total_cortex_analyst_credits,
    SUM(credits_attributed_compute) AS total_warehouse_credits,
    SUM(total_credits_wh_and_ca) AS total_combined_credits,
    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
    ROUND(SUM(total_credits_wh_and_ca) / COUNT(*), 4) AS avg_credits_per_query,
    ROUND((SUM(total_credits_wh_and_ca) / SUM(SUM(total_credits_wh_and_ca)) OVER ()) * 100, 2) AS percentage_of_total_cost
FROM joined_data
GROUP BY semantic_model_name
ORDER BY total_combined_credits DESC;

-- View: User Activity by Semantic Model
CREATE OR REPLACE VIEW V_USER_ACTIVITY_BY_SEMANTIC_MODEL AS
WITH joined_data AS (
    SELECT 
        cal.*,
        qh.credits_attributed_compute,
        (cal.cortex_analyst_credits + qh.credits_attributed_compute) AS total_credits_wh_and_ca
    FROM CORTEX_ANALYST_LOGS cal
    JOIN CORTEX_ANALYTICS.PUBLIC.SF_INTELLIGENCE_QUERY_HISTORY qh
        ON TRIM(cal.generated_sql) = qh.cleaned_query_text
)
SELECT 
    user_name,
    semantic_model_name,
    COUNT(*) AS queries_count,
    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
    SUM(total_credits_wh_and_ca) AS total_credits,
    ROUND((COUNT(*) * 100.0) / SUM(COUNT(*)) OVER (PARTITION BY user_name), 2) AS percentage_of_user_queries
FROM joined_data
GROUP BY user_name, semantic_model_name
ORDER BY user_name, queries_count DESC;

-- View: Query Type Analysis
CREATE OR REPLACE VIEW V_QUERY_TYPE_ANALYSIS AS
SELECT 
    semantic_model_name,
    query_type,
    COUNT(*) AS request_count,
    ROUND((COUNT(*) * 100.0) / SUM(COUNT(*)) OVER (PARTITION BY semantic_model_name), 2) AS percentage,
    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
    SUM(cortex_analyst_credits) AS total_cortex_analyst_credits
FROM CORTEX_ANALYST_LOGS
GROUP BY semantic_model_name, query_type
ORDER BY semantic_model_name, request_count DESC;

-- =====================================================
-- 4. Create Helper Functions (if supported)
-- =====================================================

-- Note: The following are example stored procedures.
-- Adjust the database/schema names as needed for your environment.

-- Procedure to refresh query history data
CREATE OR REPLACE PROCEDURE REFRESH_QUERY_HISTORY()
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    -- Refresh the intelligence query history table with latest data
    CREATE OR REPLACE TABLE CORTEX_ANALYTICS.PUBLIC.SF_INTELLIGENCE_QUERY_HISTORY AS
    SELECT 
        qh.query_id,
        qh.query_text,
        qh.start_time,
        qh.total_elapsed_time,
        qh.warehouse_name,
        qh.user_name,
        qah.credits_attributed_compute,
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
      AND qah.credits_attributed_compute > 0;
      
    RETURN 'Query history refreshed successfully';
END;
$$;

-- =====================================================
-- 5. Grant Permissions (adjust as needed)
-- =====================================================

-- Grant permissions to roles that need access
-- GRANT SELECT ON TABLE CORTEX_ANALYST_LOGS TO ROLE ANALYST_ROLE;
-- GRANT SELECT ON TABLE CORTEX_ANALYTICS.PUBLIC.SF_INTELLIGENCE_QUERY_HISTORY TO ROLE ANALYST_ROLE;
-- GRANT SELECT ON VIEW V_COST_SUMMARY_BY_SEMANTIC_MODEL TO ROLE ANALYST_ROLE;
-- GRANT SELECT ON VIEW V_USER_ACTIVITY_BY_SEMANTIC_MODEL TO ROLE ANALYST_ROLE;
-- GRANT SELECT ON VIEW V_QUERY_TYPE_ANALYSIS TO ROLE ANALYST_ROLE;

-- =====================================================
-- 6. Validation Queries
-- =====================================================

-- Check if tables are created successfully
SELECT 'CORTEX_ANALYST_LOGS' AS table_name, COUNT(*) AS row_count FROM CORTEX_ANALYST_LOGS
UNION ALL
SELECT 'SF_INTELLIGENCE_QUERY_HISTORY' AS table_name, COUNT(*) AS row_count 
FROM CORTEX_ANALYTICS.PUBLIC.SF_INTELLIGENCE_QUERY_HISTORY;

-- Test views
SELECT * FROM V_COST_SUMMARY_BY_SEMANTIC_MODEL LIMIT 5;
SELECT * FROM V_USER_ACTIVITY_BY_SEMANTIC_MODEL LIMIT 5;
SELECT * FROM V_QUERY_TYPE_ANALYSIS LIMIT 5;

SHOW TABLES LIKE '%CORTEX%';
SHOW VIEWS LIKE 'V_%';

-- =====================================================
-- 7. AI Services Budget Alerting System
-- =====================================================

-- Comprehensive AI Services Budget Alert Procedure
-- Monitors all AI services and sends notifications when thresholds are exceeded
CREATE OR REPLACE PROCEDURE AI_SERVICES_BUDGET_ALERT(
    billing_period VARCHAR,
    alert_limit FLOAT,
    notification_integration VARCHAR,
    include_cortex_functions BOOLEAN DEFAULT TRUE,
    include_cortex_analyst BOOLEAN DEFAULT TRUE,
    include_document_ai BOOLEAN DEFAULT TRUE,
    include_cortex_search BOOLEAN DEFAULT TRUE
)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
DECLARE
    start_of_period DATE;
    INVALID_PERIOD EXCEPTION (-20001, 'Invalid billing_period specified. Must be DAY, WEEK, or MONTH.');
    
    -- Usage variables for each service
    cortex_functions_credits FLOAT DEFAULT 0;
    cortex_analyst_credits FLOAT DEFAULT 0;
    document_ai_credits FLOAT DEFAULT 0;
    cortex_search_credits FLOAT DEFAULT 0;
    total_ai_credits FLOAT DEFAULT 0;
    
    -- Notification variables
    notification_body VARCHAR;
    service_breakdown VARCHAR DEFAULT '';
    alert_triggered BOOLEAN DEFAULT FALSE;
BEGIN
    -- Determine billing period start date
    CASE UPPER(billing_period)
        WHEN 'DAY' THEN
            start_of_period := CURRENT_DATE();
        WHEN 'WEEK' THEN
            start_of_period := DATE_TRUNC('week', CURRENT_DATE());
        WHEN 'MONTH' THEN
            start_of_period := DATE_TRUNC('month', CURRENT_DATE());
        ELSE
            RAISE INVALID_PERIOD;
    END CASE;

    -- Get Cortex Functions usage
    IF (include_cortex_functions) THEN
        SELECT COALESCE(SUM(TOKEN_CREDITS), 0) INTO cortex_functions_credits
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
        WHERE START_TIME >= :start_of_period;
        
        service_breakdown := service_breakdown || 'Cortex Functions: ' || cortex_functions_credits || ' credits\\n';
    END IF;

    -- Get Cortex Analyst usage
    IF (include_cortex_analyst) THEN
        SELECT COALESCE(SUM(CREDITS), 0) INTO cortex_analyst_credits
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_ANALYST_USAGE_HISTORY
        WHERE START_TIME >= :start_of_period;
        
        service_breakdown := service_breakdown || 'Cortex Analyst: ' || cortex_analyst_credits || ' credits\\n';
    END IF;

    -- Get Document AI usage
    IF (include_document_ai) THEN
        SELECT COALESCE(SUM(CREDITS_USED), 0) INTO document_ai_credits
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_DOCUMENT_PROCESSING_USAGE_HISTORY
        WHERE START_TIME >= :start_of_period;
        
        service_breakdown := service_breakdown || 'Document AI: ' || document_ai_credits || ' credits\\n';
    END IF;

    -- Get Cortex Search usage
    IF (include_cortex_search) THEN
        SELECT COALESCE(SUM(CREDITS), 0) INTO cortex_search_credits
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_SERVING_USAGE_HISTORY
        WHERE START_TIME >= :start_of_period;
        
        service_breakdown := service_breakdown || 'Cortex Search: ' || cortex_search_credits || ' credits\\n';
    END IF;

    -- Calculate total AI services usage
    total_ai_credits := cortex_functions_credits + cortex_analyst_credits + document_ai_credits + cortex_search_credits;

    -- Build notification message
    notification_body := '🚨 AI Services Budget Alert\\n\\n'
                      || 'Account: ' || CURRENT_ACCOUNT() || ' (' || CURRENT_REGION() || ')\\n'
                      || 'Period: ' || billing_period || ' (since ' || start_of_period || ')\\n'
                      || 'Alert Threshold: ' || alert_limit || ' credits\\n'
                      || 'Current Total Usage: ' || total_ai_credits || ' credits\\n\\n'
                      || 'Service Breakdown:\\n' || service_breakdown;

    -- Check if alert threshold is exceeded
    IF (total_ai_credits >= alert_limit) THEN
        alert_triggered := TRUE;
        notification_body := notification_body || '\\n⚠️ ALERT: Usage has exceeded the threshold of ' || alert_limit || ' credits!';
        
        -- Send notification if integration is provided
        IF (notification_integration IS NOT NULL AND TRIM(notification_integration) <> '') THEN
            CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
                '{"text/plain":"' || :notification_body || '"}',
                '{"' || :notification_integration || '": {}}'
            );
        END IF;
    ELSE
        notification_body := notification_body || '\\n✅ Usage is within budget limits.';
    END IF;

    RETURN notification_body;
END;
$$;

-- Specific Cortex Functions Budget Alert (compatible with existing example)
CREATE OR REPLACE PROCEDURE MINI_BUDGET_ON_CORTEX(
    billing_period VARCHAR,
    alert_limit FLOAT,
    ni_name VARCHAR
)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
DECLARE
    start_of_period       DATE;
    INVALID_PERIOD        EXCEPTION (-20001, 'Invalid billing_period specified. Must be DAY, WEEK, or MONTH.');
    current_usage_credits FLOAT;
    notification_body     VARCHAR;
BEGIN
    CASE UPPER(billing_period)
        WHEN 'DAY' THEN
            start_of_period := CURRENT_DATE();
        WHEN 'WEEK' THEN
            start_of_period := DATE_TRUNC('week', CURRENT_DATE());
        WHEN 'MONTH' THEN
            start_of_period := DATE_TRUNC('month', CURRENT_DATE());
        ELSE
            RAISE INVALID_PERIOD;
    END CASE;

    SELECT COALESCE(SUM(TOKEN_CREDITS), 0) INTO current_usage_credits
    FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
    WHERE START_TIME >= :start_of_period;

    notification_body := 'Account: ' || CURRENT_ACCOUNT() || ' (' || CURRENT_REGION() || ')'
                      || ' - Cortex functions credit usage in this ' || billing_period
                      || ' = ' || current_usage_credits;
    IF (current_usage_credits >= alert_limit) THEN
        IF (ni_name IS NOT NULL AND TRIM(ni_name) <> '') THEN
            CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
                '{"text/plain":"' || :notification_body || '"}',
                '{"' || :ni_name || '": {}}'
            );
        END IF;
    END IF;

    RETURN notification_body;
END;
$$;

-- =====================================================
-- 8. Example Tasks for Automated Budget Monitoring
-- =====================================================

-- Comprehensive AI Services Budget Monitoring Task
-- Monitors all AI services every hour during business hours
CREATE OR REPLACE TASK AI_SERVICES_BUDGET_MONITOR
WAREHOUSE = 'COMPUTE_WH'  -- Replace with your warehouse
SCHEDULE = 'USING CRON 0 8-18 * * MON-FRI America/New_York'  -- Every hour, 8 AM to 6 PM, Monday-Friday
AS
CALL AI_SERVICES_BUDGET_ALERT(
    'MONTH',           -- Billing period
    1000,              -- Alert threshold (adjust as needed)
    'MY_SLACK_NI'      -- Replace with your notification integration name
);

-- Cortex Functions Specific Monitoring Task (every 30 minutes)
CREATE OR REPLACE TASK MINI_BUDGET_CORTEX_MONITOR
WAREHOUSE = 'COMPUTE_WH'  -- Replace with your warehouse
SCHEDULE = '30 minute'
AS
CALL MINI_BUDGET_ON_CORTEX('MONTH', 1000, 'MY_SLACK_NI');

-- Daily Summary Task (runs at 9 AM every day)
CREATE OR REPLACE TASK AI_SERVICES_DAILY_SUMMARY
WAREHOUSE = 'COMPUTE_WH'  -- Replace with your warehouse
SCHEDULE = 'USING CRON 0 9 * * * America/New_York'
AS
CALL AI_SERVICES_BUDGET_ALERT(
    'DAY',             -- Daily summary
    500,               -- Lower threshold for daily alerts
    'MY_EMAIL_NI'      -- Replace with your email notification integration
);

-- =====================================================
-- 9. Notification Integration Setup Examples
-- =====================================================

-- Example: Create Slack notification integration
-- CREATE NOTIFICATION INTEGRATION MY_SLACK_NI
--   TYPE=SLACK
--   SLACK_URL='YOUR_SLACK_WEBHOOK_URL'
--   ENABLED=TRUE;

-- Example: Create email notification integration  
-- CREATE NOTIFICATION INTEGRATION MY_EMAIL_NI
--   TYPE=EMAIL
--   ENABLED=TRUE;

-- =====================================================
-- 10. Task Management Commands
-- =====================================================

-- To start the tasks (uncomment and run when ready):
-- ALTER TASK AI_SERVICES_BUDGET_MONITOR RESUME;
-- ALTER TASK MINI_BUDGET_CORTEX_MONITOR RESUME;
-- ALTER TASK AI_SERVICES_DAILY_SUMMARY RESUME;

-- To stop the tasks:
-- ALTER TASK AI_SERVICES_BUDGET_MONITOR SUSPEND;
-- ALTER TASK MINI_BUDGET_CORTEX_MONITOR SUSPEND;
-- ALTER TASK AI_SERVICES_DAILY_SUMMARY SUSPEND;

-- To check task status:
-- SHOW TASKS LIKE '%BUDGET%';
-- SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY()) WHERE NAME LIKE '%BUDGET%' ORDER BY SCHEDULED_TIME DESC;

-- =====================================================
-- Usage Examples and Testing
-- =====================================================

-- Test the comprehensive AI services alert (run manually):
-- CALL AI_SERVICES_BUDGET_ALERT('MONTH', 10, 'MY_SLACK_NI');

-- Test with selective services:
-- CALL AI_SERVICES_BUDGET_ALERT('WEEK', 100, 'MY_SLACK_NI', TRUE, TRUE, FALSE, FALSE);

-- Test the Cortex Functions specific alert:
-- CALL MINI_BUDGET_ON_CORTEX('MONTH', 10, 'MY_SLACK_NI');

-- =====================================================
-- 11. Git Integration Setup (Optional)
-- =====================================================

-- Create API integration with GitHub for the AI Cost Toolkit repository
CREATE OR REPLACE API INTEGRATION GITHUB_INTEGRATION_AI_COST_TOOLKIT
    API_PROVIDER = git_https_api
    API_ALLOWED_PREFIXES = ('https://github.com/sfc-gh-kwells/')
    ENABLED = true
    COMMENT = 'Git integration with Snowflake AI Cost Toolkit repository';

-- Create the git repository integration
CREATE OR REPLACE GIT REPOSITORY GITHUB_REPO_AI_COST_TOOLKIT
    ORIGIN = 'https://github.com/sfc-gh-kwells/snowflake_ai_cost_observability_tools'
    API_INTEGRATION = 'GITHUB_INTEGRATION_AI_COST_TOOLKIT'
    COMMENT = 'Snowflake AI Cost Toolkit repository for cost observability and analysis';

-- Grant usage on the git repository (adjust role as needed)
-- GRANT USAGE ON GIT REPOSITORY GITHUB_REPO_AI_COST_TOOLKIT TO ROLE ACCOUNTADMIN;

-- =====================================================
-- 12. Deploy Streamlit App from Git Repository
-- =====================================================

-- Create Streamlit app directly from the git repository
CREATE OR REPLACE STREAMLIT CORTEX_ANALYTICS.PUBLIC.AI_COST_TOOLKIT_DASHBOARD
    ROOT_LOCATION = '@CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT/branches/main'
    MAIN_FILE = 'streamlit_app.py'
    QUERY_WAREHOUSE = 'COMPUTE_WH'  -- Replace with your warehouse name
    COMMENT = 'Comprehensive AI Cost Analysis Dashboard';

-- Grant usage on the Streamlit app (adjust role as needed)
-- GRANT USAGE ON STREAMLIT CORTEX_ANALYTICS.PUBLIC.AI_COST_TOOLKIT_DASHBOARD TO ROLE YOUR_ROLE;

-- =====================================================
-- 13. Alternative: Create Snowflake Notebook from Git
-- =====================================================

-- Create notebook from the setup_and_populate.ipynb file in the repository
-- Note: This requires manual execution through Snowsight UI
-- 1. Go to Snowsight > Projects > Notebooks
-- 2. Create notebook from Git repository: @CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT/branches/main/setup_and_populate.ipynb

-- =====================================================
-- 14. Refresh Repository Content
-- =====================================================

-- To update the repository with latest changes from GitHub:
-- ALTER GIT REPOSITORY GITHUB_REPO_AI_COST_TOOLKIT FETCH;

-- To check repository status:
-- SHOW GIT BRANCHES IN GITHUB_REPO_AI_COST_TOOLKIT;
-- SHOW GIT TAGS IN GITHUB_REPO_AI_COST_TOOLKIT;

-- List files in the repository:
-- LIST @CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT/branches/main/;

-- =====================================================
-- 15. Alternative Deployment Commands
-- =====================================================

-- If you have additional SQL scripts in your repository, you can execute them:
-- EXECUTE IMMEDIATE FROM @CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT/branches/main/additional_setup.sql;

-- To deploy utility functions, you can reference them directly from the repository:
-- Example: Creating a function that uses the repository files
-- CREATE OR REPLACE FUNCTION GET_UTILS_VERSION()
-- RETURNS STRING
-- LANGUAGE PYTHON
-- RUNTIME_VERSION = '3.8'
-- HANDLER = 'get_version'
-- IMPORTS = ('@CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT/branches/main/utils.py')
-- AS
-- $$
-- def get_version():
--     return "Snowflake AI Cost Toolkit v1.0"
-- $$;

-- =====================================================
-- Setup Complete!
-- =====================================================
-- Your Snowflake AI Cost Toolkit is now ready to use.
-- 
-- Git Integration Features:
-- • Direct deployment from GitHub repository
-- • Automatic Streamlit app creation from repository
-- • Easy updates with ALTER GIT REPOSITORY FETCH
-- 
-- Next steps:
-- 1. Commit and push your changes to the GitHub repository
-- 2. Use the notebook setup_and_populate.ipynb to populate data
-- 3. Access the Streamlit dashboard for interactive analysis
-- 4. Configure notification integrations for budget alerts
-- 5. Set up automated monitoring tasks
-- 
-- Repository URL: https://github.com/sfc-gh-kwells/snowflake_ai_cost_observability_tools
-- =====================================================
