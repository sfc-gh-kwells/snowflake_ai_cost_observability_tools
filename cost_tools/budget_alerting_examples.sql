-- =====================================================
-- AI Services Budget Alerting Examples
-- =====================================================
-- This file provides comprehensive examples for setting up
-- budget alerts for AI services in Snowflake.
--
-- Copyright 2024 Snowflake AI Cost Toolkit Contributors
-- Licensed under the Apache License, Version 2.0
-- =====================================================

-- =====================================================
-- 1. NOTIFICATION INTEGRATION SETUP
-- =====================================================

-- Example 1: Slack Integration
CREATE NOTIFICATION INTEGRATION MY_SLACK_ALERTS
  TYPE=SLACK
  SLACK_URL='https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'  -- Replace with your webhook URL
  ENABLED=TRUE;

-- Example 2: Email Integration
CREATE NOTIFICATION INTEGRATION MY_EMAIL_ALERTS
  TYPE=EMAIL
  ENABLED=TRUE;

-- Example 3: Multiple integrations for different alert levels
CREATE NOTIFICATION INTEGRATION CRITICAL_ALERTS_SLACK
  TYPE=SLACK
  SLACK_URL='https://hooks.slack.com/services/YOUR/CRITICAL/WEBHOOK'
  ENABLED=TRUE;

CREATE NOTIFICATION INTEGRATION DAILY_SUMMARY_EMAIL
  TYPE=EMAIL
  ENABLED=TRUE;

-- =====================================================
-- 2. BASIC USAGE EXAMPLES
-- =====================================================

-- Example 1: Test comprehensive AI services alert with low threshold
CALL AI_SERVICES_BUDGET_ALERT('MONTH', 10, 'MY_SLACK_ALERTS');

-- Example 2: Monitor only Cortex Functions and Analyst
CALL AI_SERVICES_BUDGET_ALERT(
    'WEEK',              -- Weekly monitoring
    500,                 -- 500 credit threshold
    'MY_EMAIL_ALERTS',   -- Email notifications
    TRUE,                -- Include Cortex Functions
    TRUE,                -- Include Cortex Analyst
    FALSE,               -- Exclude Document AI
    FALSE                -- Exclude Cortex Search
);

-- Example 3: Daily monitoring for high-usage environments
CALL AI_SERVICES_BUDGET_ALERT('DAY', 100, 'MY_SLACK_ALERTS');

-- Example 4: Use the original Cortex Functions specific procedure
CALL MINI_BUDGET_ON_CORTEX('MONTH', 1000, 'MY_SLACK_ALERTS');

-- =====================================================
-- 3. PRODUCTION TASK CONFIGURATIONS
-- =====================================================

-- Configuration 1: High-frequency monitoring for production environments
CREATE OR REPLACE TASK PRODUCTION_AI_BUDGET_MONITOR
WAREHOUSE = 'COMPUTE_WH'
SCHEDULE = '15 minute'  -- Every 15 minutes
USER_TASK_TIMEOUT_MS = 300000  -- 5 minute timeout
AS
CALL AI_SERVICES_BUDGET_ALERT(
    'MONTH',
    5000,  -- $5000 threshold (adjust based on your budget)
    'CRITICAL_ALERTS_SLACK'
);

-- Configuration 2: Business hours monitoring
CREATE OR REPLACE TASK BUSINESS_HOURS_AI_MONITOR
WAREHOUSE = 'COMPUTE_WH'
SCHEDULE = 'USING CRON 0 8-18 * * MON-FRI America/New_York'
AS
CALL AI_SERVICES_BUDGET_ALERT(
    'DAY',
    1000,  -- Daily threshold
    'MY_SLACK_ALERTS'
);

-- Configuration 3: Weekly executive summary
CREATE OR REPLACE TASK WEEKLY_AI_SUMMARY
WAREHOUSE = 'COMPUTE_WH'
SCHEDULE = 'USING CRON 0 9 * * MON America/New_York'  -- Monday at 9 AM
AS
CALL AI_SERVICES_BUDGET_ALERT(
    'WEEK',
    2000,  -- Weekly threshold
    'DAILY_SUMMARY_EMAIL'
);

-- Configuration 4: Monthly budget review
CREATE OR REPLACE TASK MONTHLY_AI_BUDGET_REVIEW
WAREHOUSE = 'COMPUTE_WH'
SCHEDULE = 'USING CRON 0 8 1 * * America/New_York'  -- 1st of each month at 8 AM
AS
CALL AI_SERVICES_BUDGET_ALERT(
    'MONTH',
    10000,  -- Monthly threshold
    'MY_EMAIL_ALERTS'
);

-- =====================================================
-- 4. ADVANCED MONITORING SCENARIOS
-- =====================================================

-- Scenario 1: Tiered alerting system
-- Low threshold warning (every hour during business hours)
CREATE OR REPLACE TASK AI_WARNING_MONITOR
WAREHOUSE = 'COMPUTE_WH'
SCHEDULE = 'USING CRON 0 8-18 * * MON-FRI America/New_York'
AS
CALL AI_SERVICES_BUDGET_ALERT('MONTH', 2500, 'MY_SLACK_ALERTS');

-- High threshold critical alert (every 15 minutes)
CREATE OR REPLACE TASK AI_CRITICAL_MONITOR
WAREHOUSE = 'COMPUTE_WH'
SCHEDULE = '15 minute'
AS
CALL AI_SERVICES_BUDGET_ALERT('MONTH', 4000, 'CRITICAL_ALERTS_SLACK');

-- Scenario 2: Service-specific monitoring
-- Cortex Functions only (high usage service)
CREATE OR REPLACE TASK CORTEX_FUNCTIONS_MONITOR
WAREHOUSE = 'COMPUTE_WH'
SCHEDULE = '30 minute'
AS
CALL AI_SERVICES_BUDGET_ALERT(
    'MONTH', 3000, 'MY_SLACK_ALERTS',
    TRUE,   -- Include Cortex Functions
    FALSE,  -- Exclude Cortex Analyst
    FALSE,  -- Exclude Document AI
    FALSE   -- Exclude Cortex Search
);

-- Document AI monitoring (for document processing heavy workloads)
CREATE OR REPLACE TASK DOCUMENT_AI_MONITOR
WAREHOUSE = 'COMPUTE_WH'
SCHEDULE = '1 hour'
AS
CALL AI_SERVICES_BUDGET_ALERT(
    'WEEK', 1000, 'MY_EMAIL_ALERTS',
    FALSE,  -- Exclude Cortex Functions
    FALSE,  -- Exclude Cortex Analyst
    TRUE,   -- Include Document AI
    FALSE   -- Exclude Cortex Search
);

-- =====================================================
-- 5. TASK MANAGEMENT AND MONITORING
-- =====================================================

-- Start all tasks
ALTER TASK PRODUCTION_AI_BUDGET_MONITOR RESUME;
ALTER TASK BUSINESS_HOURS_AI_MONITOR RESUME;
ALTER TASK WEEKLY_AI_SUMMARY RESUME;
ALTER TASK MONTHLY_AI_BUDGET_REVIEW RESUME;

-- Check task status
SHOW TASKS LIKE '%AI%MONITOR%';

-- View task execution history
SELECT 
    name,
    state,
    scheduled_time,
    query_start_time,
    completed_time,
    return_value,
    error_code,
    error_message
FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY()) 
WHERE name LIKE '%AI%MONITOR%' 
ORDER BY scheduled_time DESC 
LIMIT 20;

-- Suspend specific tasks
-- ALTER TASK PRODUCTION_AI_BUDGET_MONITOR SUSPEND;
-- ALTER TASK BUSINESS_HOURS_AI_MONITOR SUSPEND;

-- =====================================================
-- 6. CUSTOMIZED ALERT MESSAGES
-- =====================================================

-- Create a procedure with custom alert formatting
CREATE OR REPLACE PROCEDURE CUSTOM_AI_BUDGET_ALERT(
    billing_period VARCHAR,
    alert_limit FLOAT,
    notification_integration VARCHAR,
    organization_name VARCHAR DEFAULT 'Your Organization'
)
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
DECLARE
    start_of_period DATE;
    total_ai_credits FLOAT DEFAULT 0;
    notification_body VARCHAR;
    percentage_used FLOAT;
BEGIN
    -- Determine billing period
    CASE UPPER(billing_period)
        WHEN 'DAY' THEN start_of_period := CURRENT_DATE();
        WHEN 'WEEK' THEN start_of_period := DATE_TRUNC('week', CURRENT_DATE());
        WHEN 'MONTH' THEN start_of_period := DATE_TRUNC('month', CURRENT_DATE());
    END CASE;

    -- Get total usage from metering history
    SELECT COALESCE(SUM(credits_used), 0) INTO total_ai_credits
    FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_HISTORY
    WHERE start_time >= :start_of_period
    AND service_type = 'AI_SERVICES';

    percentage_used := (total_ai_credits / alert_limit) * 100;

    -- Custom formatted message
    notification_body := '📊 ' || organization_name || ' AI Services Budget Report\\n\\n'
                      || '🏢 Account: ' || CURRENT_ACCOUNT() || '\\n'
                      || '📅 Period: ' || billing_period || ' (since ' || start_of_period || ')\\n'
                      || '💰 Budget Limit: ' || alert_limit || ' credits\\n'
                      || '💳 Current Usage: ' || total_ai_credits || ' credits\\n'
                      || '📈 Usage Percentage: ' || ROUND(percentage_used, 2) || '%\\n\\n';

    IF (total_ai_credits >= alert_limit) THEN
        notification_body := notification_body || '🚨 BUDGET EXCEEDED! Immediate action required.\\n';
        
        IF (notification_integration IS NOT NULL) THEN
            CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
                '{"text/plain":"' || :notification_body || '"}',
                '{"' || :notification_integration || '": {}}'
            );
        END IF;
    ELSIF (percentage_used >= 80) THEN
        notification_body := notification_body || '⚠️ WARNING: 80% of budget used. Monitor closely.\\n';
        
        IF (notification_integration IS NOT NULL) THEN
            CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
                '{"text/plain":"' || :notification_body || '"}',
                '{"' || :notification_integration || '": {}}'
            );
        END IF;
    ELSE
        notification_body := notification_body || '✅ Budget usage is within normal limits.\\n';
    END IF;

    RETURN notification_body;
END;
$$;

-- =====================================================
-- 7. TESTING AND VALIDATION
-- =====================================================

-- Test notification integration
CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
    '{"text/plain":"Test message: AI Services Budget Monitoring is now active!"}',
    '{"MY_SLACK_ALERTS": {}}'
);

-- Test budget alert with very low threshold (should trigger)
CALL AI_SERVICES_BUDGET_ALERT('DAY', 0.01, 'MY_SLACK_ALERTS');

-- Test custom alert procedure
CALL CUSTOM_AI_BUDGET_ALERT('MONTH', 1000, 'MY_EMAIL_ALERTS', 'Acme Corporation');

-- =====================================================
-- 8. BUDGET OPTIMIZATION QUERIES
-- =====================================================

-- Query to understand current spend patterns
SELECT 
    service_type,
    DATE_TRUNC('day', start_time) as usage_date,
    SUM(credits_used) as daily_credits
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_HISTORY
WHERE start_time >= DATE_TRUNC('month', CURRENT_DATE())
AND service_type = 'AI_SERVICES'
GROUP BY 1, 2
ORDER BY 2 DESC, 3 DESC;

-- Query to identify highest usage days
SELECT 
    DATE_TRUNC('day', start_time) as usage_date,
    SUM(credits_used) as total_credits,
    COUNT(*) as usage_events
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_HISTORY
WHERE start_time >= DATE_TRUNC('month', CURRENT_DATE())
AND service_type = 'AI_SERVICES'
GROUP BY 1
ORDER BY 2 DESC
LIMIT 10;

-- =====================================================
-- 9. CLEANUP COMMANDS (USE WITH CAUTION)
-- =====================================================

-- Drop all created tasks (uncomment if needed)
-- DROP TASK IF EXISTS PRODUCTION_AI_BUDGET_MONITOR;
-- DROP TASK IF EXISTS BUSINESS_HOURS_AI_MONITOR;
-- DROP TASK IF EXISTS WEEKLY_AI_SUMMARY;
-- DROP TASK IF EXISTS MONTHLY_AI_BUDGET_REVIEW;
-- DROP TASK IF EXISTS AI_WARNING_MONITOR;
-- DROP TASK IF EXISTS AI_CRITICAL_MONITOR;
-- DROP TASK IF EXISTS CORTEX_FUNCTIONS_MONITOR;
-- DROP TASK IF EXISTS DOCUMENT_AI_MONITOR;

-- Drop notification integrations (uncomment if needed)
-- DROP NOTIFICATION INTEGRATION IF EXISTS MY_SLACK_ALERTS;
-- DROP NOTIFICATION INTEGRATION IF EXISTS MY_EMAIL_ALERTS;
-- DROP NOTIFICATION INTEGRATION IF EXISTS CRITICAL_ALERTS_SLACK;
-- DROP NOTIFICATION INTEGRATION IF EXISTS DAILY_SUMMARY_EMAIL;
