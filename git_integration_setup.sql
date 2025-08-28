-- =====================================================
-- Snowflake AI Cost Toolkit - Git Integration Setup
-- =====================================================
-- This script sets up Git integration for direct deployment
-- from the GitHub repository to Snowflake.
--
-- Copyright 2024 Snowflake AI Cost Toolkit Contributors
-- Licensed under the Apache License, Version 2.0
-- =====================================================

-- =====================================================
-- 1. Prerequisites Check
-- =====================================================
-- Ensure you have the necessary privileges:
-- - CREATE INTEGRATION privilege
-- - CREATE STREAMLIT privilege  
-- - CREATE GIT REPOSITORY privilege

-- =====================================================
-- 2. Create GitHub API Integration
-- =====================================================

-- Create API integration with GitHub for the AI Cost Toolkit repository
CREATE OR REPLACE API INTEGRATION GITHUB_INTEGRATION_AI_COST_TOOLKIT
    API_PROVIDER = git_https_api
    API_ALLOWED_PREFIXES = ('https://github.com/sfc-gh-kwells/')
    ENABLED = true
    COMMENT = 'Git integration with Snowflake AI Cost Toolkit repository';

-- Show the integration to verify creation
SHOW INTEGRATIONS LIKE 'GITHUB_INTEGRATION_AI_COST_TOOLKIT';

-- =====================================================
-- 3. Create Git Repository Integration
-- =====================================================

-- Create the git repository integration
CREATE OR REPLACE GIT REPOSITORY CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT
    ORIGIN = 'https://github.com/sfc-gh-kwells/snowflake_ai_cost_observability_tools'
    API_INTEGRATION = 'GITHUB_INTEGRATION_AI_COST_TOOLKIT'
    COMMENT = 'Snowflake AI Cost Toolkit repository for cost observability and analysis';

-- Verify repository creation
SHOW GIT REPOSITORIES IN SCHEMA CORTEX_ANALYTICS.PUBLIC;

-- =====================================================
-- 4. Fetch Repository Content
-- =====================================================

-- Fetch the latest content from the repository
ALTER GIT REPOSITORY CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT FETCH;

-- List files in the repository to verify content
LIST @CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT/branches/main/;

-- =====================================================
-- 5. Deploy Streamlit App from Repository
-- =====================================================

-- Create Streamlit app directly from the git repository
CREATE OR REPLACE STREAMLIT CORTEX_ANALYTICS.PUBLIC.AI_COST_TOOLKIT_DASHBOARD
    ROOT_LOCATION = '@CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT/branches/main'
    MAIN_FILE = 'streamlit_app.py'
    QUERY_WAREHOUSE = 'COMPUTE_WH'  -- Replace with your warehouse name
    COMMENT = 'Comprehensive AI Cost Analysis Dashboard deployed from Git repository';

-- Show the created Streamlit app
SHOW STREAMLITS IN SCHEMA CORTEX_ANALYTICS.PUBLIC;

-- =====================================================
-- 6. Permission Management
-- =====================================================

-- Grant usage on the git repository (customize roles as needed)
GRANT USAGE ON GIT REPOSITORY CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT TO ROLE ACCOUNTADMIN;

-- Grant usage on the Streamlit app (customize roles as needed)
GRANT USAGE ON STREAMLIT CORTEX_ANALYTICS.PUBLIC.AI_COST_TOOLKIT_DASHBOARD TO ROLE ACCOUNTADMIN;

-- Example: Grant to a specific role
-- GRANT USAGE ON GIT REPOSITORY CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT TO ROLE DATA_ANALYST;
-- GRANT USAGE ON STREAMLIT CORTEX_ANALYTICS.PUBLIC.AI_COST_TOOLKIT_DASHBOARD TO ROLE DATA_ANALYST;

-- =====================================================
-- 7. Repository Management Commands
-- =====================================================

-- Check repository branches
SHOW GIT BRANCHES IN CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT;

-- Check repository tags (if any)
SHOW GIT TAGS IN CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT;

-- Get repository details
DESC GIT REPOSITORY CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT;

-- =====================================================
-- 8. Update and Refresh Procedures
-- =====================================================

-- Create a procedure to refresh repository content
CREATE OR REPLACE PROCEDURE REFRESH_GIT_REPOSITORY()
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    -- Fetch latest changes from GitHub
    ALTER GIT REPOSITORY CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT FETCH;
    
    -- Optional: Recreate Streamlit app to pick up changes
    -- DROP STREAMLIT IF EXISTS CORTEX_ANALYTICS.PUBLIC.AI_COST_TOOLKIT_DASHBOARD;
    -- CREATE STREAMLIT CORTEX_ANALYTICS.PUBLIC.AI_COST_TOOLKIT_DASHBOARD
    --     ROOT_LOCATION = '@CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT/branches/main'
    --     MAIN_FILE = 'streamlit_app.py'
    --     QUERY_WAREHOUSE = 'COMPUTE_WH';
    
    RETURN 'Repository refreshed successfully';
END;
$$;

-- Test the refresh procedure
-- CALL REFRESH_GIT_REPOSITORY();

-- =====================================================
-- 9. Alternative: Deploy from Fork or Different Branch
-- =====================================================

-- If you want to deploy from your own fork, modify these values:
/*
CREATE OR REPLACE API INTEGRATION YOUR_GITHUB_INTEGRATION
    API_PROVIDER = git_https_api
    API_ALLOWED_PREFIXES = ('https://github.com/YOUR_USERNAME/')  -- Change this
    ENABLED = true
    COMMENT = 'Git integration with your forked repository';

CREATE OR REPLACE GIT REPOSITORY YOUR_GITHUB_REPO
    ORIGIN = 'https://github.com/YOUR_USERNAME/snowflake_ai_cost_observability_tools'  -- Change this
    API_INTEGRATION = 'YOUR_GITHUB_INTEGRATION'
    COMMENT = 'Your forked AI Cost Toolkit repository';
*/

-- =====================================================
-- 10. Troubleshooting Commands
-- =====================================================

-- Check integration status
SHOW INTEGRATIONS LIKE '%GITHUB%';

-- Check repository status
SHOW GIT REPOSITORIES;

-- Check for any errors in Streamlit app creation
SHOW STREAMLITS;

-- View specific file from repository
-- SELECT GET(@CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT/branches/main/README.md);

-- =====================================================
-- 11. Cleanup Commands (Use with caution)
-- =====================================================

-- To remove everything (uncomment if needed):
-- DROP STREAMLIT IF EXISTS CORTEX_ANALYTICS.PUBLIC.AI_COST_TOOLKIT_DASHBOARD;
-- DROP GIT REPOSITORY IF EXISTS CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT;
-- DROP API INTEGRATION IF EXISTS GITHUB_INTEGRATION_AI_COST_TOOLKIT;

-- =====================================================
-- Setup Complete!
-- =====================================================

SELECT 
    '🎉 Git Integration Setup Complete!' as status,
    'Repository: https://github.com/sfc-gh-kwells/snowflake_ai_cost_observability_tools' as repository_url,
    'Next: Access your Streamlit dashboard in Snowsight > Streamlit Apps' as next_step;

-- =====================================================
-- Usage Instructions
-- =====================================================
/*
After running this script, you can:

1. 📱 Access the Streamlit Dashboard:
   - Go to Snowsight > Apps > Streamlit
   - Find "AI_COST_TOOLKIT_DASHBOARD" 
   - Click to launch the interactive dashboard

2. 📔 Use the Setup Notebook:
   - Go to Snowsight > Projects > Notebooks
   - Create a new notebook from Git repository
   - Use: @CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT/branches/main/setup_and_populate.ipynb

3. 🔄 Update the Repository:
   - When new versions are released, run: CALL REFRESH_GIT_REPOSITORY();
   - Or manually: ALTER GIT REPOSITORY CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT FETCH;

4. 📊 Access Utility Functions:
   - All functions from utils.py are available through the repository
   - Can be imported in notebooks or used in custom functions

5. 🚨 Set up Budget Alerts:
   - Use the budget_alerting_examples.sql file from the repository
   - Reference: @CORTEX_ANALYTICS.PUBLIC.GITHUB_REPO_AI_COST_TOOLKIT/branches/main/budget_alerting_examples.sql
*/
