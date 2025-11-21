# Snowflake AI Cost Toolkit

A collection of utilities for analyzing Cortex Analyst logs and costs in Snowflake.

## Overview

This toolkit provides functions to:
- Fetch semantic model paths from Snowflake agents
- Retrieve and analyze Cortex Analyst logs
- Calculate query latency statistics
- Analyze user activity and costs
- Join Cortex Analyst logs with query attribution data

## Installation

### Prerequisites
This toolkit is designed to run within Snowflake environments (Snowpark notebooks, Streamlit apps, etc.) where the following dependencies are available:
- `pandas`
- `snowflake.snowpark`

### Setup

#### Option 1: Git Integration (Recommended for Production)
1. **Run the Git integration setup** to deploy directly from GitHub:
   ```sql
   -- Execute the git_integration_setup.sql script in your Snowflake environment
   -- This creates Git repository integration and deploys the Streamlit app
   ```
2. **Access the deployed Streamlit dashboard** in Snowsight > Apps > Streamlit
3. **Create notebook from repository** using the Git-integrated setup_and_populate.ipynb

#### Option 2: Snowflake Notebook (Recommended for Development)
1. **Upload the notebook** `setup_and_populate.ipynb` to your Snowflake account
2. **Run all cells** in the notebook to:
   - Create database schema (tables, views, procedures) - only if they don't exist
   - Automatically discover and populate Cortex Analyst logs - skips existing data
   - Verify the setup with sample queries
   - **Cost-efficient**: Avoids expensive table recreation and duplicate data processing

#### Option 3: Manual Setup
1. **Run the SQL setup script** to create the necessary database objects:
   ```sql
   -- Execute the setup.sql script in your Snowflake environment
   -- This creates tables, views, and stored procedures
   ```

2. **Manually populate data** using the utility functions in a Python environment

## Files Included

- **`git_integration_setup.sql`** - Git integration setup for direct deployment from GitHub repository
- **`setup_and_populate.ipynb`** - Snowflake notebook that sets up the complete toolkit and populates data
- **`setup.sql`** - Complete SQL setup script with views, stored procedures, and budget alerting system
- **`utils.py`** - Python utility functions for data analysis and processing
- **`streamlit_app.py`** - Comprehensive Streamlit dashboard for AI cost analysis
- **`environment.yml`** - Snowflake environment dependencies
- **`budget_alerting_examples.sql`** - Comprehensive examples for setting up budget alerts and notifications
- **`README.md`** - This documentation file

## Usage

Import the utility functions in your Snowflake notebook or application:

```python
from utils import (
    fetch_semantic_model_paths,
    get_cortex_analyst_logs,
    verified_query_count,
    top_verified_queries,
    slowest_queries,
    latency_summary_by_semantic_model,
    user_activity_by_semantic_model,
    semantic_model_usage_summary,
    create_cortex_analyst_query_history
)

# Get active Snowpark session
from snowflake.snowpark.context import get_active_session
session = get_active_session()

# Fetch semantic model paths
semantic_models = fetch_semantic_model_paths(session)

# Get logs for a specific semantic model
logs_df = get_cortex_analyst_logs(session, semantic_model_file_path)

# Analyze the logs
latency_stats = latency_summary_by_semantic_model(logs_df)
user_activity = user_activity_by_semantic_model(logs_df)

# Join with query history for cost analysis
ca_query_history = create_cortex_analyst_query_history(session, logs_df)

# Calculate total costs by semantic model
total_costs = total_cost_by_semantic_model(ca_query_history)
cost_breakdown = cost_breakdown_by_semantic_model(ca_query_history)

# Use LLM Usage Dashboard functions for broader AI cost analysis
import datetime
start_date = datetime.datetime.now() - datetime.timedelta(days=30)
end_date = datetime.datetime.now()

# Get AI Services overview
ai_total = get_ai_services_total_credits(session, start_date, end_date)
llm_summary = get_llm_inference_summary(session, start_date, end_date)
ca_summary = get_cortex_analyst_summary(session, start_date, end_date)

# Get detailed breakdowns
credits_by_model = get_credits_by_model(session, start_date, end_date)
credits_by_warehouse = get_credits_by_warehouse(session, start_date, end_date)
```

## Local Development & Testing

### Running Streamlit Locally

The toolkit supports local development and testing using your Snowflake connection. This is perfect for development, testing, and troubleshooting.

#### Prerequisites

1. **Python Environment**: Python 3.8+ with required dependencies
   ```bash
   pip install streamlit snowflake-snowpark-python pandas configparser
   ```

2. **Snowflake Connection**: Create a `config.env` file in the project directory with your connection details:
   ```ini
   [connections.my_example_connection]
   account = "your-account.region"
   user = "your-username"
   password = "your-password"
   role = "your-role"
   warehouse = "your-warehouse"
   database = "CORTEX_ANALYTICS"
   schema = "PUBLIC"
   ```

#### Running the Streamlit App

1. **Navigate to the project directory**:
   ```bash
   cd snowflake_ai_cost_observability_tools
   ```

2. **Start the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the dashboard**:
   - Local URL: `http://localhost:8501`
   - Network URL: `http://your-ip:8501` (for remote access)

#### Expected Output

When starting locally, you should see:
```
🔧 No active session found, creating local session for development...
✅ Local Snowpark session created successfully
   Account: your-account.region
   User: your-username  
   Role: your-role
   Warehouse: your-warehouse
   Database: CORTEX_ANALYTICS
   Schema: PUBLIC

  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

#### Using Jupyter Notebooks for Development

For interactive development and testing:

1. **Use the development notebook**:
   ```bash
   jupyter notebook setup_local_dev.ipynb
   ```

2. **Or test functions directly**:
   ```python
   from utils import get_session, fetch_semantic_model_paths
   
   # Create local session
   session = get_session()
   
   # Test functions
   semantic_models = fetch_semantic_model_paths(session)
   print(f"Found {len(semantic_models)} semantic models")
   ```

#### Troubleshooting

- **Connection Issues**: Verify your `config.env` file has correct credentials
- **Missing Data**: Run the setup scripts first to populate your Cortex Analyst logs
- **Import Errors**: Ensure all dependencies are installed in your Python environment
- **Kernel Restart**: In Jupyter notebooks, restart the kernel after modifying `utils.py`

## Streamlit Dashboard Deployment

### Deploy in Snowflake
To deploy the Streamlit dashboard in your Snowflake account:

1. **Run the setup script** to create all necessary database objects
2. **Upload the files** to a Snowflake stage:
   ```sql
   -- Create a stage for your Streamlit app
   CREATE STAGE IF NOT EXISTS CORTEX_ANALYTICS.PUBLIC.STREAMLIT_STAGE;
   
   -- Upload the files using SnowSQL or Snowsight
   PUT file://streamlit_app.py @CORTEX_ANALYTICS.PUBLIC.STREAMLIT_STAGE;
   PUT file://utils.py @CORTEX_ANALYTICS.PUBLIC.STREAMLIT_STAGE;
   PUT file://environment.yml @CORTEX_ANALYTICS.PUBLIC.STREAMLIT_STAGE;
   ```

3. **Create the Streamlit app**:
   ```sql
   CREATE STREAMLIT CORTEX_ANALYTICS.PUBLIC.AI_COST_TOOLKIT_DASHBOARD
   ROOT_LOCATION = '@CORTEX_ANALYTICS.PUBLIC.STREAMLIT_STAGE'
   MAIN_FILE = 'streamlit_app.py'
   QUERY_WAREHOUSE = 'YOUR_WAREHOUSE_NAME';
   ```

4. **Grant permissions** as needed:
   ```sql
   GRANT USAGE ON STREAMLIT CORTEX_ANALYTICS.PUBLIC.AI_COST_TOOLKIT_DASHBOARD TO ROLE YOUR_ROLE;
   ```

### Dashboard Features
The Streamlit dashboard provides:
- **📅 Flexible date range selection** with quick preset buttons
- **🤖 AI Services overview** with total credits and service breakdown
- **🧠 Cortex Functions analysis** by function, model, and warehouse
- **🔍 Detailed Cortex Analyst analysis** including query types, performance, user activity, and costs
- **📈 Time series charts** showing usage trends over time
- **🔎 Data explorer** with detailed breakdowns and filtering

## Git Integration

The toolkit includes comprehensive Git integration for seamless deployment and updates directly from the GitHub repository.

### Key Features
- **Direct deployment** from GitHub repository to Snowflake
- **Automatic Streamlit app creation** from repository files
- **One-command updates** with repository refresh
- **Version control integration** for production deployments
- **Easy forking and customization** for organization-specific needs

### Git Integration Benefits
1. **Production Ready** - Deploy directly from source control
2. **Always Up-to-Date** - Easy updates with `ALTER GIT REPOSITORY FETCH`
3. **Consistent Deployments** - Same codebase across environments
4. **Collaboration Friendly** - Easy sharing and contribution
5. **Enterprise Grade** - Full audit trail and version control

### Quick Git Setup
```sql
-- Run the Git integration setup
-- This creates the repository integration and deploys the Streamlit app
-- Execute the git_integration_setup.sql script in your Snowflake environment

-- Access your dashboard at: Snowsight > Apps > Streamlit > AI_COST_TOOLKIT_DASHBOARD
```

## Budget Alerting System

The toolkit includes a comprehensive budget alerting system to proactively monitor AI services spending and send notifications when thresholds are exceeded.

### Key Features
- **Multi-service monitoring** - Covers Cortex Functions, Cortex Analyst, Document AI, and Cortex Search
- **Flexible time periods** - Daily, weekly, or monthly budget periods
- **Selective monitoring** - Enable/disable specific services as needed
- **Multiple notification channels** - Slack, email, or other Snowflake notification integrations
- **Automated scheduling** - Tasks to run alerts at specified intervals
- **Tiered alerting** - Different thresholds for warnings vs critical alerts

### Quick Setup
1. **Create notification integration** (Slack or Email):
   ```sql
   CREATE NOTIFICATION INTEGRATION MY_SLACK_ALERTS
     TYPE=SLACK
     SLACK_URL='YOUR_SLACK_WEBHOOK_URL'
     ENABLED=TRUE;
   ```

2. **Test the alert system**:
   ```sql
   CALL AI_SERVICES_BUDGET_ALERT('MONTH', 1000, 'MY_SLACK_ALERTS');
   ```

3. **Set up automated monitoring**:
   ```sql
   CREATE OR REPLACE TASK AI_BUDGET_MONITOR
   WAREHOUSE = 'YOUR_WAREHOUSE'
   SCHEDULE = '1 hour'
   AS
   CALL AI_SERVICES_BUDGET_ALERT('MONTH', 5000, 'MY_SLACK_ALERTS');
   
   ALTER TASK AI_BUDGET_MONITOR RESUME;
   ```

### Available Procedures
- **`AI_SERVICES_BUDGET_ALERT()`** - Comprehensive monitoring across all AI services
- **`MINI_BUDGET_ON_CORTEX()`** - Cortex Functions specific monitoring (compatible with existing implementations)

For detailed examples and advanced configurations, see `budget_alerting_examples.sql`.

## Functions

### Core Functions

- **`fetch_semantic_model_paths(session)`**: Retrieves semantic model file paths from all agents
- **`get_cortex_analyst_logs(session, semantic_model_file)`**: Fetches Cortex Analyst logs for a specific semantic model
- **`write_logs_to_table(session, semantic_model_files, table_name)`**: Writes logs for multiple semantic models to a Snowflake table

### Analysis Functions

- **`verified_query_count(df)`**: Analyzes verified vs non-verified query breakdown
- **`top_verified_queries(df)`**: Gets the most frequently used verified queries
- **`slowest_queries(df, number=10)`**: Identifies the slowest queries by latency
- **`latency_summary_by_semantic_model(df)`**: Provides latency statistics by semantic model
- **`user_activity_by_semantic_model(df)`**: Analyzes user activity patterns
- **`semantic_model_usage_summary(df)`**: Summarizes overall semantic model usage

### Integration Functions

- **`create_sf_intelligence_query_history(session, target_table)`**: Creates a table joining query history with attribution data
- **`create_cortex_analyst_query_history(session, cortex_analyst_df, sf_intelligence_table)`**: Joins Cortex Analyst logs with Snowflake Intelligence query history

### Cost Analysis Functions

- **`total_cost_by_semantic_model(ca_query_history_df)`**: Calculates total combined credits (Cortex Analyst + Warehouse) by semantic model
- **`cost_breakdown_by_semantic_model(ca_query_history_df)`**: Provides detailed cost breakdown with separate Cortex Analyst and warehouse costs by semantic model

### LLM Usage Dashboard Functions

- **`get_ai_services_total_credits(session, start_date, end_date)`**: Get total AI Services credits used
- **`get_llm_inference_summary(session, start_date, end_date)`**: Get LLM inference credits and tokens for COMPLETE function
- **`get_cortex_analyst_summary(session, start_date, end_date)`**: Get Cortex Analyst credits and message count
- **`get_document_ai_credits(session, start_date, end_date)`**: Get Document AI total credits
- **`get_cortex_functions_total_credits(session, start_date, end_date)`**: Get total Cortex Functions credits
- **`get_credits_by_function(session, start_date, end_date)`**: Get credits breakdown by function name
- **`get_credits_by_model(session, start_date, end_date, limit=10)`**: Get credits and tokens by model for COMPLETE function
- **`get_credits_by_warehouse(session, start_date, end_date)`**: Get Cortex and compute credits by warehouse
- **`get_cortex_functions_query_history(session, start_date, end_date)`**: Get detailed Cortex functions query history
- **`get_cortex_analyst_requests_by_day(session, start_date, end_date)`**: Get daily Cortex Analyst request counts
- **`get_document_processing_metrics(session, start_date, end_date)`**: Get Document Processing summary metrics
- **`get_document_processing_by_day(session, start_date, end_date)`**: Get daily document processing metrics
- **`get_cortex_search_total_credits(session, start_date, end_date)`**: Get total Cortex Search credits
- **`get_cortex_search_by_service(session, start_date, end_date)`**: Get Cortex Search credits by service
- **`get_cortex_search_by_day(session, start_date, end_date)`**: Get daily Cortex Search credits

## Database Schema

The `setup.sql` script automatically creates the following key objects:

### Tables
- **`CORTEX_ANALYST_LOGS`** - Stores processed Cortex Analyst logs with computed fields
- **`CORTEX_ANALYTICS.PUBLIC.SF_INTELLIGENCE_QUERY_HISTORY`** - Cleaned query history joined with attribution data

### Views
- **`V_COST_SUMMARY_BY_SEMANTIC_MODEL`** - Cost summary aggregated by semantic model
- **`V_USER_ACTIVITY_BY_SEMANTIC_MODEL`** - User activity breakdown by semantic model  
- **`V_QUERY_TYPE_ANALYSIS`** - Analysis of verified vs non-verified queries

### Stored Procedures
- **`REFRESH_QUERY_HISTORY()`** - Refreshes the query history table with latest data
- **`AI_SERVICES_BUDGET_ALERT()`** - Comprehensive budget monitoring across all AI services
- **`MINI_BUDGET_ON_CORTEX()`** - Cortex Functions specific budget monitoring

### Tasks (Optional)
- **`AI_SERVICES_BUDGET_MONITOR`** - Automated comprehensive budget monitoring
- **`MINI_BUDGET_CORTEX_MONITOR`** - Automated Cortex Functions monitoring
- **`AI_SERVICES_DAILY_SUMMARY`** - Daily budget summary task

All schema objects are created automatically when you run the `setup.sql` script.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
