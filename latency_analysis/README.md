# Cortex Analyst Latency Analysis Report

A Streamlit application for analyzing latency issues in Cortex Analyst-powered applications. This tool helps diagnose performance bottlenecks by separating SQL generation latency from SQL execution latency.

## Features

- **Dual Data Source Support**:
  - Analyze specific semantic model using `SNOWFLAKE.LOCAL.CORTEX_ANALYST_REQUESTS()`
  - Analyze all semantic models using `SNOWFLAKE.LOCAL.CORTEX_ANALYST_REQUESTS_V` (requires ACCOUNTADMIN)

- **Three-Tab Analysis**:
  - **Cortex Analyst Analysis**: SQL generation latency metrics with semantic model filtering
  - **SQL Execution Analysis**: Query execution performance with query tag and warehouse filtering
  - **Combined Analysis**: Experimental joined view with latency breakdown pie chart

- **Key Metrics**:
  - Average, median, and max latencies for both SQL generation and execution
  - Slowest queries with Request IDs, Query IDs, and full SQL text
  - Query tag distribution for application-level filtering
  - Warehouse distribution analysis

- **Downloadable HTML Reports**: Export all analysis sections for sharing with colleagues

## Deployment Options

### Option 1: Streamlit in Snowflake (Recommended)

#### Prerequisites
- Snowflake account with Streamlit in Snowflake enabled
- Required role access (see below)
- Snowflake CLI installed

#### Role Requirements
- **For all semantic models**: Use `ACCOUNTADMIN` role to access `SNOWFLAKE.LOCAL.CORTEX_ANALYST_REQUESTS_V`
- **For specific semantic model**: Use any role with access to the semantic model stage file

**Important**: Streamlit in Snowflake does NOT use secondary roles. Ensure your execution role has the necessary privileges.

#### Required Packages
Add to your `environment.yml` or specify during deployment:
```yaml
dependencies:
  - streamlit
  - snowflake-snowpark-python
  - pandas
  - plotly
```

#### Deployment Steps

1. **Navigate to the latency_analysis directory**:
   ```bash
   cd latency_analysis
   ```

2. **Deploy using Snowflake CLI**:
   ```bash
   snow streamlit deploy \
     --file cortex_analyst_latency_report.py \
     --database <YOUR_DATABASE> \
     --schema <YOUR_SCHEMA> \
     --role <ACCOUNTADMIN_OR_APPROPRIATE_ROLE> \
     --replace
   ```

3. **Access the app** in Snowsight UI under Data > Streamlit Apps

#### Example Deployment Command
```bash
snow streamlit deploy \
  --file cortex_analyst_latency_report.py \
  --database ANALYTICS \
  --schema PUBLIC \
  --role ACCOUNTADMIN \
  --replace
```

### Option 2: Local Streamlit Development

#### Prerequisites
- Python 3.8+
- Snowflake connection configured
- Local Streamlit installation

#### Setup

1. **Install required packages**:
   ```bash
   pip install streamlit snowflake-connector-python snowflake-snowpark-python pandas plotly
   ```

2. **Create a `config.env` file** in the project root:
   ```bash
   # Example config.env
   SNOWFLAKE_ACCOUNT=<your_account>
   SNOWFLAKE_USER=<your_user>
   SNOWFLAKE_PASSWORD=<your_password>
   SNOWFLAKE_ROLE=<ACCOUNTADMIN_OR_APPROPRIATE_ROLE>
   SNOWFLAKE_WAREHOUSE=<your_warehouse>
   SNOWFLAKE_DATABASE=<your_database>
   SNOWFLAKE_SCHEMA=<your_schema>
   ```

   Or use Snowflake connection name:
   ```bash
   SNOWFLAKE_CONNECTION_NAME=<your_connection_name>
   ```

3. **Run the application**:
   ```bash
   streamlit run cortex_analyst_latency_report.py
   ```

4. **Access the app** at `http://localhost:8501`

## Usage Guide

### Data Source Selection
- **Specific Semantic Model**: Analyzes requests for a single semantic model file
- **All Models (ACCOUNTADMIN)**: Aggregates data across all semantic models in your account

### Date Range Selection
Use quick buttons (7/30/60/90 days) or custom date range picker to filter analysis period.

### Tab 1: Cortex Analyst Analysis
- View average SQL generation latency metrics
- Filter by semantic model when using "All Models" mode
- See slowest requests with generated SQL text
- Analyze semantic model distribution

### Tab 2: SQL Execution Analysis
- View SQL execution performance metrics
- Filter by query tag (application identifier)
- Filter by warehouse
- See slowest queries with executed SQL text
- Analyze query tag and warehouse distributions

### Tab 3: Combined Analysis (Experimental)
- ⚠️ Note: SQL text matching can be inaccurate. Use Tabs 1 & 2 for reliable analysis.
- View pie chart breakdown of SQL generation vs execution latency
- See combined slowest queries with both generated and executed SQL

### Exporting Reports
Click "📥 Download HTML Report" to generate a comprehensive HTML file containing all analysis sections for sharing.

## Troubleshooting

### "No data found" errors
- Verify your role has access to the required views/functions
- Check date range includes periods with Cortex Analyst activity
- For specific model mode, ensure semantic model file path is correct

### Permission errors accessing CORTEX_ANALYST_REQUESTS_V
- This view requires ACCOUNTADMIN or CORTEX_ANALYST_REQUESTS_ADMIN role
- Use "Specific Semantic Model" mode with a role that has stage file access instead

### Query History filtering issues
- Ensure queries contain "-- Generated by Cortex Analyst" comment
- Verify ACCOUNT_USAGE.QUERY_HISTORY latency (up to 45 minutes for recent queries)

## Technical Details

### Data Sources
- **Cortex Analyst Logs**: `SNOWFLAKE.LOCAL.CORTEX_ANALYST_REQUESTS()` or `SNOWFLAKE.LOCAL.CORTEX_ANALYST_REQUESTS_V`
- **Query Execution**: `SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY`
- **Semantic Models**: Retrieved from `SNOWFLAKE_INTELLIGENCE.AGENTS` schema

### Key Columns
- `analyst_latency_ms`: Time taken for Cortex Analyst to generate SQL
- `total_elapsed_time`: Time taken to execute the generated SQL query
- `query_tag`: Application identifier for filtering
- `warehouse_name`: Warehouse used for query execution

### SQL Normalization
Combined analysis uses SQL text matching by normalizing queries (removing comments/whitespace). This can be inaccurate, hence the experimental warning.

## Support

For issues or questions:
- Check Snowflake's Cortex Analyst documentation: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst
- Review ACCOUNT_USAGE views documentation: https://docs.snowflake.com/en/sql-reference/account-usage
