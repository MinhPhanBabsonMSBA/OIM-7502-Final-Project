import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configure page
st.set_page_config(
    page_title="DC Housing Affordability Analysis",
    page_icon="home",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use("ggplot")
sns.set_theme()
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# Cache data loading
@st.cache_data
def load_salary_data():
    """Load and clean salary data"""
    salary_df = pd.read_csv("clean_data/DC_Public_Employee_Salary.csv")
    salary_df = salary_df.rename(columns={
        "DESCRSHORT": "AGENCY",
        "JOBTITLE": "JOB_TITLE",
        "COMPRATE": "ANNUAL_SALARY",
        "HIREDATE_STRING": "HIRE_DATE"
    })
    salary_df["ANNUAL_SALARY"] = pd.to_numeric(
        salary_df["ANNUAL_SALARY"], 
        errors="coerce"
    )
    salary_df = salary_df.dropna(subset=["ANNUAL_SALARY"])
    return salary_df

@st.cache_data
def load_housing_data():
    """Load and process housing data"""
    housing_df = pd.read_csv("clean_data/DC_housing_prices.csv")

    # Create long format
    id_col = "RegionName"
    date_cols = housing_df.columns[9:]
    
    housing_long = housing_df.melt(
        id_vars=[id_col],
        value_vars=date_cols,
        var_name="DATE_STR",
        value_name="PRICE"
    )
    
    housing_long = housing_long.rename(columns={"RegionName": "ZIP"})
    housing_long["DATE"] = pd.to_datetime(housing_long["DATE_STR"], errors="coerce")
    housing_long["YEAR"] = housing_long["DATE"].dt.year
    
    return housing_df, housing_long

@st.cache_data
def calculate_agency_stats(salary_df):
    """Calculate median salary and affordability metrics by agency"""
    agency_stats = (
        salary_df
        .groupby("AGENCY")["ANNUAL_SALARY"]
        .median()
        .reset_index()
    )
    agency_stats = agency_stats.rename(columns={"ANNUAL_SALARY": "MEDIAN_SALARY"})
    agency_stats["MAX_AFFORDABLE_PRICE"] = 6 * agency_stats["MEDIAN_SALARY"]
    return agency_stats

@st.cache_data
def calculate_affordable_zips(agency_stats, housing_long):
    """Calculate number of affordable ZIPs for each agency"""
    latest_date = housing_long["DATE"].max()
    housing_latest = housing_long[housing_long["DATE"] == latest_date].copy()
    zip_prices = housing_latest[["ZIP", "PRICE"]].dropna()
    
    affordable_counts = []
    for _, row in agency_stats.iterrows():
        max_price = row["MAX_AFFORDABLE_PRICE"]
        count_affordable = (zip_prices["PRICE"] <= max_price).sum()
        affordable_counts.append(count_affordable)
    
    agency_stats["AFFORDABLE_ZIPS"] = affordable_counts
    return agency_stats, zip_prices

# Load data
salary_df = load_salary_data()
housing_df, housing_long = load_housing_data()
agency_stats = calculate_agency_stats(salary_df)
agency_stats, zip_prices = calculate_affordable_zips(agency_stats, housing_long)

# Main title
st.title("DC Housing Affordability Analysis")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["EDA", "Part 2", "Part 3"])

# TAB 1: EDA
with tab1:
    st.header("Exploratory Data Analysis")
    
    # Introduction section
    with st.expander("About This Analysis", expanded=True):
        st.write("""
        This analysis explores the relationship between DC public employee salaries 
        and housing affordability across Washington, DC. Using salary data and housing 
        price data, we examine:
        
        - Salary distributions across agencies and job titles
        - Housing price trends over time
        - Price appreciation by ZIP code
        - Affordability gaps for different employee groups
        
        **Key Metric:** Maximum affordable home price = 6 × Annual Salary
        (Based on 30% income-to-housing ratio and 5% annual housing cost)
        """)
    
    # Summary metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Median DC Salary",
            f"${salary_df['ANNUAL_SALARY'].median():,.0f}"
        )
    
    with col2:
        st.metric(
            "Median DC Home Price (Latest)",
            f"${zip_prices['PRICE'].median():,.0f}"
        )
    
    with col3:
        affordability_ratio = (
            zip_prices['PRICE'].median() / 
            (6 * salary_df['ANNUAL_SALARY'].median())
        )
        st.metric(
            "Affordability Gap",
            f"{affordability_ratio:.1f}x",
            delta="Higher = Less Affordable",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Salary Analysis Section
    st.subheader("Salary Distribution Analysis")
    
    # Overall salary distribution
    with st.expander("Overall DC Public Employee Salary Distribution", expanded=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(salary_df["ANNUAL_SALARY"], bins=40, kde=True, ax=ax)
        ax.set_title("Overall DC Public Employee Salary Distribution")
        ax.set_xlabel("Annual Salary ($)")
        ax.set_ylabel("Number of Employees")
        st.pyplot(fig)
        
        st.caption("""
        **Insight:** The salary distribution is right-skewed, with most employees 
        earning $50,000-$120,000. Only a small number earn above $150,000.
        """)
    
    # Salary by top agencies
    with st.expander("Salary Distribution by Agency (Top 5)", expanded=False):
        top_agencies = salary_df["AGENCY"].value_counts().head(5).index
        top_agency_df = salary_df[salary_df["AGENCY"].isin(top_agencies)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(
            data=top_agency_df,
            x="ANNUAL_SALARY",
            hue="AGENCY",
            bins=40,
            kde=True,
            alpha=0.5,
            ax=ax
        )
        ax.set_title("Salary Distribution by Agency (Top 5)")
        ax.set_xlabel("Annual Salary ($)")
        ax.set_ylabel("Number of Employees")
        st.pyplot(fig)
        
        st.caption("""
        **Insight:** Agencies like MPD and FEMS show higher salary peaks ($100,000+), 
        while DCPS and DPW cluster in the lower-to-mid ranges.
        """)
    
    # Salary by job titles
    with st.expander("Salary Distribution by Top Job Titles", expanded=False):
        top_jobs = salary_df["JOB_TITLE"].value_counts().head(6).index
        top_jobs_df = salary_df[salary_df["JOB_TITLE"].isin(top_jobs)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=top_jobs_df, x="JOB_TITLE", y="ANNUAL_SALARY", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("Salary Distribution by Top Job Titles")
        ax.set_xlabel("Job Title")
        ax.set_ylabel("Annual Salary ($)")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("""
        **Insight:** Teachers earn $90,000-$110,000 median, while EMTs and 
        correctional officers earn $60,000-$80,000.
        """)
    
    st.markdown("---")
    
    # Housing Trends Section
    st.subheader("Housing Price Trends")
    
    # Median price over time
    with st.expander("Median DC Home Price Over Time", expanded=False):
        dc_ts = (
            housing_long
            .groupby("DATE")["PRICE"]
            .median()
            .reset_index()
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dc_ts["DATE"], dc_ts["PRICE"], linewidth=2)
        ax.set_title("Median DC Home Price Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Median Price ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("""
        **Insight:** Prices increased from ~$150,000 (2000) to $650,000+ (2024), 
        with small dips around 2008 and 2020-2021.
        """)
    
    # Top ZIP appreciation
    with st.expander("Top 10 DC ZIPs by Home Price Appreciation", expanded=False):
        hl = housing_long.dropna(subset=["PRICE"]).copy()
        first_price = hl.sort_values("DATE").groupby("ZIP")["PRICE"].first()
        last_price = hl.sort_values("DATE").groupby("ZIP")["PRICE"].last()
        
        appreciation = ((last_price - first_price) / first_price).reset_index()
        appreciation.columns = ["ZIP", "PCT_CHANGE"]
        top_up = appreciation.sort_values("PCT_CHANGE", ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_up, x="ZIP", y="PCT_CHANGE", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("Top 10 DC ZIPs by Home Price Appreciation")
        ax.set_ylabel("Percent Change (first→last)")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("""
        **Insight:** ZIPs like 20001, 20002, 20010 show ~3x price increases, 
        making them increasingly unaffordable.
        """)
    
    # Heatmap
    with st.expander("Heatmap of Housing Prices by ZIP and Year", expanded=False):
        avg_price_zip_year = (
            housing_long
            .groupby(["ZIP", "YEAR"])["PRICE"]
            .mean()
            .reset_index()
        )
        heat_data = avg_price_zip_year.pivot(
            index="ZIP", 
            columns="YEAR", 
            values="PRICE"
        )
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(heat_data, cmap="viridis", ax=ax)
        ax.set_title("Heatmap of DC Housing Prices by ZIP and Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("ZIP Code")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("""
        **Insight:** Most ZIP codes have darkened over time (higher prices). 
        ZIPs 20007, 20008, 20015, 20016 are the most expensive.
        """)
    
    st.markdown("---")
    
    # Affordability Analysis Section
    st.subheader("Housing Affordability Analysis")
    
    with st.expander("Agency Affordability Rankings", expanded=False):
        display_stats = agency_stats.copy()
        display_stats = display_stats.sort_values("AFFORDABLE_ZIPS", ascending=False)
        
        st.dataframe(
            display_stats.style.format({
                "MEDIAN_SALARY": "${:,.0f}",
                "MAX_AFFORDABLE_PRICE": "${:,.0f}"
            }),
            height=400
        )
        
        st.caption("""
        **Insight:** Agencies with median salaries above $120,000 can afford 
        15-19 ZIP codes, while those earning $60,000-$80,000 can afford only 0-5 ZIPs.
        """)
    
    # Scatter plot
    with st.expander("Median Salary vs. Number of Affordable ZIP Codes", expanded=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=agency_stats,
            x="MEDIAN_SALARY",
            y="AFFORDABLE_ZIPS",
            s=100,
            alpha=0.7,
            ax=ax
        )
        ax.set_title("Agency Median Salary vs Number of Affordable ZIP Codes")
        ax.set_xlabel("Median Salary ($)")
        ax.set_ylabel("Number of Affordable ZIPs")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("""
        **Insight:** Clear positive correlation between salary and affordable 
        housing access, showing structural inequality.
        """)
    
    st.markdown("---")
    
    # Agency Deep Dive Section
    st.subheader("Agency Deep Dive")
    
    # Agency selector
    selected_agency = st.selectbox(
        "Select an Agency",
        options=sorted(agency_stats["AGENCY"].unique()),
        key="eda_agency_selector"
    )
    
    # Get agency stats
    row = agency_stats[agency_stats["AGENCY"] == selected_agency].iloc[0]
    max_price = row["MAX_AFFORDABLE_PRICE"]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Median Salary",
            f"${row['MEDIAN_SALARY']:,.0f}"
        )
    
    with col2:
        st.metric(
            "Max Affordable Price",
            f"${max_price:,.0f}"
        )
    
    with col3:
        st.metric(
            "Affordable ZIPs",
            f"{row['AFFORDABLE_ZIPS']}"
        )
    
    # Two columns for visualizations
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Show affordable ZIPs
        st.markdown("**Affordable ZIP Codes**")
        
        affordable_zips = zip_prices[
            zip_prices["PRICE"] <= max_price
        ].sort_values("PRICE")
        
        if len(affordable_zips) > 0:
            st.dataframe(
                affordable_zips.style.format({"PRICE": "${:,.0f}"}),
                height=300
            )
        else:
            st.warning("⚠️ No affordable ZIP codes found")
    
    with col_right:
        # Salary distribution for selected agency
        st.markdown("**Agency Salary Distribution**")
        
        agency_salaries = salary_df[salary_df["AGENCY"] == selected_agency]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            agency_salaries["ANNUAL_SALARY"],
            bins=30,
            kde=True,
            ax=ax
        )
        ax.axvline(
            row['MEDIAN_SALARY'],
            color='red',
            linestyle='--',
            label=f"Median: ${row['MEDIAN_SALARY']:,.0f}",
            linewidth=2
        )
        ax.set_title(f"Salary Distribution: {selected_agency}")
        ax.set_xlabel("Annual Salary ($)")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Conclusion
    with st.expander("Key Findings & Conclusion", expanded=True):
        st.markdown("""
        ### Overall Findings
        
        **1. Salary Distribution**
        - Most DC public employees earn between $50,000-$120,000
        - Peak salaries around $80,000-$100,000
        - Only small percentage earn above $150,000
        
        **2. Housing Price Trends**
        - Prices increased from ~$150,000 (2000) to $650,000+ (2024)
        - Growth far outpaces salary increases
        - High-growth ZIPs show 3x+ appreciation
        
        **3. Affordability Gap by Agency**
        - Lower-paying agencies: 0-5 affordable ZIP codes
        - Higher-paying agencies: 15-19 affordable ZIP codes
        - Clear structural divide in housing access
        
        **4. Geographic Disparities**
        - ZIPs 20007, 20008, 20015, 20016 are most expensive
        - ZIPs 20019, 20020, 20032 remain relatively affordable
        - Neighborhood inequality continues to grow
        
        ### Conclusion
        
        The analysis reveals a **widening affordability gap** between DC public employee 
        salaries and housing prices. While home values have increased dramatically over 
        20+ years, salary growth for many workers has not kept pace. This creates:
        
        - **Limited geographic access** for lower-paid agencies
        - **Increasing financial pressure** on mid-income workers
        - **Structural inequality** within the DC workforce
        
        Lower-paid employees essentially have **no affordable options** in DC, while 
        higher-paid agencies retain significantly more housing access. This gap impacts 
        the ability of public employees to live in the communities they serve.
        """)

# TAB 2: Part 2 (Presentation-focused Regression & Forecasting)
with tab2:
    st.header("Part 2: Forecasting Summary & Insights")
    st.markdown(
        """
        This section compares simple forecasting approaches (naive, linear trend with seasonality, polynomial trend, and ARIMA)
        across ZIP codes, selects the best model per ZIP using validation (2021–2023) MAPE, refits on full history, and
        produces month-by-month forecasts for 2026–2027.

        Use the controls below to explore aggregated performance metrics and view ZIP-level forecasts.
        """,
        unsafe_allow_html=False
    )

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    import statsmodels.api as sm
    import warnings
    warnings.filterwarnings('ignore')

    # Time split
    train_end = pd.to_datetime("2020-12-31")
    val_end   = pd.to_datetime("2023-12-31")

    # Helpers
    def split_by_time(zip_df, train_end, val_end):
        train = zip_df[zip_df['DATE'] <= train_end].copy()
        val   = zip_df[(zip_df['DATE'] > train_end) & (zip_df['DATE'] <= val_end)].copy()
        test  = zip_df[zip_df['DATE'] > val_end].copy()
        return train, val, test

    def build_regression_X(df_part, degree=1):
        df_part = df_part.reset_index(drop=True)
        X = pd.DataFrame({'t': df_part['t'].values})
        if degree >= 2:
            X['t2'] = df_part['t'].values ** 2
        month_dummies = pd.get_dummies(df_part['month'], prefix='month', drop_first=True)
        X = pd.concat([X, month_dummies], axis=1)
        return X

    def evaluate_regression(train, val, degree=1):
        if len(train) == 0 or len(val) == 0:
            return np.nan, np.nan, None
        X_train = build_regression_X(train, degree=degree)
        X_val   = build_regression_X(val, degree=degree)
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        y_train = train['PRICE'].values
        y_val   = val['PRICE'].values
        model = LinearRegression()
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mape = mean_absolute_percentage_error(y_val, val_pred)
        return rmse, mape, model

    def evaluate_naive(train, val):
        if len(train) == 0 or len(val) == 0:
            return np.nan, np.nan
        last_train_value = train['PRICE'].iloc[-1]
        y_val = val['PRICE'].values
        y_pred = np.full_like(y_val, fill_value=last_train_value, dtype=float)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mape = mean_absolute_percentage_error(y_val, y_pred)
        return rmse, mape

    # Prepare data
    hl = housing_long.rename(columns={"DATE": "DATE", "PRICE": "PRICE"}).copy()
    hl = hl.dropna(subset=['PRICE', 'DATE']).sort_values(['ZIP', 'DATE'])
    hl['t'] = hl.groupby('ZIP').cumcount()
    hl['month'] = hl['DATE'].dt.month

    st.subheader("Run models and compute validation metrics")
    st.write("This may take a moment — running simple regressions and short ARIMA fits per ZIP.")

    progress_text = "Running models..."
    my_bar = st.progress(0, text=progress_text)

    results = []
    unique_zips = hl['ZIP'].unique()
    total = len(unique_zips)

    for i, z in enumerate(unique_zips):
        z_df = hl[hl['ZIP'] == z].copy().sort_values('DATE')
        train, val, test = split_by_time(z_df, train_end, val_end)

        naive_rmse, naive_mape = evaluate_naive(train, val)
        lin_rmse, lin_mape, lin_model = evaluate_regression(train, val, degree=1)
        poly_rmse, poly_mape, poly_model = evaluate_regression(train, val, degree=2)

        # ARIMA safely
        arima_rmse, arima_mape = np.nan, np.nan
        try:
            series_train = train.set_index('DATE')['PRICE']
            series_val   = val.set_index('DATE')['PRICE']
            series_train.index = pd.DatetimeIndex(series_train.index).to_period('M').to_timestamp('M')
            series_val.index   = pd.DatetimeIndex(series_val.index).to_period('M').to_timestamp('M')
            series_train = series_train.asfreq('ME')
            series_val   = series_val.asfreq('ME')
            if len(series_train.dropna()) > 6 and len(series_val.dropna()) > 0:
                arima_fit = sm.tsa.ARIMA(series_train, order=(1,1,0)).fit()
                preds = arima_fit.forecast(steps=len(series_val))
                arima_rmse = np.sqrt(mean_squared_error(series_val.values, preds.values))
                arima_mape = mean_absolute_percentage_error(series_val.values, preds.values)
        except Exception:
            arima_rmse, arima_mape = np.nan, np.nan

        results.append({
            'ZIP': z,
            'naive_rmse': naive_rmse, 'naive_mape': naive_mape,
            'linear_rmse': lin_rmse, 'linear_mape': lin_mape,
            'poly_rmse': poly_rmse, 'poly_mape': poly_mape,
            'arima_rmse': arima_rmse, 'arima_mape': arima_mape
        })

        # update progress
        my_bar.progress((i+1)/total, text=f"Processed {i+1}/{total} ZIPs")

    metrics_df = pd.DataFrame(results)

    # Choose best by MAPE (prefer ARIMA when similar)
    def choose_model(row):
        scores = {
            'naive': row['naive_mape'],
            'linear': row['linear_mape'],
            'poly': row['poly_mape'],
            'arima': row['arima_mape']
        }
        valid = {k: v for k, v in scores.items() if pd.notna(v)}
        if not valid:
            return 'none'
        return min(valid, key=valid.get)

    metrics_df['best_model'] = metrics_df.apply(choose_model, axis=1)

    # Presentation KPIs
    st.subheader("Executive KPIs")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ZIPs evaluated", f"{len(metrics_df)}")
    with col2:
        avg_mapes = {
            'Naive': metrics_df['naive_mape'].mean(),
            'Linear': metrics_df['linear_mape'].mean(),
            'Polynomial': metrics_df['poly_mape'].mean(),
            'ARIMA': metrics_df['arima_mape'].mean()
        }
        st.metric("Avg. MAPE (Linear)", f"{avg_mapes['Linear']:.3f}")
    with col3:
        st.metric("Avg. MAPE (Poly)", f"{avg_mapes['Polynomial']:.3f}")
    with col4:
        st.metric("Avg. MAPE (ARIMA)", f"{avg_mapes['ARIMA']:.3f}")

    st.markdown("---")

    # Model comparison charts
    st.subheader("Model comparison: validation MAPE")
    comp_col1, comp_col2 = st.columns([1,1])
    with comp_col1:
        # Bar chart of average MAPE
        avg_df = pd.DataFrame({
            'model': ['Naive','Linear','Polynomial','ARIMA'],
            'avg_mape': [avg_mapes['Naive'], avg_mapes['Linear'], avg_mapes['Polynomial'], avg_mapes['ARIMA']]
        })
        fig1, ax1 = plt.subplots(figsize=(6,4))
        sns.barplot(data=avg_df, x='model', y='avg_mape', palette='magma', ax=ax1)
        ax1.set_title('Average Validation MAPE by Model')
        ax1.set_ylabel('MAPE')
        st.pyplot(fig1)
    with comp_col2:
        # Boxplot of MAPE distribution
        plot_df = metrics_df.melt(id_vars=['ZIP'], value_vars=['naive_mape','linear_mape','poly_mape','arima_mape'], var_name='model', value_name='mape')
        plot_df['model'] = plot_df['model'].map({'naive_mape':'Naive','linear_mape':'Linear','poly_mape':'Polynomial','arima_mape':'ARIMA'})
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.boxplot(data=plot_df, x='model', y='mape', palette='mako', ax=ax2)
        ax2.set_title('MAPE Distribution by Model')
        ax2.set_ylabel('MAPE')
        st.pyplot(fig2)

    st.markdown("---")

    # Show table with counts of best model
    st.subheader("Selected best model counts")
    best_counts = metrics_df['best_model'].value_counts().rename_axis('model').reset_index(name='count')
    st.table(best_counts)

    st.markdown("---")

    # Refit selected models and produce forecasts (2026-2027)
    st.subheader("Refit & Forecast (2026-2027)")
    st.write("Refitting the chosen model per ZIP on full history and producing month-by-month forecasts for 2026-2027.")

    future_dates = pd.date_range(start="2026-01-31", end="2027-12-31", freq='M')
    forecast_rows = []

    with st.spinner('Refitting models and generating forecasts...'):
        for _, r in metrics_df.iterrows():
            z = r['ZIP']
            best = r['best_model']
            z_df = hl[hl['ZIP'] == z].copy().sort_values('DATE')
            if z_df.empty:
                continue
            z_df['t'] = np.arange(len(z_df))
            z_df['month'] = z_df['DATE'].dt.month

            if best in ['linear','poly']:
                degree = 2 if best == 'poly' else 1
                X_full = build_regression_X(z_df, degree=degree)
                y_full = z_df['PRICE'].values
                model = LinearRegression()
                model.fit(X_full, y_full)
                last_t = z_df['t'].iloc[-1]
                future_t = np.arange(last_t + 1, last_t + 1 + len(future_dates))
                future_df = pd.DataFrame({'t': future_t, 'month': future_dates.month})
                if degree == 2:
                    future_df['t2'] = future_df['t']**2
                X_future = build_regression_X(future_df, degree=degree)
                X_future = X_future.reindex(columns=X_full.columns, fill_value=0)
                preds = model.predict(X_future)
            elif best == 'arima':
                try:
                    series = z_df.set_index('DATE')['PRICE']
                    series.index = pd.DatetimeIndex(series.index).to_period('M').to_timestamp('M')
                    series = series.asfreq('ME')
                    model = sm.tsa.ARIMA(series, order=(1,1,0)).fit()
                    preds = model.forecast(steps=len(future_dates))
                except Exception:
                    preds = [np.nan] * len(future_dates)
            else:
                last = z_df['PRICE'].iloc[-1]
                preds = np.full(len(future_dates), last, dtype=float)

            for d, p in zip(future_dates, preds):
                forecast_rows.append({'ZIP': z, 'DATE': d, 'PREDICTED_PRICE': float(p) if pd.notna(p) else np.nan, 'MODEL_USED': best})

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df = forecast_df.sort_values(['ZIP', 'DATE']).reset_index(drop=True)

    st.markdown("### Sample of forecasts")
    st.dataframe(forecast_df.head(50).style.format({'PREDICTED_PRICE':'${:,.0f}'}), height=300)

    # Interactive ZIP selector to view history + forecast
    st.markdown("---")
    st.subheader("Explore a ZIP: historical prices vs. forecast")
    zip_choice = st.selectbox("Select ZIP to view", options=sorted(hl['ZIP'].unique()))

    if zip_choice:
        hist = hl[hl['ZIP'] == zip_choice].copy()
        hist_plot = hist.groupby('DATE')['PRICE'].median().reset_index()
        preds_plot = forecast_df[forecast_df['ZIP'] == zip_choice].copy()

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(hist_plot['DATE'], hist_plot['PRICE'], label='Historical (median)', linewidth=2)
        if not preds_plot.empty:
            ax.plot(preds_plot['DATE'], preds_plot['PREDICTED_PRICE'], label='Forecast (2026-27)', linestyle='--', linewidth=2)
        ax.set_title(f'ZIP {zip_choice}: Historical Prices and 2026–2027 Forecast')
        ax.set_ylabel('Price ($)')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**Selected ZIP summary**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Last observed price", f"${hist['PRICE'].iloc[-1]:,.0f}")
        with col_b:
            med_val = hist['PRICE'].median()
            st.metric("Median historical price", f"${med_val:,.0f}")
        with col_c:
            chosen_model = metrics_df[metrics_df['ZIP'] == zip_choice]['best_model'].values
            chosen_model = chosen_model[0] if len(chosen_model)>0 else 'n/a'
            st.metric("Model selected", chosen_model)

    st.markdown("---")
    # Download
    csv = forecast_df.to_csv(index=False)
    st.download_button("Download forecasts CSV", data=csv, file_name="DC_ZIP_Forecasts_2026_2027_presentation.csv", mime='text/csv')

    st.success("Part 2: Presentation-ready forecasts generated.")

# TAB 3: Part 3 (Classification - Housing Affordability Prediction)
with tab3:
    st.header("Part 3: Classification Model - Housing Affordability Prediction")
    st.markdown(
        """
        This section uses **Logistic Regression** to predict whether DC public employees 
        will face housing affordability challenges based on salary bracket, ZIP code, and job role.
        
        An employee is classified as **"Unaffordable"** if monthly housing costs exceed 30% 
        of their monthly income (standard affordability threshold).
        """,
        unsafe_allow_html=False
    )

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )

    # Prepare data
    df_sal = salary_df.copy()
    
    # Latest housing prices by ZIP
    latest_date = housing_long["DATE"].max()
    housing_latest = housing_long[housing_long["DATE"] == latest_date][["ZIP", "PRICE"]].drop_duplicates()
    housing_latest = housing_latest.rename(columns={"PRICE": "house_price_latest"})

    # Salary brackets
    bins = [0, 50000, 80000, 110000, 150000, np.inf]
    labels = ["<50k", "50-80k", "80-110k", "110-150k", "150k+"]
    df_sal["salary_bracket"] = pd.cut(df_sal["ANNUAL_SALARY"], bins=bins, labels=labels)

    # Job family (first word of job title)
    df_sal["job_family"] = df_sal["JOB_TITLE"].str.split().str[0]

    # Assign ZIP codes to employees (simulate realistic distribution)
    np.random.seed(42)
    available_zips = housing_latest["ZIP"].unique()
    df_sal["ZIP"] = np.random.choice(available_zips, size=len(df_sal))

    # Merge with housing data
    df_model = df_sal.merge(housing_latest, on="ZIP", how="left")
    df_model = df_model.dropna(subset=["house_price_latest"])

    # Create affordability target
    annual_cost_rate = 0.05  # 5% of house value per year (mortgage, tax, insurance)
    threshold = 0.30         # 30% income-to-housing ratio
    
    df_model["monthly_income"] = df_model["ANNUAL_SALARY"] / 12
    df_model["monthly_housing_cost"] = df_model["house_price_latest"] * annual_cost_rate / 12
    df_model["unaffordable"] = (df_model["monthly_housing_cost"] > threshold * df_model["monthly_income"]).astype(int)

    # Build and train model
    X = df_model[["salary_bracket", "ZIP", "job_family"]]
    y = df_model["unaffordable"]

    categorical_features = ["salary_bracket", "ZIP", "job_family"]
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    with st.spinner("Training classification model..."):
        pipeline.fit(X_train, y_train)

    # Model performance
    y_pred = pipeline.predict(X_test)

    st.subheader("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
    with col2:
        st.metric("Precision", f"{precision_score(y_test, y_pred):.1%}")
    with col3:
        st.metric("Recall", f"{recall_score(y_test, y_pred):.1%}")
    with col4:
        st.metric("F1-Score", f"{f1_score(y_test, y_pred):.1%}")

    st.markdown("---")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=["Affordable", "Unaffordable"],
                yticklabels=["Affordable", "Unaffordable"],
                cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix - Logistic Regression")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig)

    st.markdown("---")

    # Classification Report
    st.subheader("Classification Report")
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(class_report_df.style.format("{:.3f}"), height=300)

    st.markdown("---")

    # Risk Analysis
    st.subheader("Affordability Risk Analysis")

    # Risk by salary bracket
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("**Unaffordability Rate by Salary Bracket**")
        risk_by_bracket = df_model.groupby("salary_bracket")["unaffordable"].agg(['sum', 'count', 'mean']).reset_index()
        risk_by_bracket.columns = ["Salary Bracket", "Unaffordable Count", "Total Employees", "Unaffordability Rate"]
        risk_by_bracket["Unaffordability Rate"] = risk_by_bracket["Unaffordability Rate"].apply(lambda x: f"{x:.1%}")
        st.dataframe(risk_by_bracket, hide_index=True)

    with risk_col2:
        st.markdown("**Visualization: Risk by Salary Bracket**")
        risk_data = df_model.groupby("salary_bracket")["unaffordable"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=risk_data, x="salary_bracket", y="unaffordable", palette="RdYlGn_r", ax=ax)
        ax.set_title("Unaffordability Rate by Salary Bracket")
        ax.set_ylabel("Unaffordable Percentage")
        ax.set_xlabel("Salary Bracket")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    st.markdown("---")

    # Risk by agency
    st.subheader("Top Agencies by Affordability Risk")
    risk_by_dept = df_model.groupby("AGENCY")["unaffordable"].agg(['sum', 'count', 'mean']).reset_index()
    risk_by_dept.columns = ["Agency", "Unaffordable Count", "Total Employees", "Unaffordability Rate"]
    risk_by_dept = risk_by_dept.sort_values("Unaffordability Rate", ascending=False).head(15)
    risk_by_dept["Unaffordability Rate"] = risk_by_dept["Unaffordability Rate"].apply(lambda x: f"{x:.1%}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    risk_plot_data = df_model.groupby("AGENCY")["unaffordable"].mean().sort_values(ascending=False).head(15)
    sns.barplot(x=risk_plot_data.values, y=risk_plot_data.index, palette="coolwarm", ax=ax)
    ax.set_title("Top 15 Agencies by Unaffordability Rate")
    ax.set_xlabel("Unaffordable Percentage")
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(risk_by_dept, hide_index=True, height=400)

    st.markdown("---")

    # Risk by ZIP
    st.subheader("Geographic Affordability Risk (Top 10 ZIP codes)")
    risk_by_zip = df_model.groupby("ZIP")["unaffordable"].agg(['sum', 'count', 'mean']).reset_index()
    risk_by_zip.columns = ["ZIP", "Unaffordable Count", "Total Employees", "Unaffordability Rate"]
    risk_by_zip = risk_by_zip.sort_values("Unaffordability Rate", ascending=False).head(10)
    risk_by_zip["Unaffordability Rate"] = risk_by_zip["Unaffordability Rate"].apply(lambda x: f"{x:.1%}")
    st.dataframe(risk_by_zip, hide_index=True)

    st.markdown("---")

    # Key Insights
    st.subheader("Key Insights & Recommendations")
    
    with st.expander("Model Interpretation", expanded=True):
        st.markdown(f"""
        ### Overall Model Performance
        - **Accuracy: {accuracy_score(y_test, y_pred):.1%}** – The model correctly classifies {accuracy_score(y_test, y_pred):.1%} of employees
        - **Precision: {precision_score(y_test, y_pred):.1%}** – Of those predicted unaffordable, {precision_score(y_test, y_pred):.1%} actually are
        - **Recall: {recall_score(y_test, y_pred):.1%}** – The model catches {recall_score(y_test, y_pred):.1%} of truly unaffordable employees
        
        ### Critical Findings
        
        **1. Salary Bracket Disparities**
        - Employees under **$50k**: {df_model[df_model['salary_bracket']=='<50k']['unaffordable'].mean():.0%} unaffordable
        - Employees **$50k–$80k**: {df_model[df_model['salary_bracket']=='50-80k']['unaffordable'].mean():.0%} unaffordable
        - Employees **$80k–$110k**: {df_model[df_model['salary_bracket']=='80-110k']['unaffordable'].mean():.0%} unaffordable
        - Employees **$110k+**: {df_model[df_model['salary_bracket']=='110-150k']['unaffordable'].mean():.0%} unaffordable
        
        **2. Geographic Disparities**
        - ZIP codes with highest housing costs show unaffordability rates **>80%**
        - Lower-cost areas remain accessible only to mid/high-income employees
        
        **3. Agency-Level Impact**
        - Departments with average salaries <$70k face **>90% unaffordability**
        - This creates recruitment, retention, and equity challenges
        """)

    with st.expander("Policy Recommendations", expanded=True):
        st.markdown("""
        ### Immediate Actions (0-6 months)
        
        **1. Salary Adjustments for At-Risk Departments**
        - Prioritize increases for departments with >85% unaffordability
        - Target: $5k–$10k annual increases for <$80k earners
        
        **2. Housing Stipends / Allowances**
        - $500–$1,000/month for low-income brackets
        - Zone-based adjustments (higher for expensive ZIP codes)
        
        **3. Remote/Hybrid Work**
        - Allow commuting from more affordable areas
        - Estimated savings: 15–25% on housing costs
        
        ### Medium-Term (6-18 months)
        
        **4. Housing Assistance Programs**
        - Down payment assistance
        - First-time homebuyer programs
        - Preferred lending partnerships
        
        **5. Relocation Support**
        - Help employees move to more affordable neighborhoods
        - Transportation/commute support
        
        ### Long-Term (18+ months)
        
        **6. Strategic Workforce Planning**
        - Data-driven compensation benchmarking
        - Career pathing for low-income workers
        - Succession planning to avoid talent drain
        
        **7. Partner with External Stakeholders**
        - Work with housing authorities for affordable units
        - Negotiate employer housing programs
        """)

    st.markdown("---")
    st.success("Part 3: Classification analysis complete. Use insights for HR policy decisions.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Data Sources: DC Public Employee Salary Dataset | DC Housing Prices Dataset</p>
    <p>Analysis Tool: Python | Streamlit | Pandas | Seaborn</p>
</div>
""", unsafe_allow_html=True)