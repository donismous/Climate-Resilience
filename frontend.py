import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Climate Resilience Dashboard", page_icon="🌍", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_real_data():
    # Load the real ND-GAIN dataset
    file_path = "raw_data/gain_long_format.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Rename 'Rate' to 'Risk Score' and 'Name' to 'Country' for consistency
        df = df.rename(columns={"Rate": "Risk Score", "Name": "Country"})
        # Drop rows with missing Risk Scores to avoid plotting errors
        df = df.dropna(subset=["Risk Score"])
        return df
    else:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()

df = load_real_data()

# --- SIDEBAR: GLOBAL CONTROLS ---
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Filter the ND-GAIN data and projections.")

if not df.empty:
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())

    # Year slider based on the actual dataset
    selected_year = st.sidebar.slider("Select Year", min_value=min_year, max_value=max_year, value=max_year)
    forecast_scenario = st.sidebar.radio("Forecast Scenario", ["Optimistic", "Base", "Pessimistic"], index=1)

    # Pre-select top 5 worsening countries if they exist in the data
    default_countries = ["Somalia", "South Sudan", "Chad", "Nigeria", "Bangladesh"]
    available_defaults = [c for c in default_countries if c in df['Country'].unique()]
    if not available_defaults:
        available_defaults = df['Country'].unique()[:5].tolist()

    selected_countries = st.sidebar.multiselect(
        "Select Countries to Compare",
        df['Country'].unique(),
        default=available_defaults
    )

    # --- MAIN DASHBOARD HEADER & KPIs ---
    st.title("🌍 Climate Resilience Dashboard")
    st.markdown("Tracking and forecasting climate vulnerability using ND-GAIN indicators.")
    st.divider()

    # Filter data for the selected year and countries
    year_data = df[df['Year'] == selected_year]
    selected_data = df[(df['Year'] == selected_year) & (df['Country'].isin(selected_countries))]

    # KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        global_avg = year_data['Risk Score'].mean()
        st.metric(label=f"Global Avg Risk Score ({selected_year})", value=f"{global_avg:.1f}")
    with col2:
        st.metric(label="Countries Monitored", value=str(df['Country'].nunique()), delta="ND-GAIN Data")
    with col3:
        highest_risk = year_data.loc[year_data['Risk Score'].idxmax()] if not year_data.empty else None
        if highest_risk is not None:
            st.metric(label="Highest Risk Country", value=highest_risk['Country'], delta=f"{highest_risk['Risk Score']:.1f}", delta_color="inverse")
    with col4:
        st.metric(label="Critical Alert", value="32 Countries", delta="Crossed 80+ Threshold", delta_color="inverse")

    st.divider()

    # --- TABS FOR NAVIGATION ---
    tab_map, tab_trends, tab_alerts = st.tabs([
        "🗺️ World Map View",
        "📈 Trend Explorer & Forecasts",
        "🚨 Alert Tracker & Attribution"
    ])

    # --- TAB 1: WORLD MAP VIEW ---
    with tab_map:
        st.subheader(f"Global Composite Risk Scores ({selected_year})")

        # Interactive Plotly Map using real data
        fig_map = px.choropleth(
            year_data,
            locations="ISO3",
            color="Risk Score",
            hover_name="Country",
            color_continuous_scale=px.colors.sequential.OrRd
        )
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, geo=dict(showocean=True, oceancolor="LightBlue"))
        st.plotly_chart(fig_map, use_container_width=True)

    # --- TAB 2: TREND EXPLORER ---
    with tab_trends:
        st.subheader("Evolution & Forecast")

        # Filter data for the line chart
        trend_data = df[df['Country'].isin(selected_countries)]

        if not trend_data.empty:
            fig_trend = px.line(
                trend_data,
                x="Year",
                y="Risk Score",
                color="Country",
                markers=True,
                title="Composite Risk Score Trajectories"
            )
            fig_trend.update_layout(hovermode="x unified")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("Please select at least one country to view the trend.")

    # --- TAB 3: ALERT TRACKER & ATTRIBUTION ---
    with tab_alerts:
        col_table, col_chart = st.columns([1.2, 1])

        with col_table:
            st.subheader("Alert Tracker")
            st.caption(f"Scores for {selected_year}")

            # Show a table of top highest risk scores for the selected year
            top_risks = year_data.sort_values(by="Risk Score", ascending=False).head(10)
            st.dataframe(top_risks[['Country', 'ISO3', 'Risk Score']].reset_index(drop=True), use_container_width=True)

        with col_chart:
            st.subheader("Feature Importance")
            st.caption("Global drivers of risk score variance")

            # Setting Default values for now - awaiting API development from Hugo
            features = pd.DataFrame({
                "Indicator": ["Food Supply", "Clean Water", "Flood Exposure", "Governance", "Health System"],
                "Importance": [0.19, 0.16, 0.15, 0.13, 0.11]
            }).sort_values(by="Importance", ascending=True)

            fig_bar = px.bar(features, x="Importance", y="Indicator", orientation='h', color="Importance", color_continuous_scale="Teal")
            fig_bar.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
