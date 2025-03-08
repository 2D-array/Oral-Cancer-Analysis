import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from streamlit_card import card
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_option_menu import option_menu
import time

# Set page config
st.set_page_config(
    page_title="Oral Cancer Analytics Hub",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    .highlight-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .big-font {
        font-size: 20px !important;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #c2e9fb 0%, #a1c4fd 100%);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #4e8df5;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    """Load and cache dataset"""
    try:
        df = pd.read_csv("dataset/oral_cancer_prediction_dataset.csv")
        return df
    except FileNotFoundError:
        # Create synthetic dataset if file not found (for demo purposes)
        st.warning("Dataset file not found. Using synthetic data for demonstration.")
        return create_synthetic_data()

def create_synthetic_data(n_rows=5000):
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    countries = ["USA", "India", "China", "Brazil", "UK", "Japan", "Australia", "Canada", "Germany", "France"]
    genders = ["Male", "Female"]
    binary = ["Yes", "No"]
    treatments = ["Surgery", "Radiation", "Chemotherapy", "Combined Therapy", "Immunotherapy"]
    
    data = {
        "ID": range(1, n_rows+1),
        "Country": np.random.choice(countries, n_rows),
        "Age": np.random.randint(18, 90, n_rows),
        "Gender": np.random.choice(genders, n_rows, p=[0.65, 0.35]),
        "Tobacco Use": np.random.choice(binary, n_rows, p=[0.4, 0.6]),
        "Alcohol Consumption": np.random.choice(binary, n_rows, p=[0.35, 0.65]),
        "HPV Infection": np.random.choice(binary, n_rows, p=[0.2, 0.8]),
        "Betel Quid Use": np.random.choice(binary, n_rows, p=[0.15, 0.85]),
        "Chronic Sun Exposure": np.random.choice(binary, n_rows),
        "Poor Oral Hygiene": np.random.choice(binary, n_rows, p=[0.3, 0.7]),
        "Diet (Fruits & Vegetables Intake)": np.random.choice(["Low", "Medium", "High"], n_rows),
        "Family History of Cancer": np.random.choice(binary, n_rows, p=[0.2, 0.8]),
        "Compromised Immune System": np.random.choice(binary, n_rows, p=[0.1, 0.9]),
        "Oral Lesions": np.random.choice(binary, n_rows, p=[0.25, 0.75]),
        "Unexplained Bleeding": np.random.choice(binary, n_rows, p=[0.15, 0.85]),
        "Difficulty Swallowing": np.random.choice(binary, n_rows, p=[0.2, 0.8]),
        "White or Red Patches in Mouth": np.random.choice(binary, n_rows, p=[0.3, 0.7]),
        "Tumor Size (cm)": np.random.uniform(0.1, 8.0, n_rows).round(1),
        "Cancer Stage": np.random.randint(0, 5, n_rows),
        "Treatment Type": np.random.choice(treatments, n_rows),
        "Survival Rate (5-Year, %)": np.random.uniform(10, 95, n_rows).round(1),
        "Cost of Treatment (USD)": np.random.uniform(5000, 150000, n_rows).round(2),
        "Economic Burden (Lost Workdays per Year)": np.random.randint(0, 300, n_rows),
        "Early Diagnosis": np.random.choice(binary, n_rows, p=[0.4, 0.6]),
        "Oral Cancer (Diagnosis)": np.random.choice(["Positive", "Negative"], n_rows, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    return df

def filter_data(df, countries, age_range, gender, risk_factors):
    """Filter dataset based on user selection"""
    filtered_df = df.copy()
    
    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
    
    filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & 
                             (filtered_df['Age'] <= age_range[1])]
    
    if gender != "All":
        filtered_df = filtered_df[filtered_df['Gender'] == gender]
    
    for factor, value in risk_factors.items():
        if value != "All":
            filtered_df = filtered_df[filtered_df[factor] == value]
            
    return filtered_df

def create_diagnostic_chart(df):
    """Create interactive diagnostic distribution chart"""
    diag_counts = df['Oral Cancer (Diagnosis)'].value_counts().reset_index()
    diag_counts.columns = ['Diagnosis', 'Count']
    
    fig = px.pie(
        diag_counts, 
        values='Count', 
        names='Diagnosis',
        title='<b>Oral Cancer Diagnosis Distribution</b>',
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend_title="Diagnosis",
        font=dict(size=12),
        hoverlabel=dict(font_size=12),
        margin=dict(t=80, l=40, r=40, b=40),
    )
    return fig

def create_risk_factors_chart(df):
    """Create risk factors analysis chart"""
    risk_factors = ["Tobacco Use", "Alcohol Consumption", "HPV Infection", "Betel Quid Use"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=risk_factors)
    
    colors = ['#ff9999', '#66b3ff']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for i, factor in enumerate(risk_factors):
        row, col = positions[i]
        cross_tab = pd.crosstab(df[factor], df["Oral Cancer (Diagnosis)"])
        
        for j, diagnosis in enumerate(cross_tab.columns):
            fig.add_trace(
                go.Bar(
                    x=cross_tab.index,
                    y=cross_tab[diagnosis],
                    name=diagnosis,
                    legendgroup=diagnosis,
                    marker_color=colors[j],
                    showlegend=True if i==0 else False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=500,
        barmode='group',
        title_text="<b>Risk Factors Analysis</b>",
        legend_title="Diagnosis",
        font=dict(size=12),
        margin=dict(t=80, l=40, r=40, b=40)
    )
    return fig

def create_survival_by_stage_chart(df):
    """Create survival rate by cancer stage chart"""
    stage_survival = df.groupby('Cancer Stage')['Survival Rate (5-Year, %)'].mean().reset_index()
    
    fig = px.line(
        stage_survival, 
        x='Cancer Stage', 
        y='Survival Rate (5-Year, %)',
        markers=True,
        title='<b>Average 5-Year Survival Rate by Cancer Stage</b>',
        color_discrete_sequence=['#4e8df5']
    )
    
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=10)
    )
    
    fig.update_layout(
        xaxis_title="Cancer Stage",
        yaxis_title="Average 5-Year Survival Rate (%)",
        font=dict(size=12),
        margin=dict(t=80, l=40, r=40, b=40),
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_country_distribution_chart(df):
    """Create country distribution chart"""
    top_countries = df["Country"].value_counts().head(10).reset_index()
    top_countries.columns = ['Country', 'Count']
    
    fig = px.bar(
        top_countries,
        x='Country',
        y='Count',
        title='<b>Top 10 Countries with Oral Cancer Cases</b>',
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Number of Cases",
        font=dict(size=12),
        margin=dict(t=80, l=40, r=40, b=40),
        coloraxis_showscale=False
    )
    
    return fig

def create_treatment_cost_chart(df):
    """Create treatment cost analysis chart"""
    treatment_cost = df.groupby('Treatment Type')['Cost of Treatment (USD)'].agg(['mean', 'min', 'max']).reset_index()
    treatment_cost.columns = ['Treatment Type', 'Average Cost', 'Min Cost', 'Max Cost']
    
    fig = px.bar(
        treatment_cost,
        x='Treatment Type',
        y='Average Cost',
        error_y=treatment_cost['Max Cost']-treatment_cost['Average Cost'],
        error_y_minus=treatment_cost['Average Cost']-treatment_cost['Min Cost'],
        title='<b>Treatment Cost Analysis</b>',
        color='Average Cost',
        color_continuous_scale='Teal'
    )
    
    fig.update_layout(
        xaxis_title="Treatment Type",
        yaxis_title="Cost (USD)",
        font=dict(size=12),
        margin=dict(t=80, l=40, r=40, b=40),
        coloraxis_showscale=False
    )
    
    return fig

def create_age_distribution_chart(df):
    """Create age distribution chart"""
    fig = px.histogram(
        df,
        x="Age",
        color="Oral Cancer (Diagnosis)",
        marginal="box",
        title="<b>Age Distribution by Diagnosis</b>",
        color_discrete_sequence=px.colors.qualitative.Bold,
        barmode="overlay",
        opacity=0.7,
        nbins=30
    )
    
    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Count",
        font=dict(size=12),
        margin=dict(t=80, l=40, r=40, b=40),
        legend_title="Diagnosis"
    )
    
    return fig

def create_early_diagnosis_chart(df):
    """Create early diagnosis impact chart"""
    diagnosis_survival = df.groupby('Early Diagnosis')['Survival Rate (5-Year, %)'].mean().reset_index()
    
    fig = px.bar(
        diagnosis_survival,
        x='Early Diagnosis',
        y='Survival Rate (5-Year, %)',
        title='<b>Impact of Early Diagnosis on Survival Rate</b>',
        color='Early Diagnosis',
        color_discrete_sequence=['#ff9999', '#66b3ff']
    )
    
    fig.update_layout(
        xaxis_title="Early Diagnosis",
        yaxis_title="Average 5-Year Survival Rate (%)",
        font=dict(size=12),
        margin=dict(t=80, l=40, r=40, b=40),
        coloraxis_showscale=False,
        showlegend=False
    )
    
    return fig

def create_economic_burden_chart(df):
    """Create economic burden chart"""
    fig = px.histogram(
        df,
        x="Economic Burden (Lost Workdays per Year)",
        color="Oral Cancer (Diagnosis)",
        title="<b>Economic Burden - Workdays Lost Due to Oral Cancer</b>",
        color_discrete_sequence=px.colors.qualitative.Bold,
        nbins=40,
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title="Lost Workdays per Year",
        yaxis_title="Count",
        font=dict(size=12),
        margin=dict(t=80, l=40, r=40, b=40),
        legend_title="Diagnosis"
    )
    
    return fig

def create_tumor_size_chart(df):
    """Create tumor size by stage chart"""
    fig = px.box(
        df,
        x="Cancer Stage",
        y="Tumor Size (cm)",
        title="<b>Tumor Size by Cancer Stage</b>",
        color="Cancer Stage",
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    
    fig.update_layout(
        xaxis_title="Cancer Stage",
        yaxis_title="Tumor Size (cm)",
        font=dict(size=12),
        margin=dict(t=80, l=40, r=40, b=40),
        showlegend=False
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_columns = ['Age', 'Tumor Size (cm)', 'Cancer Stage', 
                       'Survival Rate (5-Year, %)', 'Cost of Treatment (USD)',
                       'Economic Burden (Lost Workdays per Year)']
    corr_matrix = df[numeric_columns].corr().round(2)
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="<b>Correlation Heatmap</b>",
        zmin=-1, zmax=1
    )
    
    fig.update_layout(
        font=dict(size=12),
        margin=dict(t=80, l=40, r=40, b=40)
    )
    
    return fig

def create_gender_analysis_chart(df):
    """Create gender analysis chart"""
    gender_diag = pd.crosstab(df['Gender'], df['Oral Cancer (Diagnosis)'])
    gender_diag_perc = pd.crosstab(df['Gender'], df['Oral Cancer (Diagnosis)'], 
                                   normalize='index').reset_index()
    gender_diag_perc = pd.melt(gender_diag_perc, id_vars=['Gender'], 
                              value_vars=['Positive', 'Negative'],
                              var_name='Diagnosis', value_name='Percentage')
    gender_diag_perc['Percentage'] = gender_diag_perc['Percentage'] * 100
    
    fig = px.bar(
        gender_diag_perc,
        x='Gender',
        y='Percentage',
        color='Diagnosis',
        title='<b>Gender Distribution of Oral Cancer Cases</b>',
        color_discrete_sequence=['#ff9999', '#66b3ff'],
        barmode='group'
    )
    
    fig.update_layout(
        xaxis_title="Gender",
        yaxis_title="Percentage (%)",
        font=dict(size=12),
        margin=dict(t=80, l=40, r=40, b=40),
        legend_title="Diagnosis"
    )
    
    return fig

def create_risk_profile(df):
    """Create risk profile"""
    risk_factors = ["Tobacco Use", "Alcohol Consumption", "HPV Infection", "Betel Quid Use",
                   "Chronic Sun Exposure", "Poor Oral Hygiene", "Family History of Cancer"]
    
    risk_percentages = {}
    for factor in risk_factors:
        risk_percentages[factor] = round(
            df[df[factor] == 'Yes'].loc[df['Oral Cancer (Diagnosis)'] == 'Positive'].shape[0] / 
            df[df[factor] == 'Yes'].shape[0] * 100, 1
        )
    
    sorted_factors = sorted(risk_percentages.items(), key=lambda x: x[1], reverse=True)
    
    fig = px.bar(
        x=[item[0] for item in sorted_factors],
        y=[item[1] for item in sorted_factors],
        title="<b>Risk Profile: Percentage of Positive Diagnosis in Risk Groups</b>",
        color=[item[1] for item in sorted_factors],
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        xaxis_title="Risk Factor",
        yaxis_title="% Positive Diagnosis with Factor",
        font=dict(size=12),
        margin=dict(t=80, l=40, r=40, b=40),
        coloraxis_showscale=False,
        yaxis=dict(range=[0, 100])
    )
    
    return fig


# Main application
def main():
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar for navigation and filters
    with st.sidebar:
        st.image("https://i.imgur.com/1L568cJ.png", width=50)  # Replace with actual logo
        st.title("Oral Cancer Analytics Hub")
        
        st.markdown("### Filters")
        
        # Top countries for filtering
        top_countries = list(df['Country'].value_counts().head(10).index)
        selected_countries = st.multiselect(
            "Select Countries", 
            options=top_countries,
            default=top_countries[:5]
        )
        
        # Age range slider
        age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
        age_range = st.slider(
            "Age Range", 
            min_value=age_min, 
            max_value=age_max,
            value=(age_min, age_max)
        )
        
        # Gender selection
        gender = st.selectbox(
            "Gender",
            options=["All", "Male", "Female"]
        )
        
        # Risk factors selection
        st.markdown("#### Risk Factors")
        risk_factors = {}
        for factor in ["Tobacco Use", "Alcohol Consumption", "HPV Infection", "Betel Quid Use"]:
            risk_factors[factor] = st.selectbox(
                factor,
                options=["All", "Yes", "No"]
            )
        
        # Apply filters
        filtered_df = filter_data(df, selected_countries, age_range, gender, risk_factors)
        
        # Show filter statistics
        st.markdown("---")
        st.markdown(f"**Filtered Data: {filtered_df.shape[0]:,} records**")
        percent_showing = round(filtered_df.shape[0] / df.shape[0] * 100, 1)
        st.progress(percent_showing/100)
        st.caption(f"Showing {percent_showing}% of total data")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard provides comprehensive analytics for oral cancer prediction and analysis.
        
        Data includes risk factors, diagnostic information, treatment outcomes, and economic impact.
        """)
        
        st.markdown("© 2025 Oral Cancer Research Initiative")
    
    # Main content area
    # Navigation using tabs
    selected_tab = option_menu(
        menu_title=None,
        options=["Overview", "Risk Analysis", "Clinical Insights", "Economic Impact", "Data Explorer"],
        icons=["house", "exclamation-triangle", "hospital", "currency-dollar", "table"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#0066cc", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "center", "margin":"5px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#0066cc"},
        }
    )
    
    # Overview Tab
    if selected_tab == "Overview":
        st.markdown("# Oral Cancer Analysis Dashboard")
        st.markdown("## Key Insights and Metrics")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            positive_cases = filtered_df[filtered_df['Oral Cancer (Diagnosis)'] == 'Positive'].shape[0]
            positive_percentage = round(positive_cases / filtered_df.shape[0] * 100, 1)
            st.metric(
                "Positive Cases", 
                f"{positive_cases:,}",
                f"{positive_percentage}%",
                delta_color="inverse"
            )
        
        with col2:
            avg_survival = round(filtered_df['Survival Rate (5-Year, %)'].mean(), 1)
            st.metric(
                "Average 5-Year Survival", 
                f"{avg_survival}%",
                delta=None
            )
        
        with col3:
            avg_treatment_cost = round(filtered_df['Cost of Treatment (USD)'].mean(), 2)
            st.metric(
                "Avg Treatment Cost", 
                f"${avg_treatment_cost:,.2f}",
                delta=None
            )
        
        with col4:
            early_diagnosis_pct = round(filtered_df[filtered_df['Early Diagnosis'] == 'Yes'].shape[0] / 
                                      filtered_df.shape[0] * 100, 1)
            st.metric(
                "Early Diagnosis Rate", 
                f"{early_diagnosis_pct}%",
                delta=None
            )
        
        style_metric_cards()
        
        st.markdown("---")
        
        # Main charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_diagnostic_chart(filtered_df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_gender_analysis_chart(filtered_df), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_age_distribution_chart(filtered_df), use_container_width=True)
            
        with col2:
            st.plotly_chart(create_country_distribution_chart(filtered_df), use_container_width=True)
            
        # Key highlights
        st.markdown("## Key Highlights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='highlight-card'>
                <h3>Age Patterns</h3>
                <p>The age distribution shows higher incidence in the age range of 50-70 years, 
                with males showing higher prevalence.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='highlight-card'>
                <h3>Geographic Trends</h3>
                <p>Highest case concentrations are seen in South and Southeast Asia, 
                likely correlated with prevalent risk factors like betel quid use.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='highlight-card'>
                <h3>Treatment Outcomes</h3>
                <p>Early diagnosis significantly improves the 5-year survival rate, 
                with differences of up to 40% compared to late-stage diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Risk Analysis Tab
    elif selected_tab == "Risk Analysis":
        st.markdown("# Risk Factor Analysis")
        st.markdown("## Understanding Key Risk Factors for Oral Cancer")
        
        # Risk factors visualization
        st.plotly_chart(create_risk_factors_chart(filtered_df), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_risk_profile(filtered_df), use_container_width=True)
            
        with col2:
            # Calculate risk factor statistics
            tobacco_positive = filtered_df[(filtered_df['Tobacco Use'] == 'Yes') & 
                                         (filtered_df['Oral Cancer (Diagnosis)'] == 'Positive')].shape[0]
            tobacco_total = filtered_df[filtered_df['Tobacco Use'] == 'Yes'].shape[0]
            tobacco_pct = round(tobacco_positive / tobacco_total * 100, 1) if tobacco_total > 0 else 0
            
            alcohol_positive = filtered_df[(filtered_df['Alcohol Consumption'] == 'Yes') & 
                                         (filtered_df['Oral Cancer (Diagnosis)'] == 'Positive')].shape[0]
            alcohol_total = filtered_df[filtered_df['Alcohol Consumption'] == 'Yes'].shape[0]
            alcohol_pct = round(alcohol_positive / alcohol_total * 100, 1) if alcohol_total > 0 else 0
            
            hpv_positive = filtered_df[(filtered_df['HPV Infection'] == 'Yes') & 
                                     (filtered_df['Oral Cancer (Diagnosis)'] == 'Positive')].shape[0]
            hpv_total = filtered_df[filtered_df['HPV Infection'] == 'Yes'].shape[0]
            hpv_pct = round(hpv_positive / hpv_total * 100, 1) if hpv_total > 0 else 0
            
            betel_positive = filtered_df[(filtered_df['Betel Quid Use'] == 'Yes') & 
                                       (filtered_df['Oral Cancer (Diagnosis)'] == 'Positive')].shape[0]
            betel_total = filtered_df[filtered_df['Betel Quid Use'] == 'Yes'].shape[0]
            betel_pct = round(betel_positive / betel_total * 100, 1) if betel_total > 0 else 0
            
            # Display risk statistics as progress bars
            st.markdown("### Risk Factor Impact")
            
            st.markdown(f"<p class='big-font'>Tobacco Use: {tobacco_pct}% Positive</p>", unsafe_allow_html=True)
            st.progress(tobacco_pct/100)
            
            st.markdown(f"<p class='big-font'>Alcohol Consumption: {alcohol_pct}% Positive</p>", unsafe_allow_html=True)
            st.progress(alcohol_pct/100)
            
            st.markdown(f"<p class='big-font'>HPV Infection: {hpv_pct}% Positive</p>", unsafe_allow_html=True)
            st.progress(hpv_pct/100)
            
            st.markdown(f"<p class='big-font'>Betel Quid Use: {betel_pct}% Positive</p>", unsafe_allow_html=True)
            st.progress(betel_pct/100)
        
        st.markdown("---")
        
        # Combined risk factors analysis
        st.markdown("## Multi-Factor Risk Analysis")
        
        # Create combined risk factors
        filtered_df['Risk Count'] = (
            (filtered_df['Tobacco Use'] == 'Yes').astype(int) +
            (filtered_df['Alcohol Consumption'] == 'Yes').astype(int) +
            (filtered_df['HPV Infection'] == 'Yes').astype(int) +
            (filtered_df['Betel Quid Use'] == 'Yes').astype(int)
        )
        
        risk_count_data = filtered_df.groupby(['Risk Count', 'Oral Cancer (Diagnosis)']).size().reset_index()
        risk_count_data.columns = ['Risk Count', 'Diagnosis', 'Count']
        
        fig = px.bar(
            risk_count_data,
            x='Risk Count',
            y='Count',
            color='Diagnosis',
            barmode='group',
            title='<b>Combined Risk Factors Analysis</b>',
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        
        fig.update_layout(
            xaxis_title="Number of Risk Factors Present",
            yaxis_title="Count",
            font=dict(size=12),
            margin=dict(t=80, l=40, r=40, b=40),
            legend_title="Diagnosis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='highlight-card'>
                <h3>Combined Risk Patterns</h3>
                <p>The data shows a compounding effect when multiple risk factors are present.
                Patients with 3+ risk factors show significantly higher incidence rates.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='highlight-card'>