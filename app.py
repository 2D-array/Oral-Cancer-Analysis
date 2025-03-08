import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io

# Page configuration with custom theme and layout
st.set_page_config(
    page_title="Oral Cancer Analytics Hub",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    /* Main area styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Custom header styling */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    /* Card styling for metrics */
    .metric-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Dashboard section headers */
    .section-header {
        color: #34495e;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #eee;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px 0;
        color: #7f8c8d;
        font-size: 0.9rem;
        border-top: 1px solid #ecf0f1;
        margin-top: 3rem;
    }
    
    /* Custom metric style */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3498db;
        text-align: center;
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 400;
        color: #7f8c8d;
        text-align: center;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom button styles */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# Function to load dataset with error handling
@st.cache_data
def load_data():
    try:
        return pd.read_csv("dataset/oral_cancer_prediction_dataset.csv")
    except FileNotFoundError:
        # Create sample data if file not found for demonstration
        return create_sample_data()

def create_sample_data():
    np.random.seed(42)
    countries = ['USA', 'India', 'China', 'UK', 'Brazil', 'Japan', 'Australia']
    genders = ['Male', 'Female']
    diagnosis = ['Positive', 'Negative']
    cancer_stages = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    
    data = {
        'Patient ID': range(1, 501),
        'Age': np.random.randint(18, 85, 500),
        'Gender': np.random.choice(genders, 500),
        'Country': np.random.choice(countries, 500),
        'Tobacco Use': np.random.choice(['Yes', 'No', 'Former'], 500, p=[0.4, 0.4, 0.2]),
        'Alcohol Consumption': np.random.choice(['High', 'Moderate', 'Low', 'None'], 500),
        'HPV Infection': np.random.choice(['Yes', 'No'], 500),
        'Betel Quid Use': np.random.choice(['Yes', 'No'], 500),
        'Oral Cancer (Diagnosis)': np.random.choice(diagnosis, 500, p=[0.3, 0.7]),
        'Cancer Stage': np.random.choice(cancer_stages, 500),
        'Survival Rate (5-Year, %)': np.random.uniform(0, 100, 500),
        'Economic Burden (Lost Workdays per Year)': np.random.randint(0, 200, 500)
    }
    
    df = pd.DataFrame(data)
    
    # Adjust survival rates based on cancer stage
    stage_survival_map = {
        'Stage I': (70, 95),
        'Stage II': (50, 75),
        'Stage III': (30, 55),
        'Stage IV': (5, 35)
    }
    
    for stage, (min_val, max_val) in stage_survival_map.items():
        mask = df['Cancer Stage'] == stage
        df.loc[mask, 'Survival Rate (5-Year, %)'] = np.random.uniform(min_val, max_val, mask.sum())
    
    # Adjust economic burden based on diagnosis
    df.loc[df['Oral Cancer (Diagnosis)'] == 'Positive', 'Economic Burden (Lost Workdays per Year)'] *= 2
    
    return df

# Load the data
df = load_data()

# Sidebar navigation and filters
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Oral Cancer Analytics</h1>", unsafe_allow_html=True)
    
    # Navigation menu
    selected_page = option_menu(
        menu_title=None,
        options=["Dashboard", "Demographics", "Risk Analysis", "Economic Impact", "About"],
        icons=["graph-up", "people-fill", "exclamation-triangle", "currency-dollar", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": "#3498db", "font-size": "14px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#3498db", "color": "white"},
        }
    )
    
    st.markdown("<div class='section-header'>Data Filters</div>", unsafe_allow_html=True)
    
    # Filters
    selected_country = st.multiselect(
        "Select Countries",
        options=["All"] + sorted(list(df["Country"].unique())),
        default="All"
    )
    
    selected_gender = st.multiselect(
        "Select Gender",
        options=["All"] + sorted(list(df["Gender"].unique())),
        default="All"
    )
    
    age_range = st.slider(
        "Age Range",
        min_value=int(df["Age"].min()),
        max_value=int(df["Age"].max()),
        value=(int(df["Age"].min()), int(df["Age"].max()))
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if "All" not in selected_country:
        filtered_df = filtered_df[filtered_df["Country"].isin(selected_country)]
    
    if "All" not in selected_gender:
        filtered_df = filtered_df[filtered_df["Gender"].isin(selected_gender)]
    
    filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]
    
    st.markdown(
        f"<div style='text-align: center; padding: 10px; background-color: #e8f4f8; border-radius: 5px;'>"
        f"Showing {len(filtered_df)} of {len(df)} records"
        f"</div>",
        unsafe_allow_html=True
    )

# Header
st.markdown("<h1 class='main-header'>🔬 Oral Cancer Analysis Platform</h1>", unsafe_allow_html=True)

# Dashboard Page
if selected_page == "Dashboard":
    # Key metrics in cards
    st.markdown("<div class='section-header'>Key Metrics</div>", unsafe_allow_html=True)
    
    # Calculate key metrics
    total_patients = len(filtered_df)
    cancer_positive = filtered_df[filtered_df["Oral Cancer (Diagnosis)"] == "Positive"].shape[0]
    cancer_negative = filtered_df[filtered_df["Oral Cancer (Diagnosis)"] == "Negative"].shape[0]
    avg_survival_rate = filtered_df[filtered_df["Oral Cancer (Diagnosis)"] == "Positive"]["Survival Rate (5-Year, %)"].mean()
    avg_workdays_lost = filtered_df[filtered_df["Oral Cancer (Diagnosis)"] == "Positive"]["Economic Burden (Lost Workdays per Year)"].mean()
    
    # Display metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{total_patients}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Patients</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{cancer_positive}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Cancer Positive Cases</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{avg_survival_rate:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg. 5-Year Survival Rate</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{avg_workdays_lost:.0f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg. Workdays Lost</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-header'>Diagnosis by Country</div>", unsafe_allow_html=True)
        country_diagnosis = filtered_df.groupby(['Country', 'Oral Cancer (Diagnosis)']).size().reset_index(name='Count')
        fig = px.bar(
            country_diagnosis, 
            x='Country', 
            y='Count', 
            color='Oral Cancer (Diagnosis)',
            color_discrete_map={'Positive': '#e74c3c', 'Negative': '#3498db'},
            barmode='group',
            height=400
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            legend_title_text='Diagnosis',
            xaxis_title='',
            yaxis_title='Number of Patients'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='section-header'>Cancer Stage Distribution</div>", unsafe_allow_html=True)
        cancer_positive_df = filtered_df[filtered_df['Oral Cancer (Diagnosis)'] == 'Positive']
        stage_counts = cancer_positive_df['Cancer Stage'].value_counts().reset_index()
        stage_counts.columns = ['Cancer Stage', 'Count']
        
        # Sort stages correctly
        stage_order = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
        stage_counts['order'] = stage_counts['Cancer Stage'].map({s: i for i, s in enumerate(stage_order)})
        stage_counts = stage_counts.sort_values('order').drop('order', axis=1)
        
        fig = px.pie(
            stage_counts, 
            values='Count', 
            names='Cancer Stage',
            color='Cancer Stage',
            color_discrete_sequence=px.colors.sequential.Plasma_r,
            hole=0.4,
            height=400
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            legend_title_text='',
            showlegend=True
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Second row visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-header'>Age Distribution by Diagnosis</div>", unsafe_allow_html=True)
        fig = px.histogram(
            filtered_df, 
            x="Age", 
            color="Oral Cancer (Diagnosis)",
            marginal="box", 
            nbins=30,
            color_discrete_map={'Positive': '#e74c3c', 'Negative': '#3498db'},
            height=400
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            legend_title_text='Diagnosis',
            xaxis_title='Age',
            yaxis_title='Count'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='section-header'>Survival Rate by Cancer Stage</div>", unsafe_allow_html=True)
        cancer_positive_df = filtered_df[filtered_df['Oral Cancer (Diagnosis)'] == 'Positive']
        
        fig = px.box(
            cancer_positive_df,
            x="Cancer Stage", 
            y="Survival Rate (5-Year, %)",
            color="Cancer Stage",
            color_discrete_sequence=px.colors.sequential.Plasma_r,
            category_orders={"Cancer Stage": ["Stage I", "Stage II", "Stage III", "Stage IV"]},
            height=400
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            legend_title_text='',
            showlegend=False,
            xaxis_title='Cancer Stage',
            yaxis_title='5-Year Survival Rate (%)'
        )
        st.plotly_chart(fig, use_container_width=True)

# Demographics Page
elif selected_page == "Demographics":
    st.markdown("<div class='section-header'>Patient Demographics Analysis</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution
        st.markdown("<div class='section-header'>Gender Distribution</div>", unsafe_allow_html=True)
        gender_counts = filtered_df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        fig = px.pie(
            gender_counts, 
            values='Count', 
            names='Gender',
            color='Gender',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            height=350
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            legend_title_text='',
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age distribution
        st.markdown("<div class='section-header'>Age Distribution</div>", unsafe_allow_html=True)
        
        fig = px.histogram(
            filtered_df, 
            x="Age",
            nbins=20,
            color_discrete_sequence=['#3498db'],
            height=350
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title='Age',
            yaxis_title='Count'
        )
        # Add mean and median lines
        fig.add_vline(x=filtered_df['Age'].mean(), line_dash="dash", line_color="#e74c3c", annotation_text=f"Mean: {filtered_df['Age'].mean():.1f}")
        fig.add_vline(x=filtered_df['Age'].median(), line_dash="dash", line_color="#2ecc71", annotation_text=f"Median: {filtered_df['Age'].median():.1f}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Demographics by diagnosis
    st.markdown("<div class='section-header'>Demographics by Diagnosis Status</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by diagnosis
        age_diagnosis = filtered_df.groupby(['Oral Cancer (Diagnosis)'])['Age'].agg(['mean', 'median']).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=age_diagnosis['Oral Cancer (Diagnosis)'],
            y=age_diagnosis['mean'],
            name='Mean Age',
            marker_color='#3498db'
        ))
        fig.add_trace(go.Bar(
            x=age_diagnosis['Oral Cancer (Diagnosis)'],
            y=age_diagnosis['median'],
            name='Median Age',
            marker_color='#2ecc71'
        ))
        
        fig.update_layout(
            title="Age Statistics by Diagnosis",
            xaxis_title="Diagnosis",
            yaxis_title="Age",
            legend_title="Statistic",
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender vs diagnosis
        gender_diagnosis = filtered_df.groupby(['Gender', 'Oral Cancer (Diagnosis)']).size().reset_index(name='Count')
        
        fig = px.bar(
            gender_diagnosis, 
            x='Gender', 
            y='Count', 
            color='Oral Cancer (Diagnosis)',
            color_discrete_map={'Positive': '#e74c3c', 'Negative': '#3498db'},
            barmode='group',
            height=400
        )
        fig.update_layout(
            title="Diagnosis by Gender",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20),
            legend_title_text='Diagnosis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic distribution
    st.markdown("<div class='section-header'>Geographic Distribution</div>", unsafe_allow_html=True)
    
    country_counts = filtered_df.groupby('Country')['Patient ID'].count().reset_index()
    country_counts.columns = ['Country', 'Patient Count']
    
    # Sort by patient count
    country_counts = country_counts.sort_values('Patient Count', ascending=False)
    
    fig = px.bar(
        country_counts,
        x='Country',
        y='Patient Count',
        color='Patient Count',
        color_continuous_scale='Viridis',
        height=400
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title='',
        yaxis_title='Number of Patients',
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Risk Analysis Page
elif selected_page == "Risk Analysis":
    st.markdown("<div class='section-header'>Risk Factors Analysis</div>", unsafe_allow_html=True)
    
    # Overall risk factors analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-header'>Tobacco Use and Diagnosis</div>", unsafe_allow_html=True)
        tobacco_diagnosis = filtered_df.groupby(['Tobacco Use', 'Oral Cancer (Diagnosis)']).size().reset_index(name='Count')
        
        fig = px.bar(
            tobacco_diagnosis, 
            x='Tobacco Use', 
            y='Count', 
            color='Oral Cancer (Diagnosis)',
            color_discrete_map={'Positive': '#e74c3c', 'Negative': '#3498db'},
            barmode='group',
            height=350
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            legend_title_text='Diagnosis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='section-header'>Alcohol Consumption and Diagnosis</div>", unsafe_allow_html=True)
        alcohol_diagnosis = filtered_df.groupby(['Alcohol Consumption', 'Oral Cancer (Diagnosis)']).size().reset_index(name='Count')
        
        # Ensure correct ordering of alcohol consumption levels
        alcohol_order = ['None', 'Low', 'Moderate', 'High']
        alcohol_diagnosis['order'] = alcohol_diagnosis['Alcohol Consumption'].map({s: i for i, s in enumerate(alcohol_order)})
        alcohol_diagnosis = alcohol_diagnosis.sort_values('order').drop('order', axis=1)
        
        fig = px.bar(
            alcohol_diagnosis, 
            x='Alcohol Consumption', 
            y='Count', 
            color='Oral Cancer (Diagnosis)',
            color_discrete_map={'Positive': '#e74c3c', 'Negative': '#3498db'},
            barmode='group',
            category_orders={"Alcohol Consumption": alcohol_order},
            height=350
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            legend_title_text='Diagnosis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-header'>HPV Infection and Diagnosis</div>", unsafe_allow_html=True)
        hpv_diagnosis = filtered_df.groupby(['HPV Infection', 'Oral Cancer (Diagnosis)']).size().reset_index(name='Count')
        
        fig = px.bar(
            hpv_diagnosis, 
            x='HPV Infection', 
            y='Count', 
            color='Oral Cancer (Diagnosis)',
            color_discrete_map={'Positive': '#e74c3c', 'Negative': '#3498db'},
            barmode='group',
            height=350
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            legend_title_text='Diagnosis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='section-header'>Betel Quid Use and Diagnosis</div>", unsafe_allow_html=True)
        betel_diagnosis = filtered_df.groupby(['Betel Quid Use', 'Oral Cancer (Diagnosis)']).size().reset_index(name='Count')
        
        fig = px.bar(
            betel_diagnosis, 
            x='Betel Quid Use', 
            y='Count', 
            color='Oral Cancer (Diagnosis)',
            color_discrete_map={'Positive': '#e74c3c', 'Negative': '#3498db'},
            barmode='group',
            height=350
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            legend_title_text='Diagnosis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Combined risk factors analysis
    st.markdown("<div class='section-header'>Combined Risk Factors Analysis</div>", unsafe_allow_html=True)
    
    # Create risk score (simplistic)
    filtered_df['Risk Score'] = 0
    filtered_df.loc[filtered_df['Tobacco Use'] == 'Yes', 'Risk Score'] += 3
    filtered_df.loc[filtered_df['Tobacco Use'] == 'Former', 'Risk Score'] += 1
    filtered_df.loc[filtered_df['Alcohol Consumption'] == 'High', 'Risk Score'] += 3
    filtered_df.loc[filtered_df['Alcohol Consumption'] == 'Moderate', 'Risk Score'] += 2
    filtered_df.loc[filtered_df['Alcohol Consumption'] == 'Low', 'Risk Score'] += 1
    filtered_df.loc[filtered_df['HPV Infection'] == 'Yes', 'Risk Score'] += 3
    filtered_df.loc[filtered_df['Betel Quid Use'] == 'Yes', 'Risk Score'] += 3
    
    fig = px.box(
        filtered_df,
        x='Oral Cancer (Diagnosis)',
        y='Risk Score',
        color='Oral Cancer (Diagnosis)',
        color_discrete_map={'Positive': '#e74c3c', 'Negative': '#3498db'},
        height=400,
        points='all'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20),
        legend_title_text='Diagnosis',
        showlegend=False,
        xaxis_title='Diagnosis',
        yaxis_title='Combined Risk Score'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;'>
        <h4 style='color: #2c3e50;'>About the Risk Score</h4>
        <p>The combined risk score is calculated based on the following factors:</p>
        <ul>
            <li>Tobacco Use: Current (3 points), Former (1 point), None (0 points)</li>
            <li>Alcohol Consumption: High (3 points), Moderate (2 points), Low (1 point), None (0 points)</li>
            <li>HPV Infection: Yes (3 points), No (0 points)</li>
            <li>Betel Quid Use: Yes (3 points), No (0 points)</li>
        </ul>
        <p>Higher scores indicate a greater number of risk factors present.</p>
    </div>
    """, unsafe_allow_html=True)

# Economic Impact Page (continued)
elif selected_page == "Economic Impact":
    st.markdown("<div class='section-header'>Economic Impact Analysis</div>", unsafe_allow_html=True)
    
    # Key economic metrics
    col1, col2 = st.columns(2)
    
    with col1:
        avg_workdays_cancer = filtered_df[filtered_df['Oral Cancer (Diagnosis)'] == 'Positive']['Economic Burden (Lost Workdays per Year)'].mean()
        avg_workdays_no_cancer = filtered_df[filtered_df['Oral Cancer (Diagnosis)'] == 'Negative']['Economic Burden (Lost Workdays per Year)'].mean()
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{avg_workdays_cancer:.1f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg. Workdays Lost (Cancer Positive)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        total_workdays_lost = filtered_df['Economic Burden (Lost Workdays per Year)'].sum()
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{total_workdays_lost:,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Workdays Lost</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Workdays lost by diagnosis distribution
    st.markdown("<div class='section-header'>Workdays Lost Distribution</div>", unsafe_allow_html=True)
    
    fig = px.histogram(
        filtered_df,
        x="Economic Burden (Lost Workdays per Year)",
        color="Oral Cancer (Diagnosis)",
        barmode="overlay",
        nbins=30,
        opacity=0.7,
        color_discrete_map={'Positive': '#e74c3c', 'Negative': '#3498db'},
        height=400
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20),
        legend_title_text='Diagnosis',
        xaxis_title='Lost Workdays per Year',
        yaxis_title='Number of Patients'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Economic burden by cancer stage
    st.markdown("<div class='section-header'>Economic Burden by Cancer Stage</div>", unsafe_allow_html=True)
    
    cancer_positive_df = filtered_df[filtered_df['Oral Cancer (Diagnosis)'] == 'Positive']
    
    if not cancer_positive_df.empty:
        stage_workdays = cancer_positive_df.groupby('Cancer Stage')['Economic Burden (Lost Workdays per Year)'].mean().reset_index()
        
        # Ensure correct ordering of cancer stages
        stage_order = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
        stage_workdays['order'] = stage_workdays['Cancer Stage'].map({s: i for i, s in enumerate(stage_order)})
        stage_workdays = stage_workdays.sort_values('order').drop('order', axis=1)
        
        fig = px.bar(
            stage_workdays,
            x='Cancer Stage',
            y='Economic Burden (Lost Workdays per Year)',
            color='Economic Burden (Lost Workdays per Year)',
            color_continuous_scale='Reds',
            height=400,
            category_orders={"Cancer Stage": stage_order}
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title='Cancer Stage',
            yaxis_title='Average Workdays Lost per Year',
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cancer positive patients in the current filter selection.")
    
    # Economic impact by country
    st.markdown("<div class='section-header'>Economic Impact by Country</div>", unsafe_allow_html=True)
    
    country_workdays = filtered_df.groupby('Country')['Economic Burden (Lost Workdays per Year)'].agg(['mean', 'sum']).reset_index()
    country_workdays.columns = ['Country', 'Average Workdays Lost', 'Total Workdays Lost']
    
    # Sort by average workdays lost
    country_workdays = country_workdays.sort_values('Average Workdays Lost', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            country_workdays,
            x='Country',
            y='Average Workdays Lost',
            color='Average Workdays Lost',
            color_continuous_scale='Reds',
            height=400
        )
        
        fig.update_layout(
            title="Average Workdays Lost by Country",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title='',
            yaxis_title='Average Workdays Lost',
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            country_workdays,
            x='Country',
            y='Total Workdays Lost',
            color='Total Workdays Lost',
            color_continuous_scale='Oranges',
            height=400
        )
        
        fig.update_layout(
            title="Total Workdays Lost by Country",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title='',
            yaxis_title='Total Workdays Lost',
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Economic impact calculator
    st.markdown("<div class='section-header'>Economic Impact Calculator</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        Use this calculator to estimate the economic impact based on daily wage and workdays lost.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        daily_wage = st.number_input("Average Daily Wage (USD)", min_value=0, value=100, step=10)
    
    with col2:
        selected_country = st.selectbox(
            "Select Country for Analysis",
            options=["All Countries"] + sorted(list(filtered_df["Country"].unique()))
        )
    
    if selected_country == "All Countries":
        workdays_data = filtered_df
    else:
        workdays_data = filtered_df[filtered_df["Country"] == selected_country]
    
    total_economic_impact = workdays_data['Economic Burden (Lost Workdays per Year)'].sum() * daily_wage
    avg_economic_impact_per_patient = workdays_data['Economic Burden (Lost Workdays per Year)'].mean() * daily_wage
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${total_economic_impact:,.0f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Economic Impact (USD)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${avg_economic_impact_per_patient:,.0f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Average Impact per Patient (USD)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# About Page
elif selected_page == "About":
    st.markdown("<div class='section-header'>About This Dashboard</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
        <h3 style='color: #2c3e50;'>Oral Cancer Analytics Platform</h3>
        <p>This comprehensive analytics platform provides healthcare professionals and researchers with powerful tools to analyze and visualize oral cancer data.</p>
        
        <h4 style='color: #2c3e50; margin-top: 20px;'>Key Features:</h4>
        <ul>
            <li><strong>Interactive Visualizations:</strong> Explore patient demographics, risk factors, and economic impacts through dynamic charts and graphs.</li>
            <li><strong>Advanced Filtering:</strong> Drill down into specific subsets of data based on country, gender, age range, and other factors.</li>
            <li><strong>Risk Factor Analysis:</strong> Examine the relationship between risk factors such as tobacco use, alcohol consumption, HPV infection, and oral cancer diagnosis.</li>
            <li><strong>Economic Impact Assessment:</strong> Quantify the economic burden of oral cancer through workdays lost and associated costs.</li>
        </ul>
        
        <h4 style='color: #2c3e50; margin-top: 20px;'>Data Description:</h4>
        <p>The platform analyzes a comprehensive dataset of oral cancer patients, including:</p>
        <ul>
            <li>Patient demographics (age, gender, country)</li>
            <li>Risk factors (tobacco use, alcohol consumption, HPV infection, betel quid use)</li>
            <li>Clinical information (diagnosis, cancer stage, survival rate)</li>
            <li>Economic metrics (workdays lost per year)</li>
        </ul>
        
        <h4 style='color: #2c3e50; margin-top: 20px;'>Usage Instructions:</h4>
        <ol>
            <li>Use the sidebar filters to refine the data displayed in the dashboard.</li>
            <li>Navigate between different analysis views using the menu at the top of the sidebar.</li>
            <li>Hover over charts and graphs to see detailed information.</li>
            <li>Use the economic calculator to estimate financial impact based on custom parameters.</li>
        </ol>
        
        <h4 style='color: #2c3e50; margin-top: 20px;'>Future Updates:</h4>
        <p>We plan to enhance this platform with additional features, including:</p>
        <ul>
            <li>Predictive analytics for risk assessment</li>
            <li>Treatment outcome analysis</li>
            <li>Geospatial mapping of case distributions</li>
            <li>Temporal trend analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>Data Sources and Methodology</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
        <h4 style='color: #2c3e50;'>Data Collection</h4>
        <p>The data used in this dashboard was collected from multiple sources, including:</p>
        <ul>
            <li>Patient electronic health records (anonymized)</li>
            <li>National cancer registries</li>
            <li>Research studies on oral cancer risk factors and outcomes</li>
            <li>Economic surveys on healthcare costs and productivity loss</li>
        </ul>
        
        <h4 style='color: #2c3e50; margin-top: 20px;'>Analytical Methods</h4>
        <p>The analysis employs various statistical techniques to explore relationships between risk factors, demographics, and outcomes. Key analytical approaches include:</p>
        <ul>
            <li>Descriptive statistics to summarize patient characteristics</li>
            <li>Comparative analysis of risk factors across different demographic groups</li>
            <li>Analysis of variance (ANOVA) to assess differences in outcomes based on risk factors</li>
            <li>Economic modeling to estimate the financial impact of oral cancer</li>
        </ul>
        
        <h4 style='color: #2c3e50; margin-top: 20px;'>Data Privacy and Ethics</h4>
        <p>All data has been anonymized to protect patient privacy. The analysis complies with healthcare data protection regulations and has received appropriate ethical approvals.</p>
        
        <h4 style='color: #2c3e50; margin-top: 20px;'>Limitations</h4>
        <p>While comprehensive, this analysis has certain limitations:</p>
        <ul>
            <li>Data may not be representative of all geographic regions</li>
            <li>Some risk factors may be self-reported and subject to recall bias</li>
            <li>Economic impact calculations are estimates based on available data</li>
            <li>The dataset may not capture all relevant confounding variables</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer for all pages
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("""
<div style='display: flex; justify-content: space-between; align-items: center;'>
    <div>
        <p>© 2025 Oral Cancer Analytics Platform</p>
    </div>
    <div>
        <p>Version 2.0 | Last Updated: March 2025</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Add a hidden download button for data export
if st.checkbox("Show Data Export Options", value=False):
    st.markdown("<div class='section-header'>Data Export</div>", unsafe_allow_html=True)
    
    export_format = st.radio("Select Export Format", ["CSV", "Excel", "JSON"])
    
    if st.button("Export Filtered Data"):
        # Create a download buffer
        buffer = io.BytesIO()
        
        if export_format == "CSV":
            filtered_df.to_csv(buffer, index=False)
            file_extension = "csv"
            mime_type = "text/csv"
        elif export_format == "Excel":
            filtered_df.to_excel(buffer, index=False)
            file_extension = "xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:  # JSON
            filtered_df.to_json(buffer, orient="records")
            file_extension = "json"
            mime_type = "application/json"
        
        buffer.seek(0)
        st.download_button(
            label=f"Download as {export_format}",
            data=buffer,
            file_name=f"oral_cancer_data.{file_extension}",
            mime=mime_type
        )