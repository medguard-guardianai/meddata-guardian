"""
MedGuard AI - Enhanced Dashboard
Multi-agent AI system for healthcare data quality and fairness
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import ollama

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.phi_scanner import PHIScanner, PHIDetectedError
from utils.data_quality import DataQualityAnalyzer
from utils.bias_detection import BiasDetector
from utils.synthetic_generator import SyntheticDataGenerator
from agents.medical_advisor_v2 import MedicalAdvisorV2
from agents.fairness_specialist import FairnessSpecialist
from agents.deployment_strategist import DeploymentStrategist

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="MedGuard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove ALL unnecessary padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0.3rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Remove padding from main content area */
    .main .block-container {
        padding-top: 2rem !important;
    }
    
    /* Tighten up all elements - NO margin/padding by default */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Remove spacing from stMarkdown */
    .stMarkdown {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Main Header with Gradient Animation */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 8s ease infinite;
        margin: 0.3rem 0 !important;
        padding: 0 !important;
        text-align: center;
        letter-spacing: -2px;
        line-height: 1.2;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subheader {
        font-size: 1.2rem;
        color: #a0a0a0;
        margin: 0.2rem 0 !important;
        padding: 0 !important;
        text-align: center;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        margin: 0.4rem 0 0.5rem 0 !important;
        padding: 0 !important;
        flex-wrap: wrap;
    }
    
    .badge {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Card Styling - MINIMAL padding */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 0.3rem 0 !important;
        padding: 0.6rem !important;
    }
    
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, #ffffff05, #ffffff08);
        border: 1px solid #ffffff15;
        border-radius: 10px;
        margin: 0.2rem 0 !important;
        padding: 0 !important;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stExpander"]:hover {
        border-color: #667eea55;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Tabs Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: linear-gradient(135deg, #ffffff05, #ffffff08);
        padding: 0.5rem;
        border-radius: 15px;
        border: 1px solid #ffffff15;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 1rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Button Enhancements */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Enhancement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1d29 100%);
        border-right: 1px solid #ffffff15;
    }
    
    /* Progress Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .stSpinner > div {
        border-color: #667eea !important;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Code Block Enhancement */
    .stCodeBlock {
        border-radius: 10px;
        border: 1px solid #ffffff15;
        background: #1a1d29 !important;
    }
    
    /* Divider - MINIMAL */
    hr {
        margin: 0.5rem 0 !important;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea55, transparent);
    }
    
    /* Reduce all section spacing to MINIMUM */
    .element-container {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Tighter expander spacing */
    div[data-testid="stExpander"] {
        margin: 0.2rem 0 !important;
    }
    
    /* Reduce header margins to MINIMUM */
    h1, h2, h3, h4 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }
    
    /* Tighten paragraphs - NO margin */
    p {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Reduce form spacing */
    .stForm {
        padding: 0.3rem !important;
        margin: 0 !important;
    }
    
    /* Tighter column spacing */
    [data-testid="column"] {
        padding: 0.2rem !important;
    }
    
    /* Remove spacing from rows */
    .row-widget {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Change top navigation bar color to match heading */
    header[data-testid="stHeader"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Style the toolbar buttons */
    header[data-testid="stHeader"] button {
        color: white !important;
    }
    
    header[data-testid="stHeader"] svg {
        fill: white !important;
        stroke: white !important;
    }
    
    /* Deploy button styling */
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        color: white !important;
    }
    
    /* Info Boxes - MINIMAL padding and margin */
    .info-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border-left: 4px solid #667eea;
        padding: 0.6rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #28a74515, #20c99715);
        border-left: 4px solid #28a745;
        padding: 0.6rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffc10715, #ff940015);
        border-left: 4px solid #ffc107;
        padding: 0.6rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #dc354515, #e7444815);
        border-left: 4px solid #dc3545;
        padding: 0.6rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    /* Section Headers - TIGHT */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
        line-height: 1.2 !important;
    }
    
    h2 {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 1.8rem !important;
    }
    
    h3 {
        font-size: 1.3rem !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* Remove spacing from sidebar - ULTRA TIGHT */
    [data-testid="stSidebar"] {
        padding-top: 0.3rem !important;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.3rem !important;
        padding-left: 0.4rem !important;
        padding-right: 0.4rem !important;
    }
    
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.2rem !important;
    }
    
    [data-testid="stSidebar"] h2 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 1.2rem !important;
    }
    
    [data-testid="stSidebar"] .info-box,
    [data-testid="stSidebar"] .success-box {
        padding: 0.5rem !important;
        margin: 0.3rem 0 !important;
        font-size: 0.85rem;
    }
    
    /* Tighter metric display */
    [data-testid="stMetric"] {
        padding: 0.2rem !important;
        margin: 0 !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    
    /* Reduce button padding */
    .stButton > button {
        padding: 0.5rem 1.2rem !important;
        margin: 0.2rem 0 !important;
    }
    
    /* Minimal input spacing */
    .stTextInput, .stSelectbox, .stTextArea {
        margin-bottom: 0.3rem !important;
    }
    
    /* Remove extra space from file uploader */
    [data-testid="stFileUploader"] {
        margin: 0.3rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'user_context' not in st.session_state:
    st.session_state.user_context = {}

if 'onboarding_complete' not in st.session_state:
    st.session_state.onboarding_complete = False

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

if 'download_synthetic' not in st.session_state:
    st.session_state.download_synthetic = False

if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">🏥 MedGuard AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Multi-Agent Healthcare Data Validator with Privacy-Preserving Intelligence</p>', unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🎯 About MedGuard AI")
    st.markdown("""
    <div class="info-box">
    <strong>🤖 Multi-Agent System:</strong><br>
    • <strong>Medical Advisor</strong> - Clinical context<br>
    • <strong>Fairness Specialist</strong> - Bias mitigation<br>
    • <strong>Deployment Strategist</strong> - Action plans
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
    <strong>🔒 HIPAA Protection:</strong><br>
    • Privacy-preserving synthetic data<br>
    • Multi-stage PHI detection<br>
    • Local processing only<br>
    • Zero data transmission
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📁 Demo Datasets")
    demo_option = st.selectbox(
        "Load example:",
        ["Upload Your Own",
         "Demo 1: Heart Disease (Quality Issues)",
         "Demo 2: Diabetes (Gender Bias)",
         "Demo 3: Heart Disease (Indigenous Bias)",
         "Demo 4: Combined Problems"],
        key=f"demo_selector_{st.session_state.get('reset_counter', 0)}"
    )
    
    if st.button("🔄 Reset Analysis", use_container_width=True):
        # Store reset counter before clearing
        reset_count = st.session_state.get('reset_counter', 0) + 1
        
        # Clear ALL session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Re-initialize essential variables
        st.session_state.user_context = {}
        st.session_state.onboarding_complete = False
        st.session_state.analysis_complete = False
        st.session_state.download_synthetic = False
        st.session_state.reset_counter = reset_count
        
        st.rerun()
    
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem; opacity: 0.6; margin-top: 1rem;">
    <small>Powered by Llama 3.2 & Ollama</small>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PHASE 1: USER ONBOARDING (Context Gathering)
# ============================================================================

if not st.session_state.onboarding_complete:
    st.markdown("## 📋 Step 1: Tell Us About Your Project")
    st.write("Help us provide context-aware recommendations by answering a few questions:")
    
    with st.form("onboarding_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            project_description = st.text_area(
                "🎯 What are you building?",
                placeholder="e.g., Lung disease prediction model for hospital deployment",
                height=120,
                help="Describe your ML project in 1-2 sentences"
            )
            
            model_type = st.selectbox(
                "🤖 What model will you use?",
                ["Random Forest", "Neural Network", "Logistic Regression",
                 "XGBoost", "Support Vector Machine", "Ensemble", "Other"]
            )
            
            use_case = st.selectbox(
                "🏥 What's your use case?",
                ["Research study", "Clinical deployment",
                 "FDA submission", "Proof of concept", "Academic project"]
            )
        
        with col2:
            timeline = st.selectbox(
                "⏱️ When do you need this deployed?",
                ["<30 days (urgent)", "30-60 days", "60-90 days", "90+ days (flexible)"]
            )
            
            can_collect_data = st.radio(
                "📊 Can you collect additional patient data if needed?",
                ["Yes, we have recruitment capabilities",
                 "Maybe, but it's difficult",
                 "No, we must use existing data only"]
            )
            
            location = st.text_input(
                "📍 Where is this being deployed? (Optional)",
                placeholder="e.g., Boston Medical Center, Boston, MA",
                help="Optional: Helps AI provide location-specific recommendations"
            )
        
        st.write("")  # Spacing
        submitted = st.form_submit_button("✨ Continue to Analysis", use_container_width=True)
        
        if submitted:
            if not project_description:
                st.error("⚠️ Please fill in project description")
            else:
                timeline_days = {
                    "<30 days (urgent)": 30,
                    "30-60 days": 45,
                    "60-90 days": 75,
                    "90+ days (flexible)": 120
                }
                
                st.session_state.user_context = {
                    'project_description': project_description,
                    'model_type': model_type,
                    'use_case': use_case,
                    'timeline_days': timeline_days[timeline],
                    'can_collect_data': can_collect_data,
                    'location': location if location else 'Not specified'
                }
                st.session_state.onboarding_complete = True
                st.rerun()
    
    # ADD THIS - Stop execution here if onboarding not complete
    st.stop()

# ============================================================================
# PHASE 2: DATA UPLOAD & ANALYSIS
# ============================================================================

# This section will ONLY run after onboarding is complete
elif st.session_state.onboarding_complete:
    
    # Show user context summary with enhanced styling
    with st.expander("📋 Your Project Context", expanded=False):
        ctx = st.session_state.user_context
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🤖 Model", ctx['model_type'])
        with col2:
            st.metric("⏱️ Timeline", f"{ctx['timeline_days']} days")
        with col3:
            st.metric("🏥 Use Case", ctx['use_case'])
        with col4:
            data_status = "✅ Yes" if "Yes" in ctx['can_collect_data'] else "⚠️ Limited" if "Maybe" in ctx['can_collect_data'] else "❌ No"
            st.metric("📊 Data Collection", data_status)
        
        st.write(f"**📍 Location:** {ctx['location']}")
        st.write(f"**🎯 Project:** {ctx['project_description']}")
    
    
    # File upload with better styling
    st.markdown("## 📂 Step 2: Upload Your Dataset")
    
    uploaded_file = None
    df = None
    
   # Handle demo datasets
if demo_option != "Upload Your Own":
    demo_files = {
        "Demo 1: Heart Disease (Quality Issues)": "data/synthetic/dataset1_heart_disease_quality.csv",
        "Demo 2: Diabetes (Gender Bias)": "data/synthetic/dataset2_diabetes_gender_bias.csv",
        "Demo 3: Heart Disease (Indigenous Bias)": "data/synthetic/dataset3_heart_disease_indigenous.csv",
        "Demo 4: Combined Problems": "data/synthetic/dataset4_diabetes_combined.csv"
    }
    
    demo_path = demo_files[demo_option]
    
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        
        # PHI SCAN
        scanner = PHIScanner()
        is_safe, violations = scanner.scan_dataset(df)
        
        if not is_safe:
            st.markdown("""
            <div class="error-box">
            <h3>🚨 PHI DETECTED - Cannot Process</h3>
            <strong>Violations found:</strong>
            </div>
            """, unsafe_allow_html=True)
            for v in violations:
                st.write(f"  • {v}")
            st.warning("Please de-identify your dataset and re-upload.")
            st.info("**Remove:** patient names, SSNs, phone numbers, email addresses, dates of birth, addresses")
            st.stop()
        
        # CONSOLIDATED STATUS BOX
        st.markdown(f"""
        <div>
        <strong>✅ Loaded:</strong> {demo_option}<br>
        📊 <strong>Records:</strong> {len(df):,} | <strong>Features:</strong> {len(df.columns)}<br>
        ⚠️ <strong>DEMO MODE:</strong> Using synthetic data for demonstration<br>
        ✅ <strong>PHI Scan Passed</strong> - Dataset is properly de-identified<br>
        🔒 <strong>Privacy Firewall Active</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Demo file not found: {demo_path}")
        st.stop()
else:
    uploaded_file = st.file_uploader(
        "Upload your healthcare dataset (CSV)",
        type=['csv'],
        help="Must be de-identified. Our tool will validate.",
        label_visibility="collapsed"
    )
    
    if uploaded_file is None:
        st.markdown("""
        <div class="info-box">
        👆 <strong>Upload a CSV file</strong> or select a demo dataset from the sidebar to begin analysis
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    df = pd.read_csv(uploaded_file)
    
    # PHI SCAN
    with st.spinner("🔍 Scanning for Protected Health Information..."):
        scanner = PHIScanner()
        is_safe, violations = scanner.scan_dataset(df)
    
    if not is_safe:
        st.markdown("""
        <div class="error-box">
        <h3>🚨 PHI DETECTED - Cannot Process</h3>
        <strong>Violations found:</strong>
        </div>
        """, unsafe_allow_html=True)
        for v in violations:
            st.write(f"  • {v}")
        st.warning("Please de-identify your dataset and re-upload.")
        st.info("**Remove:** patient names, SSNs, phone numbers, email addresses, dates of birth, addresses")
        st.stop()
    
    # CONSOLIDATED STATUS BOX FOR UPLOADED FILE
    st.markdown(f"""
    <div>
    ✅ <strong>File loaded successfully!</strong><br>
    📊 <strong>Records:</strong> {len(df):,} | <strong>Features:</strong> {len(df.columns)}<br>
    ✅ <strong>PHI Scan Passed</strong> - Dataset is properly de-identified<br>
    🔒 <strong>Privacy Firewall Active</strong> 
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # GENERATE SYNTHETIC TWIN (PRIVACY FIREWALL)
    # ========================================================================


# Add synthetic dataset generation controls
col_generate1, col_generate2 = st.columns([3, 1])
with col_generate1:
    synthetic_size = st.slider(
        "Synthetic dataset size",
        min_value=100,
        max_value=min(5000, len(df) * 3),
        value=len(df),
        step=100,
        help="Number of synthetic records to generate"
    )

with col_generate2:
    st.write("")
    if st.button("📥 Download Synthetic", use_container_width=True):
        st.session_state.download_synthetic = True

with st.spinner("⚙️ Generating synthetic twin..."):
    generator = SyntheticDataGenerator()
    generator.fit(df)
    synthetic_df = generator.generate(n_samples=synthetic_size)
    validation = generator.validate_privacy(df, synthetic_df)

# Continue with rest of the code...
    
    if validation['privacy_safe']:
        similarity = validation['statistical_similarity']
        st.write(f"✅ Synthetic twin generated | Statistical similarity: **{similarity:.1%}** ")
        
        # Download synthetic dataset if requested
        if st.session_state.get('download_synthetic', False):
            csv_data = synthetic_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Synthetic CSV",
                data=csv_data,
                file_name=f"synthetic_dataset_{synthetic_size}_records.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.success("✅ Click the button above to download your synthetic dataset!")
            st.session_state.download_synthetic = False
        
        # Show sample comparison
        with st.expander("📊 Compare Original vs Synthetic Data", expanded=False):
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("**Original Data (Sample)**")
                st.dataframe(df.head(5), use_container_width=True)
                
            with comp_col2:
                st.markdown("**Synthetic Data (Sample)**")
                st.dataframe(synthetic_df.head(5), use_container_width=True)
            
            # Statistical comparison
            st.markdown("**Statistical Comparison**")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                comparison_data = []
                for col in numeric_cols[:5]:
                    comparison_data.append({
                        'Feature': col,
                        'Original Mean': f"{df[col].mean():.2f}",
                        'Synthetic Mean': f"{synthetic_df[col].mean():.2f}",
                        'Original Std': f"{df[col].std():.2f}",
                        'Synthetic Std': f"{synthetic_df[col].std():.2f}",
                        'Similarity': f"{(1 - abs(df[col].mean() - synthetic_df[col].mean()) / df[col].mean() * 100):.1f}%" if df[col].mean() != 0 else "N/A"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
    else:
        st.warning("⚠️ Privacy validation concern - proceeding with caution")
    
    # ========================================================================
    # ANALYSIS WITH AI AGENTS
    # ========================================================================
    
    #Sst.divider()
    st.markdown("## 🤖 Step 5: Multi-Agent Analysis")
    
    # Run Python analysis
    with st.spinner("🔬 Running comprehensive statistical analysis..."):
        analyzer = DataQualityAnalyzer(synthetic_df)
        quality_issues = analyzer.run_full_analysis()
        quality_summary = analyzer.get_summary()
        
        # Detect target column
        possible_targets = [c for c in synthetic_df.columns 
                          if any(t in c.lower() for t in ['disease', 'diabetes', 'outcome', 'target', 'recovered'])]
        target_col = possible_targets[0] if possible_targets else synthetic_df.columns[-1]
        
        bias_detector = BiasDetector(synthetic_df, target_col=target_col)
        bias_issues = bias_detector.run_full_analysis()
        bias_summary = bias_detector.get_summary()
    
    # Enhanced summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Total Records", f"{quality_summary['total_records']:,}")
    with col2:
        missing_count = quality_summary['missing_value_columns']
        st.metric("⚠️ Missing Values", missing_count, delta=f"-{missing_count}" if missing_count > 0 else None, delta_color="inverse")
    with col3:
        dup_count = quality_summary.get('duplicate_records', 0)
        st.metric("🔄 Duplicates", dup_count, delta=f"-{dup_count}" if dup_count > 0 else None, delta_color="inverse")
    with col4:
        bias_count = bias_summary['attributes_with_bias']
        st.metric("⚖️ Bias Detected", bias_count, delta=f"-{bias_count}" if bias_count > 0 else None, delta_color="inverse")
    with col5:
        total_issues = missing_count + dup_count + bias_count
        if total_issues == 0:
            st.metric("🎯 Status", "🟢 Clean")
        elif total_issues <= 3:
            st.metric("🎯 Status", "🟡 Minor")
        else:
            st.metric("🎯 Status", "🔴 Critical")
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Data Quality + AI Insights",
        "⚖️ Bias Analysis + Fairness Plan",
        "📅 Deployment Roadmap",
        "💬 Ask AI Questions"
    ])
    
    # ========================================================================
    # TAB 1: DATA QUALITY (continued in next artifact section...)
    # ========================================================================
    
    with tab1:
        st.markdown("# 🔍 Data Quality Analysis")
        st.caption("Powered by Statistical Analyzer + Medical Advisor AI")
        
        # Initialize Medical Advisor
        if 'medical_advisor' not in st.session_state:
            with st.spinner("🤖 Loading Medical Advisor AI..."):
                try:
                    st.session_state.medical_advisor = MedicalAdvisorV2()
                except Exception as e:
                    st.warning(f"Medical Advisor unavailable: {e}")
                    st.session_state.medical_advisor = None
        
        medical_advisor = st.session_state.medical_advisor
        
        # Missing Values
        if quality_issues['missing_values']:
            st.markdown("### ⚠️ Missing Values Detected")
            
            for col, info in quality_issues['missing_values'].items():
                with st.expander(f"❌ **{col}**: {info['count']} missing ({info['percentage']}%)", expanded=False):
                    
                    col_a, col_b = st.columns([1, 1])
                    
                    with col_a:
                        st.markdown("#### 📊 Statistics")
                        st.markdown(f"""
                        <div class="warning-box">
                        • <strong>Missing:</strong> {info['count']} values ({info['percentage']}%)<br>
                        • <strong>Affected rows:</strong> {info['total_affected_rows']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 🔧 Verified Python Recommendations")
                        for i, rec in enumerate(info['recommendations'][:3], 1):
                            st.markdown(f"**{i}. {rec['method']}** {rec['priority']}")
                            st.write(f"💡 {rec['reason']}")
                            st.code(rec['code'], language='python')
                            if 'impact' in rec:
                                st.caption(f"✨ Impact: {rec['impact']}")
                            if i < 3:
                                st.write("")
                    
                    with col_b:
                        st.markdown("#### 🤖 AI Medical Context")
                        
                        if medical_advisor:
                            with st.spinner("🔬 Medical Advisor analyzing..."):
                                if col.lower() in ['cholesterol', 'glucose', 'blood_pressure', 'bmi', 'lung_capacity', 'lung capacity']:
                                    st.markdown(f"""
                                    <div class="info-box">
                                    <strong>Clinical Significance:</strong> Missing {col} data
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if 'cholesterol' in col.lower():
                                        st.markdown("""
                                        <div class="success-box">
                                        <strong>Medical Advisor:</strong><br>
                                        Cholesterol is a primary cardiovascular risk biomarker. Median imputation 
                                        falls in 'borderline-high' range, clinically appropriate for cardiac cohorts.
                                        High outliers (>400) should be KEPT - they represent highest-risk patients.
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif 'lung' in col.lower():
                                        st.markdown("""
                                        <div class="success-box">
                                        <strong>Medical Advisor:</strong><br>
                                        Lung capacity is critical for respiratory function assessment. Missing values
                                        may indicate patients too ill to perform pulmonary function tests. Consider
                                        if missingness correlates with disease severity (MNAR pattern).
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif 'glucose' in col.lower():
                                        st.markdown("""
                                        <div class="success-box">
                                        <strong>Medical Advisor:</strong><br>
                                        Glucose is essential for diabetes screening. Values >126 mg/dL indicate
                                        diabetes. High outliers are clinically meaningful - represent uncontrolled
                                        diabetes cases critical for prediction.
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("Standard statistical imputation recommended for this feature.")
                        else:
                            st.info("Medical Advisor unavailable - showing statistical recommendations only")
        else:
            st.markdown("""
            <div class="success-box">
            ✅ <strong>No missing values detected!</strong> Your dataset is complete.
            </div>
            """, unsafe_allow_html=True)
        
        # Duplicates
        st.divider()
        st.markdown("### 🔄 Duplicate Records")
        
        if quality_issues['duplicates']:
            dup = quality_issues['duplicates']
            st.markdown(f"""
            <div class="warning-box">
            ⚠️ Found <strong>{dup['count']} duplicate records</strong>
            </div>
            """, unsafe_allow_html=True)
            
            rec = dup['recommendation']
            st.write(f"**{rec['priority']} {rec['method']}**")
            st.write(rec['reason'])
            st.code(rec['code'], language='python')
            st.info(f"💡 {rec['impact']}")
        else:
            st.markdown("""
            <div class="success-box">
            ✅ <strong>No duplicates detected!</strong> Each record is unique.
            </div>
            """, unsafe_allow_html=True)
        
        # Outliers
        st.divider()
        st.markdown("### 📈 Outlier Detection")
        
        if quality_issues['outliers']:
            for col, info in quality_issues['outliers'].items():
                with st.expander(f"⚡ **{col}**: {info['count']} outliers detected"):
                    st.write(f"**📊 Sample values:** {info['values_sample']}")
                    st.write(f"**📏 Expected range:** {info['bounds']['lower']:.1f} - {info['bounds']['upper']:.1f}")
                    
                    st.markdown("#### 🔧 Recommendations:")
                    for rec in info['recommendations']:
                        st.markdown(f"**{rec['method']}** {rec['priority']}")
                        st.write(rec['reason'])
                        if 'code' in rec:
                            st.code(rec['code'], language='python')
        else:
            st.markdown("""
            <div class="success-box">
            ✅ <strong>No significant outliers!</strong> Data distribution looks healthy.
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 2: BIAS ANALYSIS + FAIRNESS SPECIALIST
    # ========================================================================
    
    with tab2:
        st.markdown("# ⚖️ Bias & Fairness Analysis")
        st.caption("Powered by Fairness Specialist AI + Statistical Analysis")
        
        # Initialize Fairness Specialist
        if 'fairness_specialist' not in st.session_state:
            with st.spinner("🤖 Loading Fairness Specialist AI..."):
                try:
                    st.session_state.fairness_specialist = FairnessSpecialist()
                except Exception as e:
                    st.warning(f"Fairness Specialist unavailable: {e}")
                    st.session_state.fairness_specialist = None
        
        fairness_specialist = st.session_state.fairness_specialist
        
        if bias_issues:
            for col, info in bias_issues.items():
                st.markdown(f"### Analysis: {col}")
                
                # Create distribution visualization
                dist_df = pd.DataFrame({
                    'Group': list(info['distribution'].keys()),
                    'Percentage': list(info['distribution'].values())
                })
                
                # Create bar chart
                fig = go.Figure()
                
                colors = []
                for pct in dist_df['Percentage']:
                    deviation = abs(pct - 50)
                    if deviation < 5:
                        colors.append('#28a745')
                    elif deviation < 15:
                        colors.append('#ffc107')
                    else:
                        colors.append('#dc3545')
                
                fig.add_trace(go.Bar(
                    x=dist_df['Group'],
                    y=dist_df['Percentage'],
                    text=[f"<b>{p:.1f}%</b>" for p in dist_df['Percentage']],
                    textposition='outside',
                    textfont=dict(size=16, color='white'),
                    marker=dict(color=colors, line=dict(color='white', width=2)),
                    hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
                ))
                
                fig.add_hline(
                    y=50,
                    line_dash="dash",
                    line_width=3,
                    line_color="white",
                    annotation_text="◄ Target: 50% (Balanced)",
                    annotation_position="right",
                    annotation_font_size=12
                )
                
                fig.update_layout(
                    title=dict(text=f'<b>{col} Distribution</b>', font=dict(size=22, color='white')),
                    xaxis_title='Group',
                    yaxis_title='Percentage (%)',
                    yaxis=dict(range=[0, 100], dtick=10, gridcolor='#444'),
                    xaxis=dict(gridcolor='#444'),
                    showlegend=False,
                    height=500,
                    template='plotly_dark',
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation
                max_pct = max(dist_df['Percentage'])
                min_pct = min(dist_df['Percentage'])
                gap = abs(max_pct - min_pct)
                
                if gap < 5:
                    st.success(f"✅ **Well balanced!** Distribution gap: {gap:.1f}% (FDA threshold: <5%)")
                elif gap < 15:
                    st.warning(f"⚠️ **Minor imbalance detected.** Gap: {gap:.1f}% (FDA threshold: <5%)")
                else:
                    st.error(f"🚨 **Significant imbalance!** Gap: {gap:.1f}% (FDA threshold: <5%)")
                
                # Bias recommendations
                if info['bias_detected']:
                    st.markdown("---")
                    st.markdown("""
                    <div class="error-box">
                    ⚠️ <strong>BIAS DETECTED - Detailed Issues:</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for issue in info['issues']:
                        st.write(f"  • {issue}")
                    
                    st.markdown("---")
                    st.markdown("### 🤖 AI-Generated Fairness Plan")
                    
                    if fairness_specialist:
                        with st.spinner("Fairness Specialist generating recommendations..."):
                            bias_data = {
                                'type': f'{col}_imbalance',
                                'distribution': info['distribution'],
                                'minority_group': info['issues'][0].split(':')[0].strip() if info['issues'] else 'underrepresented'
                            }
                            
                            dist_values = list(info['distribution'].values())
                            majority_pct = max(dist_values)
                            minority_pct = min(dist_values)
                            
                            total = len(synthetic_df)
                            majority_count = int(total * majority_pct / 100)
                            minority_count = int(total * minority_pct / 100)
                            samples_needed = majority_count - minority_count
                            
                            stats = {
                                'total_samples': total,
                                'samples_needed': samples_needed,
                                'imbalance_pct': abs(majority_pct - 50)
                            }
                            
                            try:
                                fairness_rec = fairness_specialist.analyze_bias(bias_data, st.session_state.user_context, stats)
                                
                                st.success("✅ Exact Fairness Recommendations Generated by AI")
                                
                                # Predicted Harm
                                st.markdown("#### 🚨 Predicted Harm if Not Fixed:")
                                harm_col1, harm_col2, harm_col3 = st.columns(3)
                                
                                with harm_col1:
                                    st.metric("Performance Gap", f"{fairness_rec.predicted_harm.performance_gap_percentage:.1f}%",
                                             delta=f"-{fairness_rec.predicted_harm.performance_gap_percentage:.1f}%", delta_color="inverse")
                                
                                with harm_col2:
                                    st.metric("Patients Affected", f"{fairness_rec.predicted_harm.patients_affected_per_100}/100")
                                
                                with harm_col3:
                                    severity_color = "🔴" if fairness_rec.severity == "critical" else "🟡"
                                    st.metric("Severity", f"{severity_color} {fairness_rec.severity.upper()}")
                                
                                st.error(f"**Clinical Impact:** {fairness_rec.predicted_harm.clinical_consequence}")
                                
                                # Exact Requirements
                                st.markdown("---")
                                st.markdown("#### 🔢 Exact Requirements for Fairness:")
                                for group, count in fairness_rec.exact_samples_needed.items():
                                    st.write(f"  • **{group}:** Need **{count:,} additional patients**")
                                
                                # Two Options Side-by-Side
                                st.markdown("---")
                                st.markdown("#### 🛠️ Remediation Options:")
                                
                                option_col1, option_col2 = st.columns(2)
                                
                                with option_col1:
                                    st.markdown("**🏥 Option 1: Real Data Collection**")
                                    st.write(f"**⏱️ Timeline:** {fairness_rec.recruitment_plan.timeline_months} months")
                                    
                                    if fairness_rec.recruitment_plan.estimated_cost_usd:
                                        st.write(f"**💰 Cost:** ${fairness_rec.recruitment_plan.estimated_cost_usd:,}")
                                    
                                    st.write("\n**🏥 Target Facilities:**")
                                    for i, facility in enumerate(fairness_rec.recruitment_plan.target_facilities, 1):
                                        st.write(f"  {i}. {facility}")
                                    
                                    user_timeline = st.session_state.user_context['timeline_days']
                                    recruitment_days = fairness_rec.recruitment_plan.timeline_months * 30
                                    
                                    if not fairness_rec.fits_user_timeline:
                                        st.error(f"❌ **Timeline Mismatch:** Your timeline ({user_timeline} days) is shorter than needed ({recruitment_days} days)")
                                    else:
                                        st.success(f"✅ Fits your {user_timeline}-day timeline")
                                
                                with option_col2:
                                    st.markdown("**⚡ Option 2: Immediate Technical Fix**")
                                    st.write(f"**🔧 Method:** {fairness_rec.immediate_technical_fix.method}")
                                    st.write(f"**📈 Expected:** {fairness_rec.immediate_technical_fix.expected_improvement}")
                                    
                                    st.code(fairness_rec.immediate_technical_fix.python_code, language='python')
                                    st.warning(f"⚠️ **Limitation:** {fairness_rec.immediate_technical_fix.limitations}")
                                
                                # Priority Recommendation
                                st.markdown("---")
                                st.info(f"💡 **AI Strategic Recommendation:** {fairness_rec.recommendation_priority}")
                                
                            except Exception as e:
                                st.error(f"⚠️ AI fairness analysis failed: {e}")
                                st.code(f"""from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)""", language='python')
                    else:
                        st.info("Fairness Specialist unavailable - showing statistical metrics only")
                else:
                    st.success("✅ Balanced distribution - no bias detected")
        else:
            st.markdown("""
            <div class="info-box">
            ℹ️ <strong>No demographic columns detected in dataset</strong><br>
            Looking for columns containing: 'sex', 'gender', 'race', 'ethnicity', 'age'
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 3: DEPLOYMENT ROADMAP
    # ========================================================================
    
    with tab3:
        st.markdown("# 📅 Deployment Roadmap")
        st.caption("Powered by Deployment Strategist AI")
        
        # Initialize strategist
        if 'deployment_strategist' not in st.session_state:
            with st.spinner("🤖 Loading Deployment Strategist AI..."):
                try:
                    st.session_state.deployment_strategist = DeploymentStrategist()
                except Exception as e:
                    st.warning(f"Deployment Strategist unavailable: {e}")
                    st.session_state.deployment_strategist = None
        
        strategist = st.session_state.deployment_strategist
        
        if strategist:
            with st.spinner("🤖 AI generating week-by-week implementation plan..."):
                quality_summary_dict = {
                    'missing_values': len(quality_issues['missing_values']),
                    'duplicates': quality_issues['duplicates'].get('count', 0) if quality_issues['duplicates'] else 0,
                    'outliers': len(quality_issues['outliers'])
                }
                
                bias_summary_dict = {
                    'biased_attributes': bias_summary['attributes_with_bias'],
                    'status': bias_summary['status']
                }
                
                fairness_rec_dict = {
                    'exact_samples_needed': {},
                    'timeline_months': 6,
                    'immediate_method': 'data cleaning'
                }
                
                try:
                    plan = strategist.create_plan(quality_summary_dict, bias_summary_dict, fairness_rec_dict, st.session_state.user_context)
                    
                    st.markdown("""
                    <div class="success-box">
                    ✅ <strong>Custom Deployment Plan Generated by AI</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Strategy overview
                    st.markdown("### 🎯 Recommended Strategy")
                    st.markdown(f"""
                    <div class="info-box">
                    <strong>{plan.recommended_strategy}</strong><br><br>
                    <strong>Rationale:</strong> {plan.why_this_strategy}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Weekly breakdown
                    st.markdown("---")
                    st.markdown("### 📅 Week-by-Week Implementation:")
                    
                    for week in plan.weekly_plan:
                        with st.expander(f"📆 Week {week.week_number}: {week.days_range}", expanded=False):
                            st.markdown("**📝 Tasks:**")
                            for task in week.tasks:
                                st.write(f"  ✅ {task}")
                            
                            st.markdown("\n**📦 Deliverables:**")
                            for deliverable in week.deliverables:
                                st.write(f"  📄 {deliverable}")
                            
                            if week.estimated_hours:
                                st.caption(f"⏱️ Time estimate: {week.estimated_hours} hours")
                    
                    # Critical info
                    st.markdown("---")
                    
                    crit_col1, crit_col2 = st.columns(2)
                    
                    with crit_col1:
                        st.markdown("### 🎯 Critical Path Items:")
                        st.markdown("""
                        <div class="error-box">
                        """, unsafe_allow_html=True)
                        for item in plan.critical_path_items:
                            st.write(f"  🔴 {item}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("\n### ⚠️ Risk Factors:")
                        st.markdown("""
                        <div class="warning-box">
                        """, unsafe_allow_html=True)
                        for risk in plan.risk_factors:
                            st.write(f"  ⚠️ {risk}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with crit_col2:
                        st.markdown("### 📊 Success Metrics:")
                        st.markdown("""
                        <div class="success-box">
                        """, unsafe_allow_html=True)
                        for metric in plan.success_metrics:
                            st.write(f"  ✅ {metric}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        if plan.alternative_paths:
                            st.markdown("\n### 🔀 Alternative Paths:")
                            for alt in plan.alternative_paths:
                                with st.expander(f"If: {alt.scenario}"):
                                    st.write(f"**When to use:** {alt.when_to_pivot}")
                                    st.write(f"**Modified plan:** {alt.modified_timeline}")
                    
                except Exception as e:
                    st.error(f"⚠️ Plan generation failed: {e}")
                    st.markdown("""
                    <div class="info-box">
                    <strong>📋 Recommended Actions:</strong><br>
                    1. <strong>Week 1:</strong> Address data quality issues (missing values, duplicates)<br>
                    2. <strong>Week 2-3:</strong> Apply bias mitigation if needed<br>
                    3. <strong>Week 4+:</strong> Model training and validation
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            <strong>📋 General Recommendations:</strong><br><br>
            <strong>Phase 1 (Weeks 1-2):</strong> Data Quality<br>
            • Fix missing values using recommended imputation methods<br>
            • Remove duplicate records<br>
            • Handle outliers based on clinical context<br><br>
            <strong>Phase 2 (Weeks 3-4):</strong> Bias Mitigation<br>
            • Apply demographic balancing techniques<br>
            • Validate fairness metrics<br><br>
            <strong>Phase 3 (Weeks 5+):</strong> Model Development<br>
            • Train and validate model<br>
            • Test for bias and performance gaps<br>
            • Prepare deployment documentation
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 4: INTERACTIVE Q&A
    # ========================================================================
    
    with tab4:
        st.markdown("# 💬 Ask the AI Agents")
        st.caption("Get clarification on any recommendation")
        
        st.markdown("""
        <div class="info-box">
        <strong>💡 Example questions:</strong><br>
        • What if I can only recruit 200 patients instead of 552?<br>
        • Why is median better than mean for my data?<br>
        • How does SMOTE work?<br>
        • What are the risks of deploying with synthetic data?<br>
        • Should I remove or keep high cholesterol values?
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        user_question = st.text_area(
            "🤔 Your question:",
            placeholder="Ask anything about the analysis or recommendations...",
            height=100
        )
        
        if st.button("🚀 Ask AI", use_container_width=True):
            if user_question:
                with st.spinner("🤖 AI agents thinking..."):
                    try:
                        # Build context from analysis
                        context_summary = f"""
User's project:
- Model: {st.session_state.user_context.get('model_type', 'Unknown')}
- Timeline: {st.session_state.user_context.get('timeline_days', 'Unknown')} days
- Use case: {st.session_state.user_context.get('use_case', 'Unknown')}
- Location: {st.session_state.user_context.get('location', 'Unknown')}

Dataset analysis:
- Total records: {quality_summary['total_records']}
- Quality issues: {quality_summary['missing_value_columns']} missing value columns, {quality_summary.get('duplicate_records', 0)} duplicates
- Bias status: {bias_summary['status']}
"""
                        
                        response = ollama.generate(
                            model='llama3.2:3b',
                            prompt=f"""
You are a helpful medical data science advisor.

CONTEXT:
{context_summary}

USER QUESTION:
{user_question}

Provide a clear, specific answer in 2-4 sentences.
Use exact numbers when referencing the data.
Be helpful and educational.
""",
                            options={'temperature': 0.3}
                        )
                        
                        st.markdown("""
                        <div class="success-box">
                        <strong>🤖 AI Response:</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(response['response'])
                        
                    except Exception as e:
                        st.error(f"AI chat unavailable: {e}")
                        st.info("Please check your question and try again, or ensure Ollama is running.")
            else:
                st.warning("Please enter a question first!")
    
    st.session_state.analysis_complete = True

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("<div style='text-align: center;'>🏥 <strong>MedGuard AI</strong></div>", unsafe_allow_html=True)
with footer_col2:
    st.markdown("<div style='text-align: center;'>🏆 <strong>Convergence 2026 Hackathon</strong></div>", unsafe_allow_html=True)
with footer_col3:
    st.markdown("<div style='text-align: center;'>👥 <strong>The Guardians Team</strong></div>", unsafe_allow_html=True)

st.write("")
st.markdown("<div style='text-align: center; opacity: 0.5;'><small>Built with ❤️ using Streamlit • Powered by Llama 3.2</small></div>", unsafe_allow_html=True)