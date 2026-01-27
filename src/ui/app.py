"""
MedData Guardian - Streamlit Dashboard
Main UI for data quality and bias detection
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.phi_scanner import PHIScanner, PHIDetectedError
from utils.data_quality import DataQualityAnalyzer
from utils.bias_detection import BiasDetector

# Page config
st.set_page_config(
    page_title="MedData Guardian",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .subheader {
        font-size: 1.2rem;
        color: #666;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🏥 MedData Guardian</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Healthcare Data Quality & Bias Auditor</p>', unsafe_allow_html=True)
st.caption("🔒 HIPAA Compliant - All processing happens locally")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    **MedData Guardian** detects:
    - Data quality issues (missing values, duplicates, outliers)
    - Demographic bias and fairness violations
    
    **HIPAA Compliant:**
    - PHI scanner prevents accidental uploads
    - All processing local
    - No data storage
    """)
    
    st.header("Demo Datasets")
    demo_option = st.selectbox(
        "Load example dataset:",
        ["None", 
         "Dataset 1: Heart Disease (Quality Issues)",
         "Dataset 2: Diabetes (Gender Bias)",
         "Dataset 3: Heart Disease (Indigenous Bias)",
         "Dataset 4: Combined Problems"]
    )

# Main content
uploaded_file = None

if demo_option != "None":
    # Load demo dataset
    demo_files = {
        "Dataset 1: Heart Disease (Quality Issues)": "data/synthetic/dataset1_heart_disease_quality.csv",
        "Dataset 2: Diabetes (Gender Bias)": "data/synthetic/dataset2_diabetes_gender_bias.csv",
        "Dataset 3: Heart Disease (Indigenous Bias)": "data/synthetic/dataset3_heart_disease_indigenous.csv",
        "Dataset 4: Combined Problems": "data/synthetic/dataset4_diabetes_combined.csv"
    }
    
    demo_path = demo_files[demo_option]
    
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        st.success(f"✅ Loaded demo dataset: {demo_option}")
        st.warning("⚠️ This is SYNTHETIC data for demonstration purposes")
    else:
        st.error(f"Demo file not found: {demo_path}")
        st.stop()
else:
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your healthcare dataset (CSV format)",
        type=['csv'],
        help="Dataset must be de-identified (no patient names, SSNs, etc.)"
    )
    
    if uploaded_file is None:
        st.info("👆 Upload a CSV file or select a demo dataset from the sidebar to begin analysis")
        st.stop()
    
    df = pd.read_csv(uploaded_file)

# Process dataset
if df is not None:
    st.success(f"✅ Dataset loaded: {len(df)} records, {len(df.columns)} features")
    
    # Show preview
    with st.expander("📊 Dataset Preview (First 10 Rows)"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # PHI SCAN (Layer 1)
    st.markdown("---")
    st.subheader("🔒 HIPAA Compliance Check")
    
    with st.spinner("Scanning for Protected Health Information..."):
        scanner = PHIScanner()
        is_safe, violations = scanner.scan_dataset(df)
    
    if not is_safe:
        st.error("⚠️ PHI DETECTED - Cannot Process Dataset")
        st.error("HIPAA Violations Found:")
        for v in violations:
            st.write(f"  • {v}")
        st.warning("Please de-identify your dataset before upload.")
        st.stop()
    else:
        st.success("✅ PHI Scan Passed - No identifiable information detected")
    
    # Create tabs
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🔍 Data Quality Report", "⚖️ Bias & Fairness Audit", "📄 Summary Report"])
    
    # TAB 1: DATA QUALITY
    with tab1:
        st.header("Data Quality Analysis")
        
        with st.spinner("Analyzing data quality..."):
            analyzer = DataQualityAnalyzer(df)
            issues = analyzer.run_full_analysis()
            summary = analyzer.get_summary()
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", summary['total_records'])
        with col2:
            st.metric("Missing Value Issues", summary['missing_value_columns'])
        with col3:
            st.metric("Duplicate Records", summary['duplicate_records'])
        with col4:
            st.metric("Outlier Columns", summary['outlier_columns'])
        
        st.markdown("---")
        
        # Missing Values Section
        st.subheader("⚠️ Missing Values")
        if issues['missing_values']:
            for col, info in issues['missing_values'].items():
                with st.expander(f"❌ {col}: {info['count']} missing ({info['percentage']}%)", expanded=True):
                    st.write(f"**Affected rows:** {info['total_affected_rows']}")
                    st.write(f"**Sample rows:** {info['rows_sample']}")
                    
                    st.write("### 🤖 Recommendations:")
                    for i, rec in enumerate(info['recommendations'], 1):
                        st.markdown(f"#### Option {i}: {rec['method']} {rec['priority']}")
                        st.write(f"**Why:** {rec['reason']}")
                        if 'code' in rec:
                            st.code(rec['code'], language='python')
                        if 'impact' in rec:
                            st.info(f"💡 Impact: {rec['impact']}")
                        st.divider()
        else:
            st.success("✅ No missing values detected!")
        
        # Duplicates Section
        st.markdown("---")
        st.subheader("🔄 Duplicate Records")
        if issues['duplicates']:
            dup_info = issues['duplicates']
            st.warning(f"⚠️ Found {dup_info['count']} duplicate records")
            
            rec = dup_info['recommendation']
            st.write(f"### {rec['priority']} {rec['method']}")
            st.write(f"**Why:** {rec['reason']}")
            st.code(rec['code'], language='python')
            st.info(f"💡 Impact: {rec['impact']}")
        else:
            st.success("✅ No duplicates detected!")
        
        # Outliers Section
        st.markdown("---")
        st.subheader("📈 Outliers")
        if issues['outliers']:
            for col, info in issues['outliers'].items():
                with st.expander(f"⚡ {col}: {info['count']} outliers detected"):
                    st.write(f"**Sample values:** {info['values_sample']}")
                    st.write(f"**Expected range:** {info['bounds']['lower']:.2f} to {info['bounds']['upper']:.2f}")
                    
                    st.write("### Recommendations:")
                    for rec in info['recommendations']:
                        st.markdown(f"**{rec['method']}** {rec['priority']}")
                        st.write(f"- {rec['reason']}")
                        if 'code' in rec:
                            st.code(rec['code'], language='python')
        else:
            st.success("✅ No significant outliers detected!")
    
    # TAB 2: BIAS DETECTION
    with tab2:
        st.header("Bias & Fairness Analysis")
        
        # Detect target column
        possible_targets = [col for col in df.columns if any(term in col.lower() for term in ['disease', 'diabetes', 'outcome', 'target', 'label'])]
        
        if possible_targets:
            target_col = possible_targets[0]
        else:
            target_col = df.columns[-1]  # Default to last column
        
        st.info(f"📊 Target variable: **{target_col}**")
        
        with st.spinner("Analyzing demographic bias..."):
            detector = BiasDetector(df, target_col=target_col)
            bias_issues = detector.run_full_analysis()
            bias_summary = detector.get_summary()
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Demographics Analyzed", bias_summary['attributes_analyzed'])
        with col2:
            st.metric("Biased Attributes", bias_summary['attributes_with_bias'])
        with col3:
            status = bias_summary['status']
            st.metric("Overall Status", status, delta="FAIL" if status == "BIASED" else "PASS")
        
        st.markdown("---")
        
        # Show bias for each demographic
        if bias_issues:
            for col, info in bias_issues.items():
                st.subheader(f"Analysis: {col}")
                
                # Distribution chart
                dist_data = pd.DataFrame({
                    'Group': list(info['distribution'].keys()),
                    'Percentage': list(info['distribution'].values())
                })
                
                fig = px.bar(
                    dist_data,
                    x='Group',
                    y='Percentage',
                    title=f'{col} Distribution',
                    color='Percentage',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Issues
                if info['bias_detected']:
                    st.error("⚠️ BIAS DETECTED")
                    for issue in info['issues']:
                        st.write(f"  • {issue}")
                else:
                    st.success("✅ Balanced distribution")
        else:
            st.info("No demographic columns detected in dataset")
    
    # TAB 3: SUMMARY REPORT
    with tab3:
        st.header("Executive Summary")
        
        st.subheader("📊 Dataset Overview")
        st.write(f"- **Total Records:** {len(df)}")
        st.write(f"- **Total Features:** {len(df.columns)}")
        st.write(f"- **Dataset Size:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        st.markdown("---")
        
        st.subheader("🔍 Quality Assessment")
        if summary['missing_value_columns'] > 0 or summary['duplicate_records'] > 0:
            st.warning("⚠️ Issues Detected")
            st.write(f"- Missing value issues: {summary['missing_value_columns']} columns")
            st.write(f"- Duplicate records: {summary['duplicate_records']}")
            st.write(f"- Outlier columns: {summary['outlier_columns']}")
        else:
            st.success("✅ No major quality issues detected")
        
        st.markdown("---")
        
        st.subheader("⚖️ Bias Assessment")
        if bias_summary['status'] == 'BIASED':
            st.error("❌ Bias Detected")
            st.write(f"- Biased attributes: {bias_summary['attributes_with_bias']}")
            st.write(f"- Status: {bias_summary['status']}")
        else:
            st.success("✅ No significant bias detected")
        
        st.markdown("---")
        
        st.subheader("📋 Recommendation")
        if summary['missing_value_columns'] > 0 or summary['duplicate_records'] > 0 or bias_summary['status'] == 'BIASED':
            st.warning("⚠️ DO NOT use this dataset for model training without addressing the issues above.")
        else:
            st.success("✅ Dataset appears suitable for model training after standard preprocessing.")

