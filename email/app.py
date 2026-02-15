import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import base64


# Page configuration
st.set_page_config(
    page_title="MITRC - Fake Account Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Custom CSS for enhanced styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E88E5, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
        animation: fadeIn 1s ease-in;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1rem 0;
        animation: slideUp 0.5s ease-out;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .fake-account {
        background: linear-gradient(135deg, #FFCDD2, #EF9A9A);
        color: #C62828;
        border: 3px solid #C62828;
    }
    
    .real-account {
        background: linear-gradient(135deg, #C8E6C9, #A5D6A7);
        color: #2E7D32;
        border: 3px solid #2E7D32;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa, #e4e8eb);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    
    .sidebar-logo {
        display: flex;
        justify-content: center;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(120deg, #1E88E5, #7C4DFF);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 136, 229, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 136, 229, 0.4);
    }
    
    .info-box {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { 
            opacity: 0;
            transform: translateY(20px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
    }
    
    .stats-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    
    div[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    .sample-btn {
        background: linear-gradient(120deg, #FF9800, #FF5722) !important;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'fake_email_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
        return model, scaler, feature_columns
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_dataset():
    """Load the dataset for analytics"""
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, 'spam_email.csv'))
        return df
    except FileNotFoundError:
        return None

def get_logo_base64():
    """Get logo as base64 string"""
    logo_path = os.path.join(BASE_DIR, 'mitrc_logo.png')
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def predict_account(model, scaler, feature_columns, features):
    """Make prediction for a single account"""
    input_df = pd.DataFrame([features], columns=feature_columns)
    scaled_features = scaler.transform(input_df)
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0]
    return prediction, probability

def show_sidebar():
    """Enhanced sidebar with logo and navigation"""
    with st.sidebar:
        # Display logo
        logo_base64 = get_logo_base64()
        if logo_base64:
            st.markdown(f"""
            <div class="sidebar-logo">
                <img src="E:\new_clg_project\mitrc.jpeg,{logo_base64}" width="150" style="border-radius: 10px;">
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 🎛️ Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔎 Detect Emails", "📊 Analytics", "🎮 Interactive Demo", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick Stats in Sidebar
        df = load_dataset()
        if df is not None:
            st.markdown("### 📈 Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", len(df), delta=None)
            with col2:
                spam_pct = (df['is_spam'].sum()/len(df)*100)
                st.metric("spam %", f"{spam_pct:.0f}%")
        
        st.markdown("---")
        
        # Theme toggle
        st.markdown("### ⚙️ Settings")
        if 'theme' not in st.session_state:
            st.session_state.theme = "Light"
        
        # Session info
        if 'prediction_count' not in st.session_state:
            st.session_state.prediction_count = 0
        
        st.info(f"🔍 Predictions made: {st.session_state.prediction_count}")
        
        st.markdown("---")
        st.markdown("*© 2026 MITRC*")
        
    return page
def show_home_page():
    """Enhanced home page"""
    # Animated header
    st.markdown('<p class="main-header">🔍 Fake email Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Machine Learning | Built at MITRC</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
        # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Smart Detection</h3>
            <p>Advanced ML algorithms analyze 12+ account features to identify fake profiles with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Real-time Analysis</h3>
            <p>Get instant predictions with detailed confidence scores and probability breakdowns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Visual Insights</h3>
            <p>Interactive charts and analytics to understand patterns in fake vs real emails.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    

    # Statistics Dashboard
    df = load_dataset()
    if df is not None:
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stats-number">{len(df)}</div>
                <div class="stats-label">Total Emails</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            real_count = len(df) - df['is_spam'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="stats-number" style="color: #2E7D32;">{real_count}</div>
                <div class="stats-label">Real Email</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            fake_count = df['is_spam'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="stats-number" style="color: #C62828;">{fake_count}</div>
                <div class="stats-label">Fake Email</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            accuracy = 100.0  # From training
            st.markdown(f"""
            <div class="metric-card">
                <div class="stats-number" style="color: #7C4DFF;">{accuracy:.0f}%</div>
                <div class="stats-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)


                # Quick visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                df, 
                names=df['is_spam'].map({0: 'Real', 1: 'Fake'}),
                title="Email Distribution",
                color_discrete_sequence=['#4CAF50', '#F44336'],
                hole=0.4
            )
            fig.update_layout(
                font_family="Poppins",
                title_font_size=18,
                legend_title_text=""
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance (if model exists)
            model, scaler, feature_columns = load_model()
            if model is not None:
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True).tail(6)
                
                fig = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Top Feature Importance",
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    font_family="Poppins",
                    title_font_size=18,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Dataset not found. Please train the model first!")
        st.code("python src/train_model.py", language="bash")



def show_detection_page(model, scaler, feature_columns):
    """Enhanced detection page with more interactivity"""
    st.markdown("## 🔎 Detect Fake Emails")
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first.")
        st.code("python src/train_model.py", language="bash")
        return
    
        # Info box
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ How it works:</strong> Enter the emails details below and click "Analyze email" 
        to get an instant prediction with confidence score.
    </div>
    """, unsafe_allow_html=True)
    
    # Sample data buttons
    st.markdown("### 🎲 Try Sample Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📱 Load Real email Sample", use_container_width=True):
            st.session_state.sample_data = {
                'has_link':1,'has_money_words':0,'has_urgent_words':0,'has_caps':0,'num_exclamation':3,'email_length':100,
                'num_digits':6,'num_special_chars':5
            }
            st.rerun()
    
    with col2:
        if st.button("🤖 Load Fake email Sample", use_container_width=True):
            st.session_state.sample_data = {
               'has_link':0,'has_money_words':1,'has_urgent_words':1,'has_caps':1,'num_exclamation':8,'email_length':10,
                'num_digits':17,'num_special_chars':20
            }
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Form", use_container_width=True):
            if 'sample_data' in st.session_state:
                del st.session_state.sample_data
            st.rerun()
    
    st.markdown("---")
    
    # Get sample data or defaults
    sample = st.session_state.get('sample_data', {})
    
    # Input form with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📱 Email Information")
        
        has_link = st.selectbox(
            "Has link?", 
            options=[1, 0], 
            format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
            index=0 if sample.get('has_link', 1) == 1 else 1
        )

        has_money_words = st.selectbox(
            "Has money words?", 
            options=[1, 0], 
            format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
            index=0 if sample.get('has_money_words', 1) == 1 else 1
        )

        has_urgent_words = st.selectbox(
            "Has urgent words?", 
            options=[1, 0], 
            format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
            index=0 if sample.get('has_urgent_words', 1) == 1 else 1
        )

        has_caps = st.selectbox(
            "Has caps?", 
            options=[1, 0], 
            format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
            index=0 if sample.get('has_caps', 1) == 1 else 1
        )
        
        
    with col2:
        st.markdown("### 📊 Engagement Metrics")
        
        
        num_exclamation = st.number_input(
            "Number of exclamation !", 
            min_value=0, max_value=1000000, 
            value=sample.get('num_exclamation', 15),
            help="number of exclamation"
        )
        
        email_length = st.number_input(
            "length of email 💬", 
            min_value=0, max_value=100000, 
            value=sample.get('email_length', 120),
            help="length of email"
        )
        
        num_digits = st.slider(
            "number of digits 🔤", 
            min_value=1, max_value=30, 
            value=sample.get('num_digits', 20),
            help="number of digits"
        )
        num_special_chars = st.slider(
            "number of special character 🔤", 
            min_value=1, max_value=30, 
            value=sample.get('num_special_chars', 20),
            help="number of digits"
        )
        

        st.markdown("---")

        features = {
            'has_link':has_link,
            'has_money_words':has_money_words,
            'has_urgent_words':has_urgent_words,
            'has_caps':has_caps,
            'num_exclamation':num_exclamation,
            'email_length':email_length,
            'num_digits':num_digits,
            'num_special_chars':num_special_chars
        }

        # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
    
    if analyze_clicked:
        # Progress animation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("🔄 Loading features...")
            elif i < 60:
                status_text.text("🧠 Running ML model...")
            elif i < 90:
                status_text.text("📊 Calculating probabilities...")
            else:
                status_text.text("✅ Analysis complete!")
            time.sleep(0.02)
        
        progress_bar.empty()
        status_text.empty()
        
        # Get prediction
        feature_values = [features[col] for col in feature_columns]
        prediction, probability = predict_account(model, scaler, feature_columns, feature_values)
        
        # Update prediction count
        st.session_state.prediction_count = st.session_state.get('prediction_count', 0) + 1


         # Display result with animation
        st.markdown("### 📋 Analysis Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 1:
                
                st.markdown("""
                <div class="result-box fake-account pulse-animation">
                    ⚠️ SPAM EMAIL DETECTED
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                
                st.markdown("""
                <div class="result-box real-account pulse-animation">
                    ✅ REAL EMAIL
                </div>
                """, unsafe_allow_html=True)
                st.snow()


        # Detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Confidence Gauge")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability[1] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fake Account Probability (%)", 'font': {'size': 16}},
                delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#F44336" if prediction == 1 else "#4CAF50"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#C8E6C9'},
                        {'range': [30, 70], 'color': '#FFF9C4'},
                        {'range': [70, 100], 'color': '#FFCDD2'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))


            fig.update_layout(height=350, font_family="Poppins")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Probability Distribution")
            
            prob_df = pd.DataFrame({
                'Type': ['Real Account', 'Fake Account'],
                'Probability': [probability[0] * 100, probability[1] * 100],
                'Color': ['#4CAF50', '#F44336']
            })
            
            fig = px.bar(
                prob_df, 
                x='Type', 
                y='Probability',
                color='Type',
                color_discrete_map={'Real Account': '#4CAF50', 'Fake Account': '#F44336'},
                text=prob_df['Probability'].apply(lambda x: f'{x:.1f}%')
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                showlegend=False, 
                height=350,
                font_family="Poppins",
                yaxis_title="Probability (%)",
                xaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature summary
        st.markdown("### 📝 Input Features Summary")
        
        feature_df = pd.DataFrame([features])
        feature_df = feature_df.T.reset_index()
        feature_df.columns = ['Feature', 'Value']
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(feature_df.iloc[:6], use_container_width=True, hide_index=True)
        with col2:
            st.dataframe(feature_df.iloc[6:], use_container_width=True, hide_index=True)
        
        # Save to history
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        
        st.session_state.prediction_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'result': 'Fake' if prediction == 1 else 'Real',
            'confidence': f"{max(probability)*100:.1f}%",
            'fake_prob': f"{probability[1]*100:.1f}%",
            **features
        })
        
        # Download report button
        st.markdown("---")
        report_data = {
            'Analysis Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Prediction': 'Fake Account' if prediction == 1 else 'Real Account',
            'Fake Probability': f"{probability[1]*100:.2f}%",
            'Real Probability': f"{probability[0]*100:.2f}%",
            **features
        }
        report_df = pd.DataFrame([report_data])
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download Report as CSV",
                data=report_df.to_csv(index=False),
                file_name=f"account_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def show_analytics_page():
    """Enhanced analytics page"""
    st.markdown("## 📊 Data Analytics Dashboard")
    

    df = load_dataset()
    
    if df is None:
        st.error("Dataset not found! Please generate the dataset first.")
        return
    
    df['email_type'] = df['is_spam'].map({0: 'Real', 1: 'spam'})


    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔬 Feature Analysis", "📉 Correlations", "📜 History"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Emails Distribution")
            fig = px.pie(
                df, 
                names='email_type',
                title="Real vs spam email",
                color='email_type',
                color_discrete_map={'Real': '#4CAF50', 'Fake': '#F44336'},
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(font_family="Poppins")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("###  Emails Distribution")
            fig = px.histogram(
                df, 
                x='email_length', 
                color='email_type',
                barmode='overlay', 
                opacity=0.7,
                color_discrete_map={'Real': '#4CAF50', 'Fake': '#F44336'},
                title="Distribution by email length"
            )
            fig.update_layout(font_family="Poppins")
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        st.markdown("### 📊 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Real Accounts**")
            st.dataframe(df[df['is_spam'] == 0].describe().round(2), use_container_width=True)
        
        with col2:
            st.markdown("**Fake Accounts**")
            st.dataframe(df[df['is_spam'] == 1].describe().round(2), use_container_width=True)

    with tab2:
        st.markdown("### Feature Comparison")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            feature = st.selectbox(
                "Select Feature", 
                ['has_link',
            'has_money_words',
            'has_urgent_words',
            'has_caps',
            'num_exclamation',
            'email_length',
            'num_digits',
            'num_special_chars']
            )
            
            chart_type = st.radio(
                "Chart Type",
                ['Box Plot', 'Violin Plot', 'Histogram', 'Scatter']
            )
        
        with col2:
            if chart_type == 'Box Plot':
                fig = px.box(
                    df, x='email_type', y=feature, color='email_type',
                    color_discrete_map={'Real': '#4CAF50', 'Fake': '#F44336'},
                    title=f"{feature} by Email Type"
                )
            elif chart_type == 'Violin Plot':
                fig = px.violin(
                    df, x='email_type', y=feature, color='email_type',
                    box=True, color_discrete_map={'Real': '#4CAF50', 'Fake': '#F44336'},
                    title=f"{feature} by Email Type"
                )
            elif chart_type == 'Histogram':
                fig = px.histogram(
                    df, x=feature, color='email_type', barmode='overlay',
                    color_discrete_map={'Real': '#4CAF50', 'Fake': '#F44336'},
                    title=f"{feature} Distribution"
                )
            else:
                fig = px.scatter(
                    df, x=feature, y='has_link', color='email_type',
                    color_discrete_map={'Real': '#4CAF50', 'Fake': '#F44336'},
                    title=f"{feature} vs Followers"
                )
            
            fig.update_layout(font_family="Poppins", height=500)
            st.plotly_chart(fig, use_container_width=True)


        with tab3:

            st.markdown("### Feature Correlations")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix, 
                labels=dict(color="Correlation"),
                x=numeric_cols, 
                y=numeric_cols,
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap"
            )
            fig.update_layout(height=600, font_family="Poppins")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature relationships
            st.markdown("### Feature Relationships")
            col1, col2 = st.columns(2)
            
            with col1:
                x_feature = st.selectbox("X-axis Feature", numeric_cols, index=0)
            with col2:
                y_feature = st.selectbox("Y-axis Feature", numeric_cols, index=1)
            
            fig = px.scatter(
                df, x=x_feature, y=y_feature, color='email_type',
                color_discrete_map={'Real': '#4CAF50', 'Fake': '#F44336'},
                trendline="ols",
                title=f"{x_feature} vs {y_feature}"
            )
            fig.update_layout(font_family="Poppins")
            st.plotly_chart(fig, use_container_width=True)


        with tab4:
            st.markdown("### 📜 Prediction History")
            
            if 'prediction_history' in st.session_state and st.session_state.prediction_history:
                history_df = pd.DataFrame(st.session_state.prediction_history)
                
                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(history_df))
                with col2:
                    fake_count = len(history_df[history_df['result'] == 'Fake'])
                    st.metric("Fake Detected", fake_count)
                with col3:
                    real_count = len(history_df[history_df['result'] == 'Real'])
                    st.metric("Real Detected", real_count)
                
                st.dataframe(history_df, use_container_width=True)
        # Download history
            if 'prediction_history' in st.session_state and st.session_state.prediction_history:
                history_df = pd.DataFrame(st.session_state.prediction_history)

                st.dataframe(history_df, use_container_width=True)

                st.download_button(
                label="📥 Download History as CSV",
                data=history_df.to_csv(index=False),
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
    )
            else:
                st.info("No predictions yet. Go to 'Detect Account' to analyze accounts.")

            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
            else:
                st.info("No predictions yet. Go to 'Detect Account' to analyze accounts.")
    
    
def show_interactive_demo():
    """Interactive demo page"""
    st.markdown("## 🎮 Interactive Demo")
    
    model, scaler, feature_columns = load_model()
    
    if model is None:
        st.error("Model not found!")
        return
    
    st.markdown("""
    <div class="info-box">
        <strong>🎮 Interactive Mode:</strong> Adjust the sliders in real-time to see how different 
        features affect the prediction probability!
    </div>
    """, unsafe_allow_html=True)


       # Real-time sliders
    col1, col2 = st.columns(2)
    
    with col1:
        has_link = st.toggle("Has link", value=True)
        has_money_words = st.toggle("Has money word", value=False)
        has_urgent_words = st.toggle("Has Urgent words", value=False)
        
        email_length = st.slider("length of email", 50, 500, 1000, 100)
        num_special_chars = st.slider("number of special characters", 0, 10, 1000, 100)
    
    with col2:
        has_caps = st.toggle("Has capital lette", value=True)
        num_exclamation = st.toggle("Has exclamation", value=True)
        
        num_digits = st.slider("number of digits", 0, 5, 1000, 100)
      

    features = {
        'has_link':int(has_link),
            'has_money_words':int(has_money_words),
            'has_urgent_words':int(has_urgent_words),
            'has_caps':int(has_caps),
            'num_exclamation':num_exclamation,
            'email_length':email_length,
            'num_digits':num_digits,
            'num_special_chars':num_special_chars
    }



    feature_values = [features[col] for col in feature_columns]
    prediction, probability = predict_account(model, scaler, feature_columns, feature_values)
    
    st.markdown("---")
    
    # Live result display
    col1, col2, col3 = st.columns([1, 2, 1])


    with col2:
        spam_prob = probability[1] * 100
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box fake-account">
                ⚠️ SPAM ({spam_prob:.1f}%)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box real-account">
                ✅ REAL ({100-spam_prob:.1f}%)
            </div>
            """, unsafe_allow_html=True)


             # Live gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=spam_prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fake Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#F44336" if prediction == 1 else "#4CAF50"},
            'steps': [
                {'range': [0, 30], 'color': '#C8E6C9'},
                {'range': [30, 70], 'color': '#FFF9C4'},
                {'range': [70, 100], 'color': '#FFCDD2'}
            ]
        }
    ))
    fig.update_layout(height=300, font_family="Poppins")
    st.plotly_chart(fig, use_container_width=True)


def show_about_page():
    # Logo at top
        logo_base64 = get_logo_base64()
        if logo_base64:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <img src="data:image/png;base64,{logo_base64}" width="200" style="border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                </div>
                """, unsafe_allow_html=True)
    
        st.markdown("## ℹ️ About This Project")
    
        st.markdown("""
    <div class="feature-card">
        <h3>🎯 Project Overview</h3>
        <p>This is a <strong>Mini Project</strong> developed at <strong>MITRC</strong> that demonstrates 
        the use of <strong>Supervised Machine Learning</strong> to detect fake social media accounts.</p>
    </div>
    """, unsafe_allow_html=True)


        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("""
        ### 🔧 Technologies Used
        - **Python** - Programming Language
        - **Scikit-learn** - Machine Learning
        - **Streamlit** - Web Dashboard
        - **Plotly** - Interactive Charts
        - **Pandas & NumPy** - Data Processing
        
        ### 🧠 Algorithm
        **Random Forest Classifier** - An ensemble learning method that builds multiple 
        decision trees and merges them for more accurate predictions.
        """)
    
        with col2:
            st.markdown("""
        ### 📊 Features Analyzed
        | Feature | Description |
        |---------|-------------|
        | Has Link | Any Link given? |
        | Has Money words | Any Money Word |
        | Has Urgent Words | Any Urgent Word |
        | Has Caps | Has any capitals letter |
        | num exclamation | Any exclamation symbol |
        | Email Length | Length of Email |
        | Num Digits | Has numbere digits |
        | Num Special Character | Any Special Character |
        """)
    
        st.markdown("---")


        st.markdown("""
    ### 📁 Project Structure
    ```
    fake_account_detection/
    ├── data/
    │   ├── generate_dataset.py
    │   └── spam_email.csv
    ├── models/
    │   ├── fake_email_model.pkl
    │   ├── scaler.pkl
    │   └── feature_columns.pkl
    ├── src/
    │   └── train_model.py
    ├── app.py
    ├── mitrc_logo.jpeg
    └── requirements.txt
    ```
    """)

        st.markdown("---")
    
        col1, col2, col3 = st.columns(3)
        with col2:
            st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <p style="color: #666;">Created for</p>
            <h3 style="color: #1E88E5;">4th Semester Project</h3>
            <p style="color: #666;">MITRC © 2026</p>
        </div>
        """, unsafe_allow_html=True)



def main():
    """Main application"""
    # Show sidebar and get selected page
    page = show_sidebar()
    
    # Load model
    model, scaler, feature_columns = load_model()
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔎 Detect Emails":
        show_detection_page(model, scaler, feature_columns)
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "🎮 Interactive Demo":
        show_interactive_demo()
    elif page == "ℹ️ About":
        show_about_page()

if __name__ == "__main__":
    main()



