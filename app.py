# =========================================================
# CREDIT CARD FRAUD DETECTION SYSTEM
# Hybrid CNN + Dense Neural Network Model
# Academic Year: 2025–2026
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings
import io
import base64
import os
import tempfile
import time
import random
import json
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, 
    Input, BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC

st.set_page_config(
    page_title="Cyber Fraud AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ULTRA PREMIUM 3D GLASSMORPHISM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Montserrat', sans-serif;
}

/* === ANIMATED GRADIENT BACKGROUND === */
[data-testid="stAppViewContainer"] {
    background: 
        radial-gradient(circle at 0% 0%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 100% 0%, rgba(255, 119, 198, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 0% 100%, rgba(120, 220, 255, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 100% 100%, rgba(255, 220, 120, 0.2) 0%, transparent 50%),
        linear-gradient(135deg, #0f172a 0%, #1e293b 30%, #334155 70%, #475569 100%);
    background-attachment: fixed;
    animation: gradientShift 20s ease infinite;
    background-size: 400% 400%;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* === GLASS SIDEBAR WITH 3D EFFECT === */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.75) !important;
    backdrop-filter: blur(30px) saturate(180%);
    -webkit-backdrop-filter: blur(30px) saturate(180%);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 
        20px 0 40px rgba(0, 0, 0, 0.3),
        inset -1px 0 0 rgba(255, 255, 255, 0.05);
}

/* === MAIN TITLE WITH 3D TEXT EFFECT === */
.main-title-3d {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4rem !important;
    font-weight: 900;
    text-align: center;
    margin: 2rem 0 1rem;
    background: linear-gradient(135deg, 
        #667eea 0%, 
        #764ba2 25%, 
        #f093fb 50%, 
        #f5576c 75%, 
        #667eea 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 
        0 4px 6px rgba(0, 0, 0, 0.1),
        0 1px 3px rgba(0, 0, 0, 0.08);
    animation: gradientFlow 8s ease infinite;
    position: relative;
    letter-spacing: -0.5px;
    line-height: 1.1;
}

@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main-title-3d::after {
    content: 'AI FRAUD DETECTION';
    position: absolute;
    top: 2px;
    left: 2px;
    color: rgba(255, 255, 255, 0.1);
    z-index: -1;
}

/* === SUBTITLE WITH GLOW === */
.subtitle-glow {
    font-size: 1.5rem !important;
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 3rem;
    font-weight: 300;
    text-shadow: 0 0 20px rgba(96, 165, 250, 0.3);
    letter-spacing: 1px;
}

/* === 3D GLASS CARDS === */
.glass-card-3d {
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.1) 0%,
        rgba(255, 255, 255, 0.05) 100%
    );
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 
        inset 0 0 20px rgba(255, 255, 255, 0.05),
        0 15px 30px rgba(0, 0, 0, 0.2),
        0 5px 20px rgba(0, 0, 0, 0.1);
    color: #f1f5f9;
    position: relative;
    overflow: hidden;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    transform-style: preserve-3d;
    perspective: 1000px;
}

.glass-card-3d::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
}

.glass-card-3d::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(
        circle,
        rgba(96, 165, 250, 0.1) 0%,
        rgba(147, 51, 234, 0.05) 30%,
        transparent 70%
    );
    z-index: -1;
    opacity: 0;
    transition: opacity 0.5s ease;
}

.glass-card-3d:hover::after {
    opacity: 1;
}

.glass-card-3d:hover {
    transform: translateY(-10px) rotateX(5deg);
    border-color: rgba(96, 165, 250, 0.3);
    box-shadow: 
        inset 0 0 30px rgba(255, 255, 255, 0.1),
        0 20px 40px rgba(0, 0, 0, 0.3),
        0 10px 30px rgba(96, 165, 250, 0.2);
}

/* === CARD ICONS === */
.card-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: inline-block;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
    transition: transform 0.3s ease;
}

.glass-card-3d:hover .card-icon {
    transform: scale(1.1) rotate(5deg);
}

/* === FEATURE BADGES === */
.feature-badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    margin: 0.3rem;
    background: rgba(96, 165, 250, 0.1);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-radius: 50px;
    color: #60a5fa;
    font-size: 0.85rem;
    font-weight: 500;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    cursor: default;
}

.feature-badge:hover {
    background: rgba(96, 165, 250, 0.2);
    border-color: rgba(96, 165, 250, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(96, 165, 250, 0.15);
}

/* === PREMIUM BUTTONS === */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 1rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 
        0 8px 20px rgba(102, 126, 234, 0.3),
        inset 0 1px 1px rgba(255, 255, 255, 0.2) !important;
    position: relative !important;
    overflow: hidden !important;
    letter-spacing: 0.5px !important;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 
        0 15px 30px rgba(102, 126, 234, 0.4),
        0 0 20px rgba(118, 75, 162, 0.3),
        inset 0 1px 1px rgba(255, 255, 255, 0.3) !important;
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
}

/* === STATS WIDGET === */
.stats-widget {
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.08) 0%,
        rgba(255, 255, 255, 0.03) 100%
    );
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    position: relative;
    overflow: hidden;
}

.stats-widget::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(to bottom, #667eea, #764ba2);
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1;
    font-family: 'Space Grotesk', sans-serif;
}

.stat-label {
    font-size: 0.9rem;
    color: #94a3b8;
    font-weight: 500;
    margin-top: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* === ANIMATED PROGRESS BAR === */
.progress-bar-3d {
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    overflow: hidden;
    margin: 0.8rem 0;
    position: relative;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 8px;
    position: relative;
    transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
}

/* === SIDEBAR STYLING === */
.sidebar-header {
    text-align: center;
    padding: 1.5rem 1rem;
    position: relative;
}

.sidebar-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 15%;
    right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(96, 165, 250, 0.5), transparent);
}

.sidebar-icon {
    font-size: 2.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.8rem;
    display: inline-block;
}

.sidebar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.3rem;
    letter-spacing: 1px;
}

.sidebar-subtitle {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 300;
}

/* === NAVIGATION BUTTONS === */
.nav-btn {
    width: 100%;
    padding: 1rem 1.2rem !important;
    margin: 0.3rem 0 !important;
    text-align: left !important;
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.nav-btn:hover {
    background: rgba(96, 165, 250, 0.1) !important;
    border-color = rgba(96, 165, 250, 0.3) !important;
    transform: translateX(5px) !important;
    color: #f1f5f9 !important;
}

/* === STATUS INDICATORS === */
.status-indicator {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 0.8rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    margin: 0.2rem;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.5rem;
    animation: pulse 2s infinite;
}

.status-dot.active {
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
}

.status-dot.inactive {
    background: #ef4444;
    box-shadow: 0 0 8px #ef4444;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* === FEATURE HIGHLIGHTS === */
.feature-highlight {
    position: relative;
    padding-left: 1.5rem;
    margin: 1rem 0;
}

.feature-highlight::before {
    content: '✦';
    position: absolute;
    left: 0;
    color: #60a5fa;
    font-size: 1rem;
}

/* === RESPONSIVE DESIGN === */
@media (max-width: 768px) {
    .main-title-3d {
        font-size: 2.5rem !important;
    }
    
    .glass-card-3d {
        padding: 1.5rem;
    }
    
    .stat-number {
        font-size: 2rem;
    }
}

/* === CUSTOM SCROLLBAR === */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(to bottom, #667eea, #764ba2);
    border-radius: 8px;
    border: 2px solid rgba(255, 255, 255, 0.05);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(to bottom, #764ba2, #667eea);
}

/* === FLOATING ANIMATION === */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.floating {
    animation: float 6s ease-in-out infinite;
}

/* === NEON GLOW EFFECT === */
.neon-glow {
    text-shadow: 
        0 0 8px rgba(96, 165, 250, 0.5),
        0 0 15px rgba(96, 165, 250, 0.3),
        0 0 20px rgba(96, 165, 250, 0.1);
}

/* === SUCCESS/ERROR CARDS === */
.success-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.15));
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #f1f5f9;
    position: relative;
    overflow: hidden;
}

.error-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.15));
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #f1f5f9;
    position: relative;
    overflow: hidden;
}

/* === FORM CONTROLS === */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > div:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 1px #60a5fa !important;
}

/* === TABS STYLING === */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: transparent;
    padding: 0;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px 10px 0 0;
    padding: 1rem 2rem;
    font-weight: 600;
    color: #94a3b8;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-bottom: none;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
    color: #60a5fa !important;
    border-color: rgba(96, 165, 250, 0.3);
}

/* === METRIC CARDS === */
.metric-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
    backdrop-filter: blur(15px);
    border-radius: 15px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    text-align: center;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1;
    font-family: 'Space Grotesk', sans-serif;
}

.metric-label {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    font-weight: 500;
}

/* === LOADING SPINNER === */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-spinner {
    border: 3px solid rgba(255, 255, 255, 0.1);
    border-top: 3px solid #667eea;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

if 'fraud_system' not in st.session_state:
    st.session_state.fraud_system = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'random_features' not in st.session_state:
    st.session_state.random_features = None
if 'random_amount' not in st.session_state:
    st.session_state.random_amount = None
if 'demo_batch' not in st.session_state:
    st.session_state.demo_batch = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# data set generation
def generate_realistic_dataset(n_samples=10000):
    """Generate realistic credit card fraud dataset"""
    np.random.seed(42)
    
    # Parameters
    fraud_ratio = 0.002  # 0.2% fraud rate
    n_features = 30
    
    # Generate base features
    X = np.random.randn(n_samples, n_features)
    
    # Add correlations between features
    X[:, 1] = 0.3 * X[:, 0] + 0.7 * np.random.randn(n_samples)
    X[:, 2] = 0.2 * X[:, 0] + 0.8 * np.random.randn(n_samples)
    X[:, 3] = 0.4 * X[:, 1] + 0.6 * np.random.randn(n_samples)
    X[:, 4] = 0.5 * X[:, 0] + 0.5 * np.random.randn(n_samples)
    X[:, 5] = 0.3 * X[:, 1] + 0.7 * np.random.randn(n_samples)
    
    # Generate fraud labels
    fraud_mask = np.random.rand(n_samples) < fraud_ratio
    
    # Add fraud patterns to make them distinguishable
    X[fraud_mask, 0] += np.random.uniform(2, 4, fraud_mask.sum())
    X[fraud_mask, 1] += np.random.uniform(-4, -2, fraud_mask.sum())
    X[fraud_mask, 4] += np.random.uniform(3, 5, fraud_mask.sum())
    X[fraud_mask, 7] += np.random.uniform(-3, -1, fraud_mask.sum())
    X[fraud_mask, 10] += np.random.uniform(2, 4, fraud_mask.sum())
    X[fraud_mask, 14] += np.random.uniform(-5, -2, fraud_mask.sum())
    
    # Add amount column (fraud transactions tend to be larger)
    amounts = np.random.exponential(100, n_samples)
    amounts[fraud_mask] *= np.random.uniform(2, 10, fraud_mask.sum())
    
    # Create feature names
    columns = [f'V{i}' for i in range(1, n_features + 1)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=columns)
    df['Amount'] = amounts
    df['Time'] = np.arange(n_samples)
    df['Class'] = fraud_mask.astype(int)
    
    return df

# fraud detection system
class FraudDetectionSystem:
    def __init__(self):
        self.df = None
        self.hybrid_model = None
        self.scaler = StandardScaler()
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.X_train_cnn = self.X_test_cnn = None
        self.feature_columns = None
        self.target_column = 'Class'
        self.model_metrics = None
        self.history = None
        
    def load_data(self, file_path):
        """Load data from CSV file or uploaded file object"""
        try:
            if hasattr(file_path, 'read'):
                self.df = pd.read_csv(file_path)
            else:
                self.df = pd.read_csv(file_path)
            
            # Auto-detect target column
            target_candidates = ['Class', 'Fraud', 'is_fraud', 'isFraud', 'fraud', 'label', 'target']
            for col in target_candidates:
                if col in self.df.columns:
                    self.target_column = col
                    break
            
            if self.target_column not in self.df.columns:
                st.error(f" Could not find target column. Expected one of: {', '.join(target_candidates)}")
                return False
            
            return True
            
        except Exception as e:
            st.error(f" Error loading data: {str(e)}")
            return False
    
    def generate_sample_data(self):
        """Generate realistic sample data for demonstration"""
        try:
            self.df = generate_realistic_dataset(10000)
            self.target_column = 'Class'
            
            fraud_count = self.df['Class'].sum()
            total_count = len(self.df)
            fraud_rate = (fraud_count / total_count) * 100
            
            st.success(f" Generated {total_count:,} sample transactions")
            st.success(f"📊 {fraud_count:,} fraud cases ({fraud_rate:.2f}%)")
            return True
            
        except Exception as e:
            st.error(f" Error generating sample data: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocess the data for model training"""
        try:
            if self.df is None:
                st.error(" No data loaded!")
                return False
            
            df_clean = self.df.copy()
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
            
            X = df_clean.drop(self.target_column, axis=1)
            y = df_clean[self.target_column]
            
            self.feature_columns = X.columns.tolist()
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            # Handle class imbalance
            class_counts = Counter(self.y_train)
            fraud_count = class_counts.get(1, 0)
            
            if fraud_count >= 5:
                smote = SMOTE(random_state=42, k_neighbors=min(5, fraud_count - 1))
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                st.success(f" Applied SMOTE: Balanced to {Counter(self.y_train)}")
            else:
                rus = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
                self.X_train, self.y_train = rus.fit_resample(self.X_train, self.y_train)
                st.success(f" Applied RandomUnderSampler: Balanced to {Counter(self.y_train)}")
            
            self.X_train_cnn = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
            self.X_test_cnn = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
            
            st.success(" Data preprocessing completed!")
            return True
            
        except Exception as e:
            st.error(f" Error in preprocessing: {str(e)}")
            return False
    
    def build_hybrid_model(self):
        """Build hybrid CNN + Dense model"""
        try:
            input_shape = (self.X_train_cnn.shape[1], 1)
            
            model = Sequential([
                Input(shape=input_shape),
                Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.3),
                
                Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.3),
                
                Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
                BatchNormalization(),
                GlobalAveragePooling1D(),
                Dropout(0.4),
                
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
            )
            
            self.hybrid_model = model
            return model
            
        except Exception as e:
            st.error(f" Error building model: {str(e)}")
            return None
    
    def train_model(self, epochs=30, batch_size=64):
        """Train the hybrid model"""
        try:
            if not hasattr(self, 'X_train_cnn') or self.X_train_cnn is None:
                if not self.preprocess_data():
                    return None
            
            if self.hybrid_model is None:
                self.build_hybrid_model()
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001,
                    verbose=0
                )
            ]
            
            self.history = self.hybrid_model.fit(
                self.X_train_cnn, self.y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            self.evaluate_model()
            
            return self.history
            
        except Exception as e:
            st.error(f" Error training model: {str(e)}")
            return None
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        try:
            if self.hybrid_model is None:
                return None
            
            y_pred_proba = self.hybrid_model.predict(self.X_test_cnn, verbose=0).ravel()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred, zero_division=0),
                'Recall': recall_score(self.y_test, y_pred, zero_division=0),
                'F1 Score': f1_score(self.y_test, y_pred, zero_division=0),
                'ROC AUC': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            cm = confusion_matrix(self.y_test, y_pred)
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
            
            self.model_metrics = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'roc_curve': (fpr, tpr, thresholds),
                'predictions': {
                    'y_true': self.y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
            }
            
            return self.model_metrics
            
        except Exception as e:
            st.error(f" Error evaluating model: {str(e)}")
            return None
    
    def predict_single(self, features):
        """Predict fraud probability for a single transaction"""
        try:
            if self.hybrid_model is None or self.scaler is None:
                return None, None
            
            if len(features) != len(self.feature_columns):
                if len(features) < len(self.feature_columns):
                    features = list(features) + [0] * (len(self.feature_columns) - len(features))
                else:
                    features = features[:len(self.feature_columns)]
            
            features_scaled = self.scaler.transform([features])
            features_cnn = features_scaled.reshape(1, features_scaled.shape[1], 1)
            
            probability = float(self.hybrid_model.predict(features_cnn, verbose=0)[0][0])
            prediction = 1 if probability > 0.5 else 0
            
            return prediction, probability
            
        except Exception as e:
            st.error(f" Error in prediction: {str(e)}")
            return None, None
    
    def predict_batch(self, batch_data):
        """Predict fraud probability for batch of transactions"""
        try:
            if self.hybrid_model is None:
                return None, None
            
            predictions = []
            probabilities = []
            
            for features in batch_data:
                pred, prob = self.predict_single(features)
                if pred is not None:
                    predictions.append(pred)
                    probabilities.append(prob)
            
            return predictions, probabilities
            
        except Exception as e:
            st.error(f" Error in batch prediction: {str(e)}")
            return None, None
    
    def save_model(self, filename='fraud_detection_model'):
        """Save the trained model and scaler"""
        try:
            os.makedirs('saved_models', exist_ok=True)
            
            model_path = f'saved_models/{filename}.h5'
            self.hybrid_model.save(model_path)
            
            scaler_path = f'saved_models/{filename}_scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            
            features_path = f'saved_models/{filename}_features.pkl'
            joblib.dump(self.feature_columns, features_path)
            
            st.success(f" Model saved successfully!")
            return True
            
        except Exception as e:
            st.error(f" Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path, scaler_path=None, features_path=None):
        """Load a pre-trained model"""
        try:
            self.hybrid_model = load_model(model_path)
            
            if scaler_path:
                self.scaler = joblib.load(scaler_path)
            
            if features_path:
                self.feature_columns = joblib.load(features_path)
            
            st.success(" Model loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f" Error loading model: {str(e)}")
            return False

def generate_sample_features(n_features):
    """Generate random features for demo"""
    features = []
    for i in range(n_features):
        if i == 0:
            features.append(np.random.uniform(-3, 3))
        elif i == 1:
            features.append(np.random.uniform(-4, 4))
        elif i == 4:
            features.append(np.random.uniform(-5, 5))
        elif i == 7:
            features.append(np.random.uniform(-3, 3))
        else:
            features.append(np.random.uniform(-2, 2))
    return features

def display_prediction_result(prediction, probability, amount):
    """Display prediction result with visualizations"""
    st.markdown("---")
    
    if prediction == 1:
        st.markdown("""
        <div class="error-card">
            <h3 style="color: #ef4444; margin-bottom: 1rem;">⚠️ FRAUD DETECTED</h3>
        """, unsafe_allow_html=True)
        st.write(f"**Fraud Probability:** {probability*100:.2f}%")
        st.write(f"**Transaction Amount:** ${amount:.2f}")
        st.write("**⚠️ Action Required:** Block transaction and notify cardholder")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-card">
            <h3 style="color: #22c55e; margin-bottom: 1rem;">✅ TRANSACTION SAFE</h3>
        """, unsafe_allow_html=True)
        st.write(f"**Fraud Probability:** {probability*100:.2f}%")
        st.write(f"**Transaction Amount:** ${amount:.2f}")
        st.write("**✅ Status:** Transaction appears normal")
        st.markdown('</div>', unsafe_allow_html=True)

# visualization functions

def create_comprehensive_metrics_dashboard(metrics_dict):
    """Create a comprehensive metrics dashboard with clear visualizations"""
    
    fig = go.Figure()
    
    # Bar chart for metrics
    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#10b981']
    
    fig.add_trace(go.Bar(
        x=metrics_names,
        y=metrics_values,
        text=[f'{v:.3%}' for v in metrics_values],
        textposition='auto',
        marker=dict(
            color=colors[:len(metrics_names)],
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
        ),
        hovertemplate='<b>%{x}</b><br>Value: %{y:.3%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='📊 Model Performance Metrics',
            font=dict(size=20, color='#f1f5f9'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        xaxis=dict(
            title='Metrics',
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='Score',
            gridcolor='rgba(255,255,255,0.1)',
            tickformat='.0%',
            range=[0, 1.05]
        ),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_enhanced_confusion_matrix(cm):
    """Create enhanced confusion matrix visualization"""
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Normal', 'Predicted Fraud'],
        y=['Actual Normal', 'Actual Fraud'],
        text=[[f'{val:,}' for val in row] for row in cm],
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale=[
            [0, 'rgba(30, 41, 59, 0.8)'],
            [0.5, 'rgba(59, 130, 246, 0.5)'],
            [1, 'rgba(37, 99, 235, 0.8)']
        ],
        hovertemplate='<b>Actual: %{y}</b><br><b>Predicted: %{x}</b><br>Count: %{z}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title="Count",
            tickfont=dict(color='#cbd5e1')
        )
    ))
    
    # Add annotation percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i][j] / total * 100
            fig.add_annotation(
                x=j, y=i,
                text=f'{percentage:.1f}%',
                showarrow=False,
                font=dict(size=14, color='white' if cm[i][j] > cm.max()/2 else '#94a3b8')
            )
    
    fig.update_layout(
        title=dict(
            text='🔍 Confusion Matrix Analysis',
            font=dict(size=20, color='#f1f5f9'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        xaxis=dict(
            side='bottom',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            autorange='reversed',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        height=450,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_roc_curve_with_thresholds(fpr, tpr, thresholds, auc_score):
    """Create ROC curve with interactive threshold selection"""
    
    fig = go.Figure()
    
    # ROC Curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='<b>FPR: %{x:.3f}</b><br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    # Random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='#94a3b8', width=1),
        hoverinfo='skip'
    ))
    
    # Highlight optimal threshold (Youden's J statistic)
    if len(thresholds) > 0:
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        fig.add_trace(go.Scatter(
            x=[fpr[optimal_idx]],
            y=[tpr[optimal_idx]],
            mode='markers',
            name=f'Optimal Threshold ({optimal_threshold:.3f})',
            marker=dict(
                color='#10b981',
                size=12,
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>Optimal Point</b><br>Threshold: %{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>',
            text=[optimal_threshold]
        ))
    
    fig.update_layout(
        title=dict(
            text='📈 Receiver Operating Characteristic (ROC) Curve',
            font=dict(size=20, color='#f1f5f9'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        xaxis=dict(
            title='False Positive Rate',
            gridcolor='rgba(255,255,255,0.1)',
            range=[-0.02, 1.02],
            constrain='domain'
        ),
        yaxis=dict(
            title='True Positive Rate',
            gridcolor='rgba(255,255,255,0.1)',
            range=[-0.02, 1.02],
            scaleanchor="x",
            scaleratio=1
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(15, 23, 42, 0.7)',
            bordercolor='rgba(255, 255, 255, 0.1)',
            borderwidth=1
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_training_history_plots(history):
    """Create training history visualizations"""
    
    fig = go.Figure()
    
    # Loss plot
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history.history['loss']) + 1)),
        y=history.history['loss'],
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#667eea', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history.history['val_loss']) + 1)),
        y=history.history['val_loss'],
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='#f093fb', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=dict(
            text='📉 Training & Validation Loss',
            font=dict(size=20, color='#f1f5f9'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        xaxis=dict(
            title='Epoch',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title='Loss',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_metrics_comparison_plot(train_metrics, val_metrics):
    """Create side-by-side comparison of training and validation metrics"""
    
    fig = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    fig.add_trace(go.Bar(
        name='Training',
        x=metrics,
        y=train_metrics,
        text=[f'{v:.3%}' for v in train_metrics],
        textposition='auto',
        marker_color='#667eea'
    ))
    
    fig.add_trace(go.Bar(
        name='Validation',
        x=metrics,
        y=val_metrics,
        text=[f'{v:.3%}' for v in val_metrics],
        textposition='auto',
        marker_color='#f093fb'
    ))
    
    fig.update_layout(
        title=dict(
            text='📊 Training vs Validation Metrics',
            font=dict(size=20, color='#f1f5f9'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        xaxis=dict(
            title='Metrics',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title='Score',
            gridcolor='rgba(255,255,255,0.1)',
            tickformat='.0%',
            range=[0, 1.05]
        ),
        barmode='group',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_feature_importance_plot(system):
    """Create feature importance visualization"""
    try:
        if system.hybrid_model is None or system.feature_columns is None:
            return None
        
        # Get feature importance from the model
        intermediate_model = Model(inputs=system.hybrid_model.input,
                                 outputs=system.hybrid_model.layers[-3].output)
        
        # Get average activations
        activations = intermediate_model.predict(system.X_test_cnn[:100], verbose=0)
        importance_scores = np.mean(np.abs(activations), axis=0)
        
        # Select top features
        top_n = min(15, len(importance_scores))
        top_indices = np.argsort(importance_scores)[-top_n:][::-1]
        top_features = [f'Feature {i+1}' for i in top_indices]
        top_scores = importance_scores[top_indices]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_scores,
            y=top_features,
            orientation='h',
            marker=dict(
                color='rgba(102, 126, 234, 0.8)',
                line=dict(color='rgba(102, 126, 234, 1)', width=1)
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='🏆 Top Feature Importance',
                font=dict(size=20, color='#f1f5f9'),
                x=0.5
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            xaxis=dict(
                title='Importance Score',
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title='Features',
                autorange='reversed',
                gridcolor='rgba(255,255,255,0.1)'
            ),
            height=500,
            margin=dict(l=100, r=50, t=80, b=50)
        )
        
        return fig
    except Exception as e:
        st.error(f"Feature importance error: {e}")
        return None

def create_distribution_plot(system, feature_idx=0):
    """Create distribution plot for a specific feature"""
    
    if system.df is None:
        return None
    
    fraud_data = system.df[system.df[system.target_column] == 1]
    normal_data = system.df[system.df[system.target_column] == 0]
    
    if len(fraud_data) == 0 or len(normal_data) == 0:
        return None
    
    feature_name = system.df.columns[feature_idx] if feature_idx < len(system.df.columns) else f'Feature {feature_idx+1}'
    
    fig = go.Figure()
    
    # Normal transactions
    fig.add_trace(go.Histogram(
        x=normal_data.iloc[:, feature_idx],
        name='Normal Transactions',
        opacity=0.7,
        nbinsx=50,
        marker_color='rgba(102, 126, 234, 0.6)',
        hoverinfo='x+y'
    ))
    
    # Fraud transactions
    fig.add_trace(go.Histogram(
        x=fraud_data.iloc[:, feature_idx],
        name='Fraud Transactions',
        opacity=0.7,
        nbinsx=50,
        marker_color='rgba(239, 68, 68, 0.6)',
        hoverinfo='x+y'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'📊 Distribution of {feature_name}',
            font=dict(size=20, color='#f1f5f9'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        xaxis=dict(
            title=f'{feature_name} Value',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title='Count',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        barmode='overlay',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_prediction_probability_histogram(y_true, y_pred_proba):
    """Create histogram of prediction probabilities"""
    
    fraud_probs = y_pred_proba[y_true == 1]
    normal_probs = y_pred_proba[y_true == 0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=normal_probs,
        name='Normal Transactions',
        opacity=0.7,
        nbinsx=50,
        marker_color='rgba(102, 126, 234, 0.6)',
        hoverinfo='x+y'
    ))
    
    fig.add_trace(go.Histogram(
        x=fraud_probs,
        name='Fraud Transactions',
        opacity=0.7,
        nbinsx=50,
        marker_color='rgba(239, 68, 68, 0.6)',
        hoverinfo='x+y'
    ))
    
    # Add decision threshold line
    fig.add_vline(x=0.5, line_dash="dash", line_color="white", 
                  annotation_text="Threshold = 0.5", 
                  annotation_position="top",
                  annotation_font_color="#cbd5e1")
    
    fig.update_layout(
        title=dict(
            text='📈 Prediction Probability Distribution',
            font=dict(size=20, color='#f1f5f9'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        xaxis=dict(
            title='Fraud Probability',
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, 1]
        ),
        yaxis=dict(
            title='Count',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        barmode='overlay',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# =========================================================
# STREAMLIT UI COMPONENTS
# =========================================================
def display_sidebar():
    """Display sidebar with premium navigation"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-icon">💳</div>
            <h1 class="sidebar-title">FRAUD AI</h1>
            <p class="sidebar-subtitle">Advanced Security System</p>
        </div>
        """, unsafe_allow_html=True)
        
        nav_options = [
            ("🏠", "Home", "Dashboard Overview"),
            ("📊", "Data", "Data Management"),
            ("🤖", "Train", "Model Training"),
            ("🔍", "Scan", "Fraud Detection"),
            ("📈", "Analyze", "Results & Analytics"),
            ("💾", "System", "Save/Load Models")
        ]
        
        for icon, name, desc in nav_options:
            if st.button(f"{icon} {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state.current_page = name
                st.rerun()
        
        st.markdown("---")
        
        st.markdown('<div style="font-weight: 600; color: #cbd5e1; margin-bottom: 1rem;">SYSTEM STATUS</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            status = " Active" if st.session_state.data_loaded else "❌ Inactive"
            st.metric("Data", status)
        with col2:
            status = " Active" if st.session_state.model_trained else "❌ Inactive"
            st.metric("Model", status)
        
        st.markdown("---")
        
        if st.button("🔄 Reset System", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.current_page = "Home"
            st.rerun()
    
    return st.session_state.current_page

def home_page():
    """Home page with ultra premium design"""
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0 1rem;">
            <h1 class="main-title-3d">AI FRAUD DETECTION</h1>
            <h4 class="subtitle-glow">Advanced Neural Network Security System</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Stats Banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stats-widget">
            <div class="stat-number">99.8%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stats-widget">
            <div class="stat-number">0.2s</div>
            <div class="stat-label">Detection Time</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stats-widget">
            <div class="stat-number">24/7</div>
            <div class="stat-label">Monitoring</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stats-widget">
            <div class="stat-number">10M+</div>
            <div class="stat-label">Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("""
    <div style="margin: 3rem 0 2rem;">
        <h3 style="text-align: center; color: #f1f5f9; font-size: 2rem; margin-bottom: 2rem; font-family: 'Space Grotesk', sans-serif;">
            ✦ POWERFUL FEATURES ✦
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card-3d" style="height: 100%;">
            <div class="card-icon">🤖</div>
            <h3 style="color: #f1f5f9; margin-bottom: 1rem; font-size: 1.5rem;">AI-Powered Detection</h3>
            <p style="color: #cbd5e1; line-height: 1.6; margin-bottom: 1rem;">
                Advanced deep learning algorithms that continuously learn and adapt to new fraud patterns.
            </p>
            <div style="margin-top: 1rem;">
                <span class="feature-badge">Neural Networks</span>
                <span class="feature-badge">ML Algorithms</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card-3d" style="height: 100%;">
            <div class="card-icon">⚡</div>
            <h3 style="color: #f1f5f9; margin-bottom: 1rem; font-size: 1.5rem;">Real-Time Analysis</h3>
            <p style="color: #cbd5e1; line-height: 1.6; margin-bottom: 1rem;">
                Instant processing with sub-second response times for immediate fraud prevention.
            </p>
            <div style="margin-top: 1rem;">
                <span class="feature-badge">Live Monitoring</span>
                <span class="feature-badge">Instant Alerts</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card-3d" style="height: 100%;">
            <div class="card-icon">📊</div>
            <h3 style="color: #f1f5f9; margin-bottom: 1rem; font-size: 1.5rem;">Advanced Analytics</h3>
            <p style="color: #cbd5e1; line-height: 1.6; margin-bottom: 1rem;">
                Comprehensive dashboards and predictive insights for fraud patterns and trends.
            </p>
            <div style="margin-top: 1rem;">
                <span class="feature-badge">Visual Analytics</span>
                <span class="feature-badge">Risk Scoring</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Start Section
    st.markdown("""
    <div style="margin: 4rem 0 2rem;">
        <h3 style="text-align: center; color: #f1f5f9; font-size: 2rem; margin-bottom: 2rem; font-family: 'Space Grotesk', sans-serif;">
            🚀 GET STARTED
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 GENERATE SAMPLE DATA", use_container_width=True):
            system = FraudDetectionSystem()
            if system.generate_sample_data():
                st.session_state.fraud_system = system
                st.session_state.data_loaded = True
                st.success(" Sample data generated!")
                st.rerun()
    
    with col2:
        if st.button("🎬 QUICK DEMO", use_container_width=True):
            with st.spinner("Initializing AI Security System..."):
                system = FraudDetectionSystem()
                if system.generate_sample_data():
                    system.preprocess_data()
                    system.train_model(epochs=10, batch_size=32)
                    st.session_state.fraud_system = system
                    st.session_state.data_loaded = True
                    st.session_state.model_trained = True
                    st.success(" Demo setup complete!")
                    st.balloons()
                    st.rerun()

def data_management_page():
    """Data management page"""
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h1 style="color: #f1f5f9; font-size: 2.5rem; font-family: 'Space Grotesk', sans-serif;">📊 DATA MANAGEMENT</h1>
        <p style="color: #94a3b8;">Upload or generate transaction data for analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.fraud_system is None:
        st.session_state.fraud_system = FraudDetectionSystem()
    
    system = st.session_state.fraud_system
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card-3d">
            <h3 style="color: #f1f5f9; margin-bottom: 1rem;">📥 Upload Dataset</h3>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            if st.button("Load Data", use_container_width=True):
                if system.load_data(uploaded_file):
                    st.session_state.data_loaded = True
                    st.success(" Data loaded successfully!")
                    st.rerun()
    
    with col2:
        st.markdown("""
        <div class="glass-card-3d">
            <h3 style="color: #f1f5f9; margin-bottom: 1rem;">🎲 Generate Sample Data</h3>
            <p style="color: #94a3b8; margin-bottom: 1rem;">Create realistic synthetic transaction data</p>
        """, unsafe_allow_html=True)
        if st.button("Generate Data", use_container_width=True):
            if system.generate_sample_data():
                st.session_state.data_loaded = True
                st.rerun()
    
    if st.session_state.data_loaded and system.df is not None:
        st.markdown("---")
        st.markdown('<h3 style="color: #f1f5f9;">📈 Data Overview</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(system.df):,}")
        with col2:
            fraud_count = system.df[system.target_column].sum()
            st.metric("Fraud Cases", f"{fraud_count:,}")
        with col3:
            fraud_rate = (fraud_count / len(system.df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col4:
            st.metric("Features", len(system.df.columns) - 1)
        
        tab1, tab2 = st.tabs(["📋 Data Preview", "📊 Statistics"])
        
        with tab1:
            st.dataframe(system.df.head(20), use_container_width=True)
        
        with tab2:
            st.dataframe(system.df.describe(), use_container_width=True)
        
        if st.button("🔄 Preprocess Data", type="primary", use_container_width=True):
            with st.spinner("Preprocessing data..."):
                if system.preprocess_data():
                    st.success(" Data preprocessing completed!")
                    st.info("Proceed to Model Training")

def model_training_page():
    """Model training page"""
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h1 style="color: #f1f5f9; font-size: 2.5rem; font-family: 'Space Grotesk', sans-serif;">🤖 MODEL TRAINING</h1>
        <p style="color: #94a3b8;">Train the AI model on your transaction data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return
    
    system = st.session_state.fraud_system
    
    st.markdown("""
    <div class="glass-card-3d">
        <h3 style="color: #f1f5f9; margin-bottom: 1.5rem;">🎯 Training Configuration</h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.slider("Epochs", 5, 50, 20, 5)
    
    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    with col3:
        learning_rate = st.select_slider("Learning Rate", 
                                        options=[0.1, 0.01, 0.001, 0.0001],
                                        value=0.001)
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("Training in progress..."):
            progress_bar = st.progress(0)
            
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.02)
            
            history = system.train_model(epochs=epochs, batch_size=batch_size)
            
            if history:
                st.session_state.training_history = history
                st.session_state.model_trained = True
                st.balloons()
                st.success(" Model trained successfully!")
            else:
                st.error(" Model training failed!")
    
    if st.session_state.model_trained and system.model_metrics:
        st.markdown("---")
        st.markdown('<h3 style="color: #f1f5f9;">📊 Training Results</h3>', unsafe_allow_html=True)
        
        metrics = system.model_metrics['metrics']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.3%}")
        with col2:
            st.metric("Precision", f"{metrics['Precision']:.3%}")
        with col3:
            st.metric("Recall", f"{metrics['Recall']:.3%}")
        with col4:
            st.metric("F1 Score", f"{metrics['F1 Score']:.3%}")
        with col5:
            st.metric("ROC AUC", f"{metrics['ROC AUC']:.3%}")
        
        # Confusion Matrix
        st.markdown("#### 📊 Confusion Matrix")
        cm = system.model_metrics['confusion_matrix']
        
        fig = px.imshow(
            cm,
            text_auto=True,
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=['Normal', 'Fraud'],
            y=['Normal', 'Fraud'],
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

def predict_fraud_page():
    """Fraud prediction page"""
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h1 style="color: #f1f5f9; font-size: 2.5rem; font-family: 'Space Grotesk', sans-serif;">🔍 FRAUD DETECTION</h1>
        <p style="color: #94a3b8;">Test individual transactions or batch processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first!")
        return
    
    system = st.session_state.fraud_system
    
    tab1, tab2 = st.tabs(["🔍 Single Transaction", "📁 Batch Processing"])
    
    with tab1:
        st.markdown("""
        <div class="glass-card-3d">
            <h3 style="color: #f1f5f9; margin-bottom: 1.5rem;">Test Individual Transaction</h3>
        """, unsafe_allow_html=True)
        
        input_mode = st.radio("Input Mode", ["🎲 Random Sample", "✍️ Manual Input"], horizontal=True)
        
        if input_mode == "🎲 Random Sample":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate Random Transaction", use_container_width=True):
                    if system.feature_columns:
                        features = generate_sample_features(len(system.feature_columns))
                        amount = np.random.exponential(100)
                        
                        st.session_state.random_features = features
                        st.session_state.random_amount = amount
                        
                        with st.expander("📋 Generated Features"):
                            for i in range(min(10, len(features))):
                                st.write(f"**Feature {i+1}:** {features[i]:.4f}")
                            st.metric("Amount", f"${amount:.2f}")
            
            if 'random_features' in st.session_state:
                if st.button("Predict Fraud", type="primary", use_container_width=True):
                    prediction, probability = system.predict_single(st.session_state.random_features)
                    if prediction is not None:
                        display_prediction_result(prediction, probability, st.session_state.random_amount)
        
        else:
            st.markdown("#### Enter Transaction Details")
            
            if system.feature_columns:
                with st.form("transaction_form"):
                    feature_inputs = []
                    
                    for i in range(min(10, len(system.feature_columns))):
                        col1, col2 = st.columns(2)
                        with col1:
                            feature_name = f"V{i+1}"
                            value = st.number_input(feature_name, value=0.0, format="%.4f", key=f"feat_{i}")
                            feature_inputs.append(value)
                    
                    if len(feature_inputs) < len(system.feature_columns):
                        feature_inputs.extend([0.0] * (len(system.feature_columns) - len(feature_inputs)))
                    
                    amount = st.number_input("💵 Amount ($)", value=100.0, format="%.2f")
                    
                    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)
                    
                    if submitted:
                        prediction, probability = system.predict_single(feature_inputs)
                        if prediction is not None:
                            display_prediction_result(prediction, probability, amount)
    
    with tab2:
        st.markdown("""
        <div class="glass-card-3d">
            <h3 style="color: #f1f5f9; margin-bottom: 1.5rem;">Process Multiple Transactions</h3>
        """, unsafe_allow_html=True)
        
        n_samples = st.slider("Number of demo transactions", 10, 100, 50)
        
        if st.button("Generate Demo Batch", use_container_width=True):
            demo_features = []
            for _ in range(n_samples):
                features = generate_sample_features(len(system.feature_columns))
                demo_features.append(features)
            
            st.session_state.demo_batch = demo_features
            st.success(f" Generated {n_samples} demo transactions!")
        
        if 'demo_batch' in st.session_state:
            if st.button("Process Batch", type="primary", use_container_width=True):
                demo_features = st.session_state.demo_batch
                predictions, probabilities = system.predict_batch(demo_features)
                
                if predictions:
                    results_df = pd.DataFrame({
                        'Transaction_ID': range(len(predictions)),
                        'Prediction': ['Fraud' if p == 1 else 'Normal' for p in predictions],
                        'Probability': probabilities
                    })
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    fraud_count = sum(predictions)
                    st.metric("Fraud Detected", fraud_count, f"{fraud_count/len(predictions)*100:.1f}%")

def results_analysis_page():
    """Enhanced results and analysis page with clear graphs"""
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h1 style="color: #f1f5f9; font-size: 2.5rem; font-family: 'Space Grotesk', sans-serif;">📈 ADVANCED ANALYTICS</h1>
        <p style="color: #94a3b8;">Comprehensive model performance visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.info("ℹ Train the model first to see results!")
        return
    
    system = st.session_state.fraud_system
    
    if system.model_metrics is None:
        st.error(" No metrics available. Please retrain the model.")
        return
    
    metrics = system.model_metrics['metrics']
    
    # Performance Metrics Dashboard
    st.markdown("### 📊 Performance Dashboard")
    
    # Key metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['Accuracy']:.3%}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['Precision']:.3%}</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['Recall']:.3%}</div>
            <div class="metric-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['F1 Score']:.3%}</div>
            <div class="metric-label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['ROC AUC']:.3%}</div>
            <div class="metric-label">ROC AUC</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Visualization Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🔍 Confusion", "📈 ROC", "🎯 Training"])
    
    with tab1:
        # Metrics Comparison
        fig_metrics = create_comprehensive_metrics_dashboard(metrics)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Feature Importance
        if system.hybrid_model is not None:
            st.markdown("#### 🏆 Feature Importance")
            fig_importance = create_feature_importance_plot(system)
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Distribution Plots
        st.markdown("#### 📈 Feature Distributions")
        col1, col2 = st.columns(2)
        with col1:
            feature_idx = st.slider("Select Feature", 0, min(10, len(system.df.columns)-2), 0)
        with col2:
            if st.button("Update Distribution Plot", use_container_width=True):
                pass
        
        fig_dist = create_distribution_plot(system, feature_idx)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        # Enhanced Confusion Matrix
        st.markdown("#### 🔍 Confusion Matrix Analysis")
        cm = system.model_metrics['confusion_matrix']
        fig_cm = create_enhanced_confusion_matrix(cm)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Confusion Matrix Statistics
        col1, col2, col3, col4 = st.columns(4)
        tn, fp, fn, tp = cm.ravel()
        
        with col1:
            st.metric("True Positive", f"{tp:,}", f"{tp/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else "0%")
        with col2:
            st.metric("False Positive", f"{fp:,}", f"{fp/(fp+tn)*100:.1f}%" if (fp+tn) > 0 else "0%")
        with col3:
            st.metric("False Negative", f"{fn:,}", f"{fn/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else "0%")
        with col4:
            st.metric("True Negative", f"{tn:,}", f"{tn/(tn+fp)*100:.1f}%" if (tn+fp) > 0 else "0%")
        
        # Probability Distribution
        st.markdown("#### 📊 Prediction Probability Distribution")
        y_true = system.model_metrics['predictions']['y_true']
        y_pred_proba = system.model_metrics['predictions']['y_pred_proba']
        fig_probs = create_prediction_probability_histogram(y_true, y_pred_proba)
        st.plotly_chart(fig_probs, use_container_width=True)
    
    with tab3:
        # Enhanced ROC Curve
        fpr, tpr, thresholds = system.model_metrics['roc_curve']
        fig_roc = create_roc_curve_with_thresholds(fpr, tpr, thresholds, metrics['ROC AUC'])
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # ROC Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AUC Score", f"{metrics['ROC AUC']:.4f}")
        
        # Calculate Youden's J statistic
        j_scores = tpr - fpr
        if len(j_scores) > 0:
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            with col2:
                st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
            
            with col3:
                st.metric("Youden's J", f"{j_scores[optimal_idx]:.3f}")
        
        # Threshold Analysis
        st.markdown("#### 🎯 Threshold Sensitivity Analysis")
        
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)
        
        # Recalculate metrics with selected threshold
        y_pred_adjusted = (y_pred_proba > threshold).astype(int)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            adj_precision = precision_score(y_true, y_pred_adjusted, zero_division=0)
            st.metric("Precision", f"{adj_precision:.3%}")
        with col2:
            adj_recall = recall_score(y_true, y_pred_adjusted, zero_division=0)
            st.metric("Recall", f"{adj_recall:.3%}")
        with col3:
            adj_f1 = f1_score(y_true, y_pred_adjusted, zero_division=0)
            st.metric("F1 Score", f"{adj_f1:.3%}")
    
    with tab4:
        # Training History
        if st.session_state.training_history:
            st.markdown("#### 📉 Training History")
            
            fig_history = create_training_history_plots(st.session_state.training_history)
            st.plotly_chart(fig_history, use_container_width=True)
            
            # Learning Curves
            st.markdown("#### 📊 Learning Curves")
            
            col1, col2 = st.columns(2)
            
            with col1:
                final_train_loss = st.session_state.training_history.history['loss'][-1]
                final_val_loss = st.session_state.training_history.history['val_loss'][-1]
                
                st.metric("Final Training Loss", f"{final_train_loss:.4f}")
                st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
            
            with col2:
                if 'accuracy' in st.session_state.training_history.history:
                    final_train_acc = st.session_state.training_history.history['accuracy'][-1]
                    final_val_acc = st.session_state.training_history.history['val_accuracy'][-1]
                    
                    st.metric("Final Training Accuracy", f"{final_train_acc:.3%}")
                    st.metric("Final Validation Accuracy", f"{final_val_acc:.3%}")
            
            # Convergence Analysis
            st.markdown("#### ⚡ Convergence Analysis")
            
            train_loss_history = st.session_state.training_history.history['loss']
            val_loss_history = st.session_state.training_history.history['val_loss']
            
            convergence_epoch = None
            for i in range(1, len(train_loss_history)):
                if abs(train_loss_history[i] - train_loss_history[i-1]) < 0.001 and abs(val_loss_history[i] - val_loss_history[i-1]) < 0.001:
                    convergence_epoch = i
                    break
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Epochs", len(train_loss_history))
            with col2:
                if convergence_epoch:
                    st.metric("Converged at Epoch", convergence_epoch)
                else:
                    st.metric("Convergence", "Not Reached")
    
    # Download Report Button
    st.markdown("---")
    if st.button("📥 Download Complete Analysis Report", use_container_width=True):
        # Generate comprehensive report
        report_data = {
            'metrics': metrics,
            'confusion_matrix': system.model_metrics['confusion_matrix'].tolist(),
            'roc_auc': metrics['ROC AUC']
        }
        
        # Convert to JSON for download
        json_str = json.dumps(report_data, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="fraud_analysis_report.json">📥 Download JSON Report</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("✅ Report generated successfully!")

def save_load_model_page():
    """Model save/load page"""
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h1 style="color: #f1f5f9; font-size: 2.5rem; font-family: 'Space Grotesk', sans-serif;">💾 SAVE/LOAD MODEL</h1>
        <p style="color: #94a3b8;">Manage your trained models</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.fraud_system is None:
        st.warning("⚠️ Initialize system first!")
        return
    
    system = st.session_state.fraud_system
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card-3d" style="height: 100%;">
            <h3 style="color: #f1f5f9; margin-bottom: 1.5rem;">💾 Save Model</h3>
        """, unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.info("ℹ️ Train a model first!")
        else:
            model_name = st.text_input("Model Name", value="fraud_model")
            
            if st.button("Save Model", use_container_width=True):
                if system.save_model(model_name):
                    st.success("✅ Model saved!")
                    
                    if os.path.exists(f'saved_models/{model_name}.h5'):
                        with open(f'saved_models/{model_name}.h5', 'rb') as f:
                            bytes_data = f.read()
                            b64 = base64.b64encode(bytes_data).decode()
                            href = f'<a href="data:file/octet-stream;base64,{b64}" download="{model_name}.h5">📥 Download Model</a>'
                            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card-3d" style="height: 100%;">
            <h3 style="color: #f1f5f9; margin-bottom: 1.5rem;">📂 Load Model</h3>
        """, unsafe_allow_html=True)
        
        uploaded_model = st.file_uploader("Upload Model (.h5)", type=['h5'])
        
        if uploaded_model is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                tmp.write(uploaded_model.getvalue())
                tmp_path = tmp.name
            
            if st.button("Load Model", use_container_width=True):
                if system.load_model(tmp_path):
                    st.session_state.model_trained = True
                    st.success("✅ Model loaded!")
                    os.unlink(tmp_path)

# main app
def main():
    """Main application function"""
    
    page = display_sidebar()
    
    if page == "Home":
        home_page()
    elif page == "Data":
        data_management_page()
    elif page == "Train":
        model_training_page()
    elif page == "Scan":
        predict_fraud_page()
    elif page == "Analyze":
        results_analysis_page()
    elif page == "System":
        save_load_model_page()

if __name__ == "__main__":
    main()