# =========================================================
# CREDIT CARD FRAUD DETECTION SYSTEM
# Hybrid CNN + Logistic Regression Model
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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, 
    Input, BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E3A8A;
        padding: 20px;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .fraud-alert {
        background: linear-gradient(135deg, #FFE5E5 0%, #FFCCCC 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 8px solid #FF4B4B;
        margin: 20px 0;
        box-shadow: 0 6px 12px rgba(255, 75, 75, 0.2);
        animation: pulse 2s infinite;
    }
    .normal-transaction {
        background: linear-gradient(135deg, #E5FFE5 0%, #CCFFCC 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 8px solid #00CC00;
        margin: 20px 0;
        box-shadow: 0 6px 12px rgba(0, 204, 0, 0.2);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stButton button {
        background: linear-gradient(135deg, #3B82F6 0%, #1E3A8A 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-green {
        background-color: #00CC00;
    }
    .status-red {
        background-color: #FF4B4B;
    }
    .status-yellow {
        background-color: #FFC107;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    .tab-content {
        padding: 20px;
        background: white;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
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

# =========================================================
# DATASET GENERATOR
# =========================================================
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
    # Fraud transactions have different statistical properties
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
    df['Time'] = np.arange(n_samples)  # Sequential time
    df['Class'] = fraud_mask.astype(int)
    
    # Add some noise and missing values
    for col in np.random.choice(columns[:10], size=3, replace=False):
        mask = np.random.rand(n_samples) < 0.01  # 1% missing
        df.loc[mask, col] = np.nan
    
    return df

# =========================================================
# FRAUD DETECTION SYSTEM CLASS
# =========================================================
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
            if hasattr(file_path, 'read'):  # If it's a file-like object
                self.df = pd.read_csv(file_path)
            else:  # If it's a file path string
                self.df = pd.read_csv(file_path)
            
            # Auto-detect target column
            target_candidates = ['Class', 'Fraud', 'is_fraud', 'isFraud', 'fraud', 'label', 'target']
            for col in target_candidates:
                if col in self.df.columns:
                    self.target_column = col
                    break
            
            # Check if target column is found
            if self.target_column not in self.df.columns:
                st.error(f"❌ Could not find target column. Expected one of: {', '.join(target_candidates)}")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def generate_sample_data(self):
        """Generate realistic sample data for demonstration"""
        try:
            self.df = generate_realistic_dataset(10000)
            self.target_column = 'Class'
            
            fraud_count = self.df['Class'].sum()
            total_count = len(self.df)
            fraud_rate = (fraud_count / total_count) * 100
            
            st.success(f"✅ Generated {total_count:,} sample transactions")
            st.success(f"📊 {fraud_count:,} fraud cases ({fraud_rate:.2f}%)")
            return True
            
        except Exception as e:
            st.error(f"❌ Error generating sample data: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocess the data for model training"""
        try:
            if self.df is None:
                st.error("❌ No data loaded!")
                return False
            
            # Handle missing values
            df_clean = self.df.copy()
            
            # Fill missing values with median for numerical columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
            
            # Prepare features and target
            X = df_clean.drop(self.target_column, axis=1)
            y = df_clean[self.target_column]
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            # Handle class imbalance
            class_counts = Counter(self.y_train)
            fraud_count = class_counts.get(1, 0)
            
            if fraud_count >= 5:
                # Use SMOTE for balancing
                smote = SMOTE(random_state=42, k_neighbors=min(5, fraud_count - 1))
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                st.success(f"✅ Applied SMOTE: Balanced to {Counter(self.y_train)}")
            else:
                # Use RandomUnderSampler if not enough fraud samples
                rus = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
                self.X_train, self.y_train = rus.fit_resample(self.X_train, self.y_train)
                st.success(f"✅ Applied RandomUnderSampler: Balanced to {Counter(self.y_train)}")
            
            # Reshape for CNN
            self.X_train_cnn = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
            self.X_test_cnn = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
            
            st.success("✅ Data preprocessing completed!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error in preprocessing: {str(e)}")
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
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
            )
            
            self.hybrid_model = model
            return model
            
        except Exception as e:
            st.error(f"❌ Error building model: {str(e)}")
            return None
    
    def train_model(self, epochs=30, batch_size=64):
        """Train the hybrid model"""
        try:
            # Check if data is preprocessed
            if not hasattr(self, 'X_train_cnn') or self.X_train_cnn is None:
                if not self.preprocess_data():
                    return None
            
            # Build model
            if self.hybrid_model is None:
                self.build_hybrid_model()
            
            # Callbacks
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
            
            # Train model
            self.history = self.hybrid_model.fit(
                self.X_train_cnn, self.y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            self.evaluate_model()
            
            return self.history
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            return None
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        try:
            if self.hybrid_model is None:
                return None
            
            # Get predictions
            y_pred_proba = self.hybrid_model.predict(self.X_test_cnn, verbose=0).ravel()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred, zero_division=0),
                'Recall': recall_score(self.y_test, y_pred, zero_division=0),
                'F1 Score': f1_score(self.y_test, y_pred, zero_division=0),
                'ROC AUC': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # ROC curve
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
            st.error(f"❌ Error evaluating model: {str(e)}")
            return None
    
    def predict_single(self, features):
        """Predict fraud probability for a single transaction"""
        try:
            if self.hybrid_model is None or self.scaler is None:
                return None, None
            
            # Ensure correct number of features
            if len(features) != len(self.feature_columns):
                # Pad or truncate if needed
                if len(features) < len(self.feature_columns):
                    features = list(features) + [0] * (len(self.feature_columns) - len(features))
                else:
                    features = features[:len(self.feature_columns)]
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Reshape for CNN
            features_cnn = features_scaled.reshape(1, features_scaled.shape[1], 1)
            
            # Make prediction
            probability = float(self.hybrid_model.predict(features_cnn, verbose=0)[0][0])
            prediction = 1 if probability > 0.5 else 0
            
            return prediction, probability
            
        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")
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
            st.error(f"❌ Error in batch prediction: {str(e)}")
            return None, None
    
    def save_model(self, filename='fraud_detection_model'):
        """Save the trained model and scaler"""
        try:
            # Create directory if it doesn't exist
            os.makedirs('saved_models', exist_ok=True)
            
            # Save Keras model
            model_path = f'saved_models/{filename}.h5'
            self.hybrid_model.save(model_path)
            
            # Save scaler
            scaler_path = f'saved_models/{filename}_scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            
            # Save feature columns
            features_path = f'saved_models/{filename}_features.pkl'
            joblib.dump(self.feature_columns, features_path)
            
            st.success(f"✅ Model saved successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path, scaler_path=None, features_path=None):
        """Load a pre-trained model"""
        try:
            # Load Keras model
            self.hybrid_model = load_model(model_path)
            
            # Load scaler if provided
            if scaler_path:
                self.scaler = joblib.load(scaler_path)
            
            # Load feature columns if provided
            if features_path:
                self.feature_columns = joblib.load(features_path)
            
            st.success("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def generate_sample_features(n_features):
    """Generate random features for demo"""
    features = []
    for i in range(n_features):
        if i == 0:  # V1
            features.append(np.random.uniform(-3, 3))
        elif i == 1:  # V2
            features.append(np.random.uniform(-4, 4))
        elif i == 4:  # V5
            features.append(np.random.uniform(-5, 5))
        elif i == 7:  # V8
            features.append(np.random.uniform(-3, 3))
        else:
            features.append(np.random.uniform(-2, 2))
    return features

def create_download_link(data, filename, text):
    """Create a download link for data"""
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False)
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display: inline-block; padding: 10px 20px; background: linear-gradient(135deg, #3B82F6 0%, #1E3A8A 100%); color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">{text}</a>'
    return href

def display_prediction_result(prediction, probability, amount):
    """Display prediction result with visualizations"""
    st.markdown("---")
    
    if prediction == 1:
        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
        st.error("### ⚠️ FRAUD DETECTED")
        st.write(f"**Fraud Probability:** {probability*100:.2f}%")
        st.write(f"**Transaction Amount:** ${amount:.2f}")
        st.write("**⚠️ Action Required:** Block transaction and notify cardholder")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="normal-transaction">', unsafe_allow_html=True)
        st.success("### ✅ TRANSACTION SAFE")
        st.write(f"**Fraud Probability:** {probability*100:.2f}%")
        st.write(f"**Transaction Amount:** ${amount:.2f}")
        st.write("**✅ Status:** Transaction appears normal")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# STREAMLIT UI COMPONENTS
# =========================================================
def display_sidebar():
    """Display sidebar with navigation and status"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 48px;">💳</span>
            <h3 style="margin: 5px 0;">Fraud Detection</h3>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigate to:",
            ["🏠 Home", "📊 Data Management", "🤖 Model Training", 
             "🔍 Predict Fraud", "📈 Results & Analysis", "💾 Save/Load Model"]
        )
        
        st.markdown("---")
        st.subheader("System Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅" if st.session_state.data_loaded else "❌"
            st.metric("Data", status, "Loaded" if st.session_state.data_loaded else "Required")
        with col2:
            status = "✅" if st.session_state.model_trained else "❌"
            st.metric("Model", status, "Trained" if st.session_state.model_trained else "Not Trained")
        with col3:
            st.metric("System", "✅", "Ready")
        
        st.markdown("---")
        
        if st.button("🔄 Reset System", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    return page

def home_page():
    """Home page with overview"""
    st.markdown("<h1 class='main-title'>💳 Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 class='sub-title'>Hybrid CNN + Dense Neural Network Model</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 High Accuracy</h3>
            <p>Advanced deep learning model for fraud detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time Detection</h3>
            <p>Instant fraud detection for transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Sample Data", use_container_width=True):
            system = FraudDetectionSystem()
            if system.generate_sample_data():
                st.session_state.fraud_system = system
                st.session_state.data_loaded = True
                st.success("✅ Sample data generated successfully!")
                st.rerun()
    
    with col2:
        if st.button("🎬 Quick Demo", use_container_width=True):
            with st.spinner("Setting up demo..."):
                system = FraudDetectionSystem()
                if system.generate_sample_data():
                    system.preprocess_data()
                    system.train_model(epochs=10, batch_size=32)
                    st.session_state.fraud_system = system
                    st.session_state.data_loaded = True
                    st.session_state.model_trained = True
                    st.success("✅ Demo setup complete!")
                    st.rerun()

def data_management_page():
    """Data management page"""
    st.subheader("📊 Data Management")
    
    if st.session_state.fraud_system is None:
        st.session_state.fraud_system = FraudDetectionSystem()
    
    system = st.session_state.fraud_system
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            if st.button("📂 Load Uploaded Data", use_container_width=True):
                if system.load_data(uploaded_file):
                    st.session_state.data_loaded = True
                    st.success("✅ Data loaded successfully!")
                    st.rerun()
    
    with col2:
        st.markdown("### 🎲 Generate Sample Data")
        if st.button("🔄 Generate Sample Data", use_container_width=True):
            if system.generate_sample_data():
                st.session_state.data_loaded = True
                st.success("✅ Sample data generated!")
                st.rerun()
    
    if st.session_state.data_loaded and system.df is not None:
        st.markdown("---")
        st.subheader("📈 Data Overview")
        
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
            with st.spinner("Preprocessing..."):
                if system.preprocess_data():
                    st.success("✅ Data preprocessing completed!")
                    st.info("Proceed to Model Training")

def model_training_page():
    """Model training page"""
    st.subheader("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return
    
    system = st.session_state.fraud_system
    
    st.markdown("### 🎯 Training Configuration")
    
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
                st.success("🎉 Model trained successfully!")
            else:
                st.error("❌ Model training failed!")
    
    if st.session_state.model_trained and system.model_metrics:
        st.markdown("---")
        st.subheader("📊 Training Results")
        
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
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

def predict_fraud_page():
    """Fraud prediction page"""
    st.subheader("🔍 Fraud Detection")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first!")
        return
    
    system = st.session_state.fraud_system
    
    tab1, tab2 = st.tabs(["🔍 Single Transaction", "📁 Batch Processing"])
    
    with tab1:
        st.markdown("### Test Individual Transaction")
        
        input_mode = st.radio("Input Mode", ["🎲 Random Sample", "✍️ Manual Input"], horizontal=True)
        
        if input_mode == "🎲 Random Sample":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎲 Generate Random Transaction", use_container_width=True):
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
                if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
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
                    
                    submitted = st.form_submit_button("🔍 Predict", type="primary", use_container_width=True)
                    
                    if submitted:
                        prediction, probability = system.predict_single(feature_inputs)
                        if prediction is not None:
                            display_prediction_result(prediction, probability, amount)
    
    with tab2:
        st.markdown("### Process Multiple Transactions")
        
        n_samples = st.slider("Number of demo transactions", 10, 100, 50)
        
        if st.button("📊 Generate Demo Batch", use_container_width=True):
            demo_features = []
            for _ in range(n_samples):
                features = generate_sample_features(len(system.feature_columns))
                demo_features.append(features)
            
            st.session_state.demo_batch = demo_features
            st.success(f"✅ Generated {n_samples} demo transactions!")
        
        if 'demo_batch' in st.session_state:
            if st.button("🔍 Process Batch", type="primary", use_container_width=True):
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
    """Results and analysis page"""
    st.subheader("📈 Results & Analysis")
    
    if not st.session_state.model_trained:
        st.info("ℹ️ Train the model first to see results!")
        return
    
    system = st.session_state.fraud_system
    
    if system.model_metrics is None:
        st.error("❌ No metrics available. Please retrain the model.")
        return
    
    metrics = system.model_metrics['metrics']
    
    st.markdown("### 📊 Performance Dashboard")
    
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
    
    tab1, tab2 = st.tabs(["📊 Confusion Matrix", "📈 ROC Curve"])
    
    with tab1:
        cm = system.model_metrics['confusion_matrix']
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                       labels=dict(x="Predicted", y="Actual"),
                       x=['Normal', 'Fraud'], y=['Normal', 'Fraud'],
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fpr, tpr, _ = system.model_metrics['roc_curve']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                name=f'ROC Curve (AUC = {metrics["ROC AUC"]:.3f})',
                                line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                name='Random', line=dict(dash='dash', color='red')))
        fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)

def save_load_model_page():
    """Model save/load page"""
    st.subheader("💾 Save/Load Model")
    
    if st.session_state.fraud_system is None:
        st.warning("⚠️ Initialize system first!")
        return
    
    system = st.session_state.fraud_system
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💾 Save Model")
        
        if not st.session_state.model_trained:
            st.info("ℹ️ Train a model first!")
        else:
            model_name = st.text_input("Model Name", value="fraud_model")
            
            if st.button("💾 Save Model", use_container_width=True):
                if system.save_model(model_name):
                    st.success("✅ Model saved!")
                    
                    # Create download links
                    if os.path.exists(f'saved_models/{model_name}.h5'):
                        with open(f'saved_models/{model_name}.h5', 'rb') as f:
                            bytes_data = f.read()
                            b64 = base64.b64encode(bytes_data).decode()
                            href = f'<a href="data:file/octet-stream;base64,{b64}" download="{model_name}.h5">📥 Download Model</a>'
                            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📂 Load Model")
        
        uploaded_model = st.file_uploader("Upload Model (.h5)", type=['h5'])
        
        if uploaded_model is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                tmp.write(uploaded_model.getvalue())
                tmp_path = tmp.name
            
            if st.button("📂 Load Model", use_container_width=True):
                if system.load_model(tmp_path):
                    st.session_state.model_trained = True
                    st.success("✅ Model loaded!")
                    os.unlink(tmp_path)

# =========================================================
# MAIN APP
# =========================================================
def main():
    """Main application function"""
    
    page = display_sidebar()
    
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Data Management":
        data_management_page()
    elif page == "🤖 Model Training":
        model_training_page()
    elif page == "🔍 Predict Fraud":
        predict_fraud_page()
    elif page == "📈 Results & Analysis":
        results_analysis_page()
    elif page == "💾 Save/Load Model":
        save_load_model_page()

# =========================================================
# RUN THE APP
# =========================================================
if __name__ == "__main__":
    main()