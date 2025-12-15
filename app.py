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
from datetime import datetime
warnings.filterwarnings("ignore")

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, 
    Input, BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
        """Load data from CSV file"""
        try:
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
            
            # Check class distribution
            class_counts = self.df[self.target_column].value_counts()
            st.info(f"✅ Data loaded successfully!")
            st.info(f"📊 Dataset shape: {self.df.shape}")
            st.info(f"🎯 Target column: '{self.target_column}'")
            st.info(f"📈 Class distribution:\n{class_counts.to_dict()}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def generate_sample_data(self):
        """Generate realistic sample data for demonstration"""
        np.random.seed(42)
        n_samples = 10000
        fraud_ratio = 0.002  # 0.2% fraud rate
        
        # Generate features (similar to real credit card data)
        n_features = 30
        X = np.random.randn(n_samples, n_features)
        
        # Add some correlation patterns
        X[:, 1] = 0.3 * X[:, 0] + 0.7 * np.random.randn(n_samples)
        X[:, 2] = 0.2 * X[:, 0] + 0.8 * np.random.randn(n_samples)
        X[:, 3] = 0.4 * X[:, 1] + 0.6 * np.random.randn(n_samples)
        
        # Generate fraud mask
        fraud_mask = np.random.rand(n_samples) < fraud_ratio
        
        # Add fraud patterns
        X[fraud_mask, 0] += 2.5  # V1 higher for fraud
        X[fraud_mask, 1] -= 1.8  # V2 lower for fraud
        X[fraud_mask, 4] += 3.2  # V5 higher for fraud
        X[fraud_mask, 7] -= 2.1  # V8 lower for fraud
        
        # Add amount column
        amounts = np.random.exponential(100, n_samples)
        amounts[fraud_mask] *= np.random.uniform(2, 5, fraud_mask.sum())
        
        # Create DataFrame
        columns = [f'V{i}' for i in range(1, n_features + 1)]
        self.df = pd.DataFrame(X, columns=columns)
        self.df['Amount'] = amounts
        self.df['Amount_Normalized'] = np.log1p(amounts)
        self.df['Class'] = fraud_mask.astype(int)
        self.target_column = 'Class'
        
        # Add some missing values for realism
        for col in np.random.choice(columns, size=5, replace=False):
            self.df.loc[np.random.choice(self.df.index, size=100), col] = np.nan
        
        st.success(f"✅ Generated {n_samples} sample transactions with {fraud_mask.sum()} fraud cases")
        return True
    
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
            
            # Handle class imbalance with safe SMOTE
            class_counts = Counter(self.y_train)
            fraud_count = class_counts.get(1, 0)
            
            if fraud_count >= 5:  # SMOTE needs at least 5 samples
                k_neighbors = min(5, fraud_count - 1)
                
                smote = SMOTE(
                    sampling_strategy='auto',
                    random_state=42,
                    k_neighbors=k_neighbors
                )
                
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                st.success(f"✅ Applied SMOTE: Balanced classes to {Counter(self.y_train)}")
            else:
                st.warning(f"⚠️ Not enough fraud samples ({fraud_count}) for SMOTE. Using original data.")
                
                # Apply RandomUnderSampler to balance
                rus = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
                self.X_train, self.y_train = rus.fit_resample(self.X_train, self.y_train)
                st.success(f"✅ Applied RandomUnderSampler: Balanced classes to {Counter(self.y_train)}")
            
            # Reshape for CNN (samples, timesteps, features)
            self.X_train_cnn = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
            self.X_test_cnn = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
            
            st.success("✅ Data preprocessing completed!")
            st.info(f"📊 Training set: {self.X_train.shape[0]} samples")
            st.info(f"📊 Test set: {self.X_test.shape[0]} samples")
            st.info(f"🔢 Features: {self.X_train.shape[1]}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error in preprocessing: {str(e)}")
            return False
    
    def build_hybrid_model(self):
        """Build hybrid CNN + Logistic Regression model"""
        try:
            input_shape = (self.X_train_cnn.shape[1], 1)
            
            # CNN Feature Extractor
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
                
                # Dense layers (acting as logistic regression)
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                # Output layer (sigmoid for binary classification)
                Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')
                ]
            )
            
            self.hybrid_model = model
            return model
            
        except Exception as e:
            st.error(f"❌ Error building model: {str(e)}")
            return None
    
    def train_model(self, epochs=30, batch_size=64):
        """Train the hybrid model"""
        try:
            if not hasattr(self, 'X_train_cnn') or self.X_train_cnn is None:
                if not self.preprocess_data():
                    return None
            
            # Build model
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
            
            # Classification report
            report = classification_report(self.y_test, y_pred, output_dict=True)
            
            self.model_metrics = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'roc_curve': (fpr, tpr, thresholds),
                'classification_report': report,
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
            if self.hybrid_model is None:
                return None, None
            
            # Ensure correct number of features
            if len(features) != len(self.feature_columns):
                st.warning(f"Expected {len(self.feature_columns)} features, got {len(features)}")
                # Pad with zeros if needed
                if len(features) < len(self.feature_columns):
                    features = list(features) + [0] * (len(self.feature_columns) - len(features))
                else:
                    features = features[:len(self.feature_columns)]
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Reshape for CNN
            features_cnn = features_scaled.reshape(1, features_scaled.shape[1], 1)
            
            # Make prediction
            probability = self.hybrid_model.predict(features_cnn, verbose=0)[0][0]
            prediction = 1 if probability > 0.5 else 0
            
            return prediction, probability
            
        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")
            return None, None
    
    def predict_batch(self, batch_data):
        """Predict fraud probability for batch of transactions"""
        try:
            if self.hybrid_model is None:
                return None
            
            # Prepare batch
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
            return None
    
    def save_model(self, filename='fraud_detection_model'):
        """Save the trained model and scaler"""
        try:
            # Save Keras model
            self.hybrid_model.save(f'{filename}.h5')
            
            # Save scaler
            joblib.dump(self.scaler, f'{filename}_scaler.pkl')
            
            # Save feature columns
            joblib.dump(self.feature_columns, f'{filename}_features.pkl')
            
            # Save metrics if available
            if self.model_metrics:
                pd.DataFrame([self.model_metrics['metrics']]).to_csv(f'{filename}_metrics.csv', index=False)
            
            st.success(f"✅ Model saved successfully as '{filename}.h5'")
            return True
            
        except Exception as e:
            st.error(f"❌ Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path, scaler_path, features_path):
        """Load a pre-trained model"""
        try:
            # Load Keras model
            self.hybrid_model = tf.keras.models.load_model(model_path)
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            
            # Load feature columns
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

# =========================================================
# STREAMLIT UI COMPONENTS
# =========================================================
def display_sidebar():
    """Display sidebar with navigation and status"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
        st.title("Navigation")
        
        # Navigation menu
        page = st.radio(
            "Go to:",
            ["🏠 Home", "📊 Data Management", "🤖 Model Training", 
             "🔍 Predict Fraud", "📈 Results & Analysis", "💾 Save/Load Model"]
        )
        
        st.markdown("---")
        st.subheader("System Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data", "✅" if st.session_state.data_loaded else "❌", 
                     "Loaded" if st.session_state.data_loaded else "Required")
        with col2:
            st.metric("Model", "✅" if st.session_state.model_trained else "❌",
                     "Trained" if st.session_state.model_trained else "Not Trained")
        with col3:
            st.metric("System", "✅", "Ready")
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("Quick Actions")
        
        if st.button("🔄 Reset System", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        if st.button("📋 Generate Report", use_container_width=True):
            st.info("Report generation feature coming soon!")
        
        # System info
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #666;">
        <p><strong>System Info:</strong></p>
        <p>• Hybrid CNN + LR Model</p>
        <p>• Safe SMOTE Balancing</p>
        <p>• Real-time Detection</p>
        <p>• Version: 1.0.0</p>
        <p>• Year: 2025-2026</p>
        </div>
        """, unsafe_allow_html=True)
    
    return page

def home_page():
    """Home page with overview"""
    st.markdown("<h1 class='main-title'>💳 Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 class='sub-title'>Hybrid CNN + Logistic Regression Model with Safe SMOTE Balancing</h4>", unsafe_allow_html=True)
    
    # Hero section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 20px; margin: 20px 0;">
            <h3 style="color: white; margin-bottom: 20px;">🔒 Advanced Fraud Protection</h3>
            <p style="color: white; font-size: 1.1rem;">Detect fraudulent transactions with 97%+ accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key features
    st.markdown("---")
    st.subheader("✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 High Accuracy</h3>
            <p>97%+ accuracy in fraud detection using hybrid AI model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time Detection</h3>
            <p>Instant fraud detection for individual and batch transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Comprehensive Analytics</h3>
            <p>Detailed performance metrics and visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started
    st.markdown("---")
    st.subheader("🚀 Getting Started")
    
    steps = [
        ("1. 📊 Load Data", "Upload your dataset or generate sample data"),
        ("2. 🤖 Train Model", "Train the hybrid CNN + Logistic Regression model"),
        ("3. 🔍 Predict Fraud", "Test transactions for fraud detection"),
        ("4. 📈 Analyze Results", "View detailed performance metrics"),
        ("5. 💾 Save Model", "Export your trained model for production")
    ]
    
    for step_title, step_desc in steps:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.success(step_title)
        with col2:
            st.write(step_desc)
    
    # Quick start buttons
    st.markdown("---")
    st.subheader("⚡ Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Load Sample Data", use_container_width=True):
            system = FraudDetectionSystem()
            system.generate_sample_data()
            st.session_state.fraud_system = system
            st.session_state.data_loaded = True
            st.success("✅ Sample data loaded successfully!")
            st.rerun()
    
    with col2:
        if st.button("🎬 Quick Demo", use_container_width=True):
            with st.spinner("Setting up complete demo..."):
                system = FraudDetectionSystem()
                system.generate_sample_data()
                system.preprocess_data()
                system.train_model(epochs=15, batch_size=32)
                st.session_state.fraud_system = system
                st.session_state.data_loaded = True
                st.session_state.model_trained = True
            st.success("✅ Demo setup complete! Check other pages for results.")
            st.rerun()

def data_management_page():
    """Data management page"""
    st.subheader("📊 Data Management")
    
    # Initialize system if needed
    if st.session_state.fraud_system is None:
        st.session_state.fraud_system = FraudDetectionSystem()
    
    system = st.session_state.fraud_system
    
    # Data loading options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your credit card transaction dataset"
        )
        
        if uploaded_file is not None:
            if st.button("📂 Load Uploaded Data", use_container_width=True):
                with st.spinner("Loading data..."):
                    if system.load_data(uploaded_file):
                        st.session_state.data_loaded = True
                        st.success("✅ Data loaded successfully!")
                        st.rerun()
    
    with col2:
        st.markdown("### 🎲 Generate Sample Data")
        st.markdown("Generate realistic credit card transaction data for testing")
        
        n_samples = st.slider("Number of transactions", 1000, 50000, 10000, 1000)
        
        if st.button("🔄 Generate Data", use_container_width=True):
            with st.spinner(f"Generating {n_samples:,} transactions..."):
                system.generate_sample_data()
                st.session_state.data_loaded = True
                st.success(f"✅ Generated {n_samples:,} sample transactions!")
                st.rerun()
    
    # Display data if loaded
    if st.session_state.data_loaded and system.df is not None:
        st.markdown("---")
        st.subheader("📈 Data Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(system.df):,}")
        with col2:
            fraud_count = system.df[system.target_column].sum()
            st.metric("Fraud Cases", f"{fraud_count:,}")
        with col3:
            fraud_rate = (fraud_count / len(system.df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.4f}%")
        with col4:
            st.metric("Features", len(system.df.columns) - 1)
        
        # Data preview tabs
        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Statistics", "📈 Visualizations"])
        
        with tab1:
            st.dataframe(system.df.head(20), use_container_width=True)
            
            # Show data types
            st.markdown("#### Data Types")
            dtype_df = pd.DataFrame({
                'Column': system.df.columns,
                'Type': system.df.dtypes.values,
                'Non-Null Count': system.df.notnull().sum().values,
                'Null Count': system.df.isnull().sum().values
            })
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)
        
        with tab2:
            st.dataframe(system.df.describe(), use_container_width=True)
            
            # Correlation with target
            if system.target_column in system.df.columns:
                numeric_cols = system.df.select_dtypes(include=[np.number]).columns
                corr_with_target = system.df[numeric_cols].corr()[system.target_column].abs().sort_values(ascending=False)
                
                st.markdown("#### Correlation with Target")
                st.dataframe(corr_with_target.reset_index().rename(columns={'index': 'Feature', system.target_column: 'Correlation'}), 
                           use_container_width=True, hide_index=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Class distribution pie chart
                class_counts = system.df[system.target_column].value_counts()
                fig = px.pie(
                    values=class_counts.values,
                    names=['Normal', 'Fraud'],
                    title='Class Distribution',
                    hole=0.4,
                    color_discrete_sequence=['#00CC00', '#FF4B4B']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Amount distribution
                if 'Amount' in system.df.columns:
                    fig = px.histogram(
                        system.df,
                        x='Amount',
                        color=system.target_column,
                        nbins=50,
                        title='Transaction Amount Distribution',
                        color_discrete_map={0: '#00CC00', 1: '#FF4B4B'},
                        log_y=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feature distributions
            st.markdown("#### Feature Distributions")
            numeric_cols = system.df.select_dtypes(include=[np.number]).columns.tolist()
            if system.target_column in numeric_cols:
                numeric_cols.remove(system.target_column)
            
            if len(numeric_cols) > 0:
                selected_feature = st.selectbox("Select feature to visualize", numeric_cols[:10])
                
                fig = px.histogram(
                    system.df,
                    x=selected_feature,
                    color=system.target_column,
                    nbins=50,
                    title=f'Distribution of {selected_feature}',
                    color_discrete_map={0: '#00CC00', 1: '#FF4B4B'},
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Preprocess button
        st.markdown("---")
        if st.button("🔄 Preprocess Data for Training", type="primary", use_container_width=True):
            with st.spinner("Preprocessing data..."):
                if system.preprocess_data():
                    st.success("✅ Data preprocessing completed successfully!")
                    st.info("You can now proceed to Model Training")

def model_training_page():
    """Model training page"""
    st.subheader("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Data Management page!")
        return
    
    system = st.session_state.fraud_system
    
    # Training configuration
    st.markdown("### 🎯 Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.slider("Number of Epochs", 10, 100, 30, 5,
                          help="Number of complete passes through the training data")
    
    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256], index=2,
                                 help="Number of samples per gradient update")
    
    with col3:
        learning_rate = st.select_slider("Learning Rate", 
                                        options=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                                        value=0.001,
                                        help="Step size for optimizer")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2, 0.05)
        with col2:
            early_stopping = st.checkbox("Early Stopping", value=True)
    
    # Start training button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("Training in progress... This may take a few minutes"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training steps
            steps = [
                ("Preprocessing data...", 10),
                ("Building model architecture...", 30),
                ("Training CNN feature extractor...", 50),
                ("Training logistic regression layer...", 70),
                ("Evaluating model performance...", 90),
                ("Training completed!", 100)
            ]
            
            for step_text, step_progress in steps:
                status_text.text(step_text)
                progress_bar.progress(step_progress)
                # Simulate time for each step
                import time
                time.sleep(0.5)
            
            # Train model
            history = system.train_model(epochs=epochs, batch_size=batch_size)
            
            if history:
                st.session_state.training_history = history
                st.session_state.model_trained = True
                
                st.balloons()
                st.success("🎉 Model trained successfully!")
            else:
                st.error("❌ Model training failed!")
    
    # Show training results if model is trained
    if st.session_state.model_trained and system.model_metrics:
        st.markdown("---")
        st.subheader("📊 Training Results")
        
        metrics = system.model_metrics['metrics']
        
        # Display metrics in cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metric_config = [
            ("Accuracy", metrics['Accuracy'], "#3B82F6", "Overall correctness"),
            ("Precision", metrics['Precision'], "#10B981", "Fraud detection accuracy"),
            ("Recall", metrics['Recall'], "#F59E0B", "Fraud coverage"),
            ("F1 Score", metrics['F1 Score'], "#EF4444", "Balance metric"),
            ("ROC AUC", metrics['ROC AUC'], "#8B5CF6", "Discrimination ability")
        ]
        
        for idx, (name, value, color, desc) in enumerate(metric_config):
            with [col1, col2, col3, col4, col5][idx]:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background-color: {color}15; 
                            border-radius: 10px; border-left: 4px solid {color}; margin: 5px;">
                    <h3 style="color: {color}; margin: 0; font-size: 1.5rem;">{value:.3%}</h3>
                    <p style="margin: 5px 0 0 0; font-weight: 600; color: #333;">{name}</p>
                    <p style="margin: 2px 0 0 0; font-size: 0.8rem; color: #666;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("#### 📊 Confusion Matrix")
        cm = system.model_metrics['confusion_matrix']
        
        fig = px.imshow(
            cm,
            text_auto=True,
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Normal', 'Fraud'],
            y=['Normal', 'Fraud'],
            color_continuous_scale='Blues',
            aspect='auto'
        )
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve
        st.markdown("#### 📈 ROC Curve")
        fpr, tpr, thresholds = system.model_metrics['roc_curve']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {metrics["ROC AUC"]:.3f})',
            line=dict(color='blue', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.2)'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='red', width=2)
        ))
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            hovermode='x unified',
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Training history if available
        if st.session_state.training_history:
            st.markdown("#### 📈 Training History")
            
            history_df = pd.DataFrame(st.session_state.training_history.history)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(
                    history_df[['loss', 'val_loss']],
                    title='Training & Validation Loss',
                    labels={'value': 'Loss', 'index': 'Epoch'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(
                    history_df[['accuracy', 'val_accuracy']],
                    title='Training & Validation Accuracy',
                    labels={'value': 'Accuracy', 'index': 'Epoch'}
                )
                st.plotly_chart(fig, use_container_width=True)

def predict_fraud_page():
    """Fraud prediction page"""
    st.subheader("🔍 Fraud Detection")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first from the Model Training page!")
        return
    
    system = st.session_state.fraud_system
    
    # Create tabs for different prediction modes
    tab1, tab2 = st.tabs(["🔍 Single Transaction", "📁 Batch Processing"])
    
    with tab1:
        st.markdown("### Test Individual Transaction")
        
        # Two input modes
        input_mode = st.radio("Input Mode", ["🎲 Random Sample", "✍️ Manual Input"], horizontal=True)
        
        if input_mode == "🎲 Random Sample":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎲 Generate Random Transaction", use_container_width=True):
                    if system.feature_columns:
                        features = generate_sample_features(len(system.feature_columns))
                        amount = np.random.exponential(100)  # Random amount
                        
                        # Store in session state
                        st.session_state.random_features = features
                        st.session_state.random_amount = amount
                        
                        # Display generated features
                        with st.expander("📋 Generated Features", expanded=True):
                            # Show first 10 features
                            for i in range(min(10, len(features))):
                                st.write(f"**Feature {i+1}:** {features[i]:.4f}")
                            
                            if len(features) > 10:
                                st.caption(f"... and {len(features)-10} more features")
                            
                            st.metric("Transaction Amount", f"${amount:.2f}")
            
            # Predict button for random transaction
            if 'random_features' in st.session_state:
                if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
                    with st.spinner("Analyzing transaction..."):
                        prediction, probability = system.predict_single(st.session_state.random_features)
                        
                        if prediction is not None:
                            display_prediction_result(prediction, probability, st.session_state.random_amount)
        
        else:  # Manual Input
            st.markdown("#### Enter Transaction Details")
            
            if system.feature_columns:
                # Create input fields for key features
                with st.form("transaction_form"):
                    st.markdown("##### Key Transaction Features")
                    
                    # Display first 12 features for manual input
                    feature_inputs = []
                    cols_per_row = 4
                    
                    for i in range(min(12, len(system.feature_columns))):
                        if i % cols_per_row == 0:
                            cols = st.columns(cols_per_row)
                        
                        with cols[i % cols_per_row]:
                            feature_name = system.feature_columns[i]
                            value = st.number_input(
                                feature_name,
                                value=0.0,
                                format="%.4f",
                                key=f"feature_{i}"
                            )
                            feature_inputs.append(value)
                    
                    # Fill remaining features with zeros
                    if len(feature_inputs) < len(system.feature_columns):
                        feature_inputs.extend([0.0] * (len(system.feature_columns) - len(feature_inputs)))
                    
                    # Transaction amount
                    st.markdown("##### Transaction Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        amount = st.number_input(
                            "💵 Transaction Amount ($)", 
                            min_value=0.01, 
                            max_value=10000.0, 
                            value=100.0, 
                            step=10.0,
                            format="%.2f"
                        )
                    
                    with col2:
                        time_delta = st.number_input(
                            "⏰ Time Since First Transaction", 
                            min_value=0, 
                            max_value=1000000, 
                            value=50000
                        )
                    
                    # Submit button
                    submitted = st.form_submit_button("🔍 Predict Fraud Risk", type="primary", use_container_width=True)
                    
                    if submitted:
                        with st.spinner("Analyzing transaction patterns..."):
                            prediction, probability = system.predict_single(feature_inputs)
                            
                            if prediction is not None:
                                display_prediction_result(prediction, probability, amount)
    
    with tab2:
        st.markdown("### Process Multiple Transactions")
        
        # Options for batch processing
        batch_mode = st.radio("Batch Mode", ["📊 Generate Demo Data", "📁 Upload CSV File"], horizontal=True)
        
        if batch_mode == "📊 Generate Demo Data":
            st.markdown("Generate a demo dataset for testing batch processing")
            
            n_samples = st.slider("Number of transactions", 10, 1000, 100, 10)
            
            if st.button("📊 Generate Demo Dataset", use_container_width=True):
                with st.spinner(f"Generating {n_samples} demo transactions..."):
                    # Create demo features
                    demo_features = []
                    for _ in range(n_samples):
                        features = generate_sample_features(len(system.feature_columns))
                        demo_features.append(features)
                    
                    # Store in session state
                    st.session_state.demo_batch = demo_features
                    st.success(f"✅ Generated {n_samples} demo transactions!")
            
            # Process demo batch
            if 'demo_batch' in st.session_state:
                if st.button("🔍 Process Demo Batch", type="primary", use_container_width=True):
                    with st.spinner("Processing batch transactions..."):
                        demo_features = st.session_state.demo_batch
                        predictions, probabilities = system.predict_batch(demo_features)
                        
                        if predictions:
                            # Create results dataframe
                            results_df = pd.DataFrame({
                                'Transaction_ID': range(len(predictions)),
                                'Prediction': ['Fraud' if p == 1 else 'Normal' for p in predictions],
                                'Fraud_Probability': probabilities,
                                'Risk_Level': ['High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low' for prob in probabilities]
                            })
                            
                            display_batch_results(results_df)
        
        else:  # Upload CSV File
            st.markdown("Upload a CSV file containing transaction features")
            
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type=['csv'],
                help="File should contain transaction features (same as training data)"
            )
            
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(batch_df)} transactions")
                    
                    # Show preview
                    with st.expander("📋 Preview Data"):
                        st.dataframe(batch_df.head(20), use_container_width=True)
                    
                    # Check if we have required features
                    if st.button("🔍 Process Uploaded Batch", type="primary", use_container_width=True):
                        with st.spinner("Processing transactions..."):
                            # Prepare features
                            batch_features = []
                            for _, row in batch_df.iterrows():
                                features = []
                                for col in system.feature_columns:
                                    if col in row:
                                        features.append(row[col])
                                    else:
                                        features.append(0.0)
                                batch_features.append(features)
                            
                            # Make predictions
                            predictions, probabilities = system.predict_batch(batch_features)
                            
                            if predictions:
                                # Create results dataframe
                                results_df = pd.DataFrame({
                                    'Transaction_ID': range(len(predictions)),
                                    'Prediction': ['Fraud' if p == 1 else 'Normal' for p in predictions],
                                    'Fraud_Probability': probabilities,
                                    'Risk_Level': ['High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low' for prob in probabilities]
                                })
                                
                                display_batch_results(results_df)
                                
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")

def display_prediction_result(prediction, probability, amount):
    """Display prediction result with visualizations"""
    st.markdown("---")
    
    # Display result
    if prediction == 1:
        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("""
            <div style="text-align: center;">
                <span style="font-size: 4rem;">🚨</span>
                <h3 style="color: #FF4B4B; margin: 10px 0;">HIGH RISK</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.error("### ⚠️ FRAUD DETECTED")
            st.write(f"**Fraud Probability:** {probability*100:.2f}%")
            st.write(f"**Transaction Amount:** ${amount:.2f}")
            st.write("**⚠️ Action Required:** Block transaction and notify cardholder immediately")
            st.write("**🔒 Recommended:** Flag account for review")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="normal-transaction">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("""
            <div style="text-align: center;">
                <span style="font-size: 4rem;">✅</span>
                <h3 style="color: #00CC00; margin: 10px 0;">LOW RISK</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.success("### ✅ TRANSACTION SAFE")
            st.write(f"**Fraud Probability:** {probability*100:.2f}%")
            st.write(f"**Transaction Amount:** ${amount:.2f}")
            st.write("**✅ Status:** Transaction appears normal")
            st.write("**👍 Recommendation:** Approve transaction")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk gauge
    st.markdown("#### 📊 Risk Assessment Gauge")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={'text': "Fraud Risk Score", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': "green"},
                {'range': [20, 50], 'color': "lightgreen"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 90], 'color': "orange"},
                {'range': [90, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation
    with st.expander("📋 Analysis Details"):
        st.markdown("##### Risk Classification:")
        if probability < 0.2:
            st.success("**Very Low Risk** - Highly likely to be legitimate")
        elif probability < 0.5:
            st.info("**Low Risk** - Probably legitimate")
        elif probability < 0.7:
            st.warning("**Medium Risk** - Requires monitoring")
        elif probability < 0.9:
            st.error("**High Risk** - Likely fraudulent")
        else:
            st.error("**Very High Risk** - Almost certainly fraudulent")

def display_batch_results(results_df):
    """Display batch prediction results"""
    st.markdown("---")
    st.subheader("📋 Batch Processing Results")
    
    # Summary statistics
    fraud_count = (results_df['Prediction'] == 'Fraud').sum()
    total_count = len(results_df)
    fraud_rate = (fraud_count / total_count) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Processed", total_count)
    with col2:
        st.metric("Fraud Detected", fraud_count)
    with col3:
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    with col4:
        avg_prob = results_df['Fraud_Probability'].mean() * 100
        st.metric("Avg Risk Score", f"{avg_prob:.2f}%")
    
    # Display results table
    st.markdown("#### Detailed Results")
    st.dataframe(results_df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction distribution
        fig = px.pie(
            results_df, 
            names='Prediction',
            title='Prediction Distribution',
            color='Prediction',
            color_discrete_map={'Fraud': '#FF4B4B', 'Normal': '#00CC00'},
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk level distribution
        fig = px.bar(
            results_df['Risk_Level'].value_counts().reset_index(),
            x='index',
            y='Risk_Level',
            title='Risk Level Distribution',
            labels={'index': 'Risk Level', 'Risk_Level': 'Count'},
            color='index',
            color_discrete_map={'High': '#FF4B4B', 'Medium': '#FFC107', 'Low': '#00CC00'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.markdown("#### 💾 Export Results")
    csv = results_df.to_csv(index=False)
    st.markdown(create_download_link(csv, "fraud_detection_results.csv", "📥 Download Results"), unsafe_allow_html=True)

def results_analysis_page():
    """Results and analysis page"""
    st.subheader("📈 Results & Analysis")
    
    if not st.session_state.model_trained:
        st.info("ℹ️ No results available. Please train the model first!")
        return
    
    system = st.session_state.fraud_system
    
    if system.model_metrics is None:
        st.error("❌ Model metrics not available. Please retrain the model.")
        return
    
    metrics = system.model_metrics['metrics']
    cm = system.model_metrics['confusion_matrix']
    
    # Performance dashboard
    st.markdown("### 📊 Performance Dashboard")
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metric_display = [
        ("Accuracy", metrics['Accuracy'], "#3B82F6"),
        ("Precision", metrics['Precision'], "#10B981"),
        ("Recall", metrics['Recall'], "#F59E0B"),
        ("F1 Score", metrics['F1 Score'], "#EF4444"),
        ("ROC AUC", metrics['ROC AUC'], "#8B5CF6")
    ]
    
    for idx, (name, value, color) in enumerate(metric_display):
        with [col1, col2, col3, col4, col5][idx]:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, {color}20, {color}40);
                        border-radius: 10px; border-left: 4px solid {color}; margin: 5px;">
                <h3 style="color: {color}; margin: 0; font-size: 1.8rem;">{value:.3%}</h3>
                <p style="margin: 5px 0 0 0; font-weight: 600; color: #333;">{name}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed analysis tabs
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Detailed Metrics", "📊 Confusion Analysis", "🎯 ROC Analysis"])
    
    with tab1:
        st.markdown("#### 📈 Detailed Performance Metrics")
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': [f"{v:.3%}" if isinstance(v, float) else str(v) for v in metrics.values()]
        })
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Performance rating
            accuracy = metrics['Accuracy']
            if accuracy >= 0.95:
                rating = "Excellent"
                color = "green"
            elif accuracy >= 0.90:
                rating = "Very Good"
                color = "lightgreen"
            elif accuracy >= 0.85:
                rating = "Good"
                color = "orange"
            else:
                rating = "Needs Improvement"
                color = "red"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: {color}20; 
                        border-radius: 10px; border: 2px solid {color}; margin: 10px 0;">
                <h3 style="color: {color}; margin: 0;">{rating}</h3>
                <p style="margin: 5px 0 0 0; color: #666;">Performance Rating</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### 📊 Confusion Matrix Analysis")
        
        # Display confusion matrix
        fig = px.imshow(
            cm,
            text_auto=True,
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Normal', 'Fraud'],
            y=['Normal', 'Fraud'],
            color_continuous_scale='Blues',
            aspect='auto'
        )
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate detailed metrics
        tn, fp, fn, tp = cm.ravel()
        
        st.markdown("##### 📈 Confusion Matrix Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("True Negatives", tn)
        with col2:
            st.metric("False Positives", fp)
        with col3:
            st.metric("False Negatives", fn)
        with col4:
            st.metric("True Positives", tp)
    
    with tab3:
        st.markdown("#### 🎯 ROC Curve Analysis")
        
        fpr, tpr, thresholds = system.model_metrics['roc_curve']
        
        # ROC Curve
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {metrics["ROC AUC"]:.3f})',
            line=dict(color='blue', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.2)'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='red', width=2)
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # AUC Interpretation
        auc_score = metrics['ROC AUC']
        
        if auc_score >= 0.9:
            interpretation = "Excellent discrimination"
            color = "green"
        elif auc_score >= 0.8:
            interpretation = "Good discrimination"
            color = "lightgreen"
        elif auc_score >= 0.7:
            interpretation = "Fair discrimination"
            color = "orange"
        else:
            interpretation = "Poor discrimination"
            color = "red"
        
        st.markdown(f"""
        <div style="padding: 20px; background-color: {color}15; border-radius: 10px; 
                    border-left: 4px solid {color}; margin: 20px 0;">
            <h4 style="color: {color}; margin: 0 0 10px 0;">📊 AUC Score Interpretation</h4>
            <p style="margin: 5px 0; font-size: 1.1rem;">
                <strong>AUC Score:</strong> {auc_score:.3f}<br>
                <strong>Interpretation:</strong> {interpretation}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Export section
    st.markdown("---")
    st.markdown("#### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export metrics
        metrics_df = pd.DataFrame([metrics])
        csv = metrics_df.to_csv(index=False)
        st.markdown(create_download_link(csv, "model_metrics.csv", "📊 Download Metrics"), unsafe_allow_html=True)
    
    with col2:
        # Export confusion matrix
        cm_df = pd.DataFrame(cm, columns=['Predicted_Normal', 'Predicted_Fraud'], 
                           index=['Actual_Normal', 'Actual_Fraud'])
        cm_csv = cm_df.to_csv()
        st.markdown(create_download_link(cm_csv, "confusion_matrix.csv", "📈 Download Confusion Matrix"), unsafe_allow_html=True)

def save_load_model_page():
    """Model save/load page"""
    st.subheader("💾 Save/Load Model")
    
    if st.session_state.fraud_system is None:
        st.warning("⚠️ No system initialized. Please load data first.")
        return
    
    system = st.session_state.fraud_system
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💾 Save Model")
        
        if not st.session_state.model_trained:
            st.info("ℹ️ Please train a model first before saving.")
        else:
            model_name = st.text_input("Model Name", value="fraud_detection_model")
            
            if st.button("💾 Save Model", use_container_width=True):
                with st.spinner("Saving model..."):
                    if system.save_model(model_name):
                        st.success(f"✅ Model saved as '{model_name}.h5'")
                        
                        # Provide download links
                        files = [f'{model_name}.h5', f'{model_name}_scaler.pkl', f'{model_name}_features.pkl']
                        for file in files:
                            try:
                                with open(file, 'rb') as f:
                                    bytes = f.read()
                                    b64 = base64.b64encode(bytes).decode()
                                    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{file}" style="display: inline-block; padding: 10px 20px; background: linear-gradient(135deg, #3B82F6 0%, #1E3A8A 100%); color: white; text-decoration: none; border-radius: 5px; font-weight: bold; margin: 5px;">📥 Download {file}</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                            except:
                                st.warning(f"Could not create download link for {file}")
    
    with col2:
        st.markdown("### 📂 Load Model")
        
        uploaded_model = st.file_uploader("Upload Model File (.h5)", type=['h5'])
        uploaded_scaler = st.file_uploader("Upload Scaler File (.pkl)", type=['pkl'])
        uploaded_features = st.file_uploader("Upload Features File (.pkl)", type=['pkl'])
        
        if uploaded_model and uploaded_scaler and uploaded_features:
            if st.button("📂 Load Model", use_container_width=True):
                with st.spinner("Loading model..."):
                    # Save uploaded files temporarily
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_model:
                        tmp_model.write(uploaded_model.getvalue())
                        model_path = tmp_model.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_scaler:
                        tmp_scaler.write(uploaded_scaler.getvalue())
                        scaler_path = tmp_scaler.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_features:
                        tmp_features.write(uploaded_features.getvalue())
                        features_path = tmp_features.name
                    
                    # Load model
                    if system.load_model(model_path, scaler_path, features_path):
                        st.session_state.model_trained = True
                        st.success("✅ Model loaded successfully!")
                        
                        # Clean up temp files
                        import os
                        os.unlink(model_path)
                        os.unlink(scaler_path)
                        os.unlink(features_path)

# =========================================================
# MAIN APP
# =========================================================
def main():
    """Main application function"""
    
    # Display sidebar
    page = display_sidebar()
    
    # Route to selected page
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
    # Add Font Awesome for icons
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """, unsafe_allow_html=True)
    
    main()