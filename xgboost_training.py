"""
Complete Network Intrusion Detection System using XGBoost
All-in-one script: Data preprocessing, training, evaluation, and model saving
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
from datetime import datetime

# Machine Learning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters"""
    
    # Dataset paths
    TRAIN_DATA_PATH = 'database/KDDTrain+.txt'
    TEST_DATA_PATH = 'database/KDDTest+.txt'
    
    # Output directories
    MODEL_DIR = 'models/'
    RESULTS_DIR = 'results/'
    
    # Model filenames
    MODEL_FILE = 'xgboost_model.json'
    SCALER_FILE = 'scaler.pkl'
    ENCODERS_FILE = 'encoders.pkl'
    LABEL_ENCODER_FILE = 'label_encoder.pkl'
    
    # XGBoost hyperparameters
    XGBOOST_PARAMS = {
        'max_depth': 10,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    }
    
    # For multi-class classification
    XGBOOST_PARAMS_MULTICLASS = {
        'max_depth': 10,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'objective': 'multi:softmax',
        'num_class': 5,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    }
    
    # Dataset columns
    COLUMN_NAMES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'attack_type', 'difficulty_level'
    ]
    
    CATEGORICAL_COLUMNS = ['protocol_type', 'service', 'flag']
    
    # Attack mapping
    ATTACK_MAPPING = {
        'normal': 'normal',
        # DoS attacks
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
        'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos', 'udpstorm': 'dos',
        'processtable': 'dos', 'mailbomb': 'dos',
        # Probe attacks
        'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
        'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
        # R2L attacks
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
        'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l',
        'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
        'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'xlock': 'r2l',
        'xsnoop': 'r2l', 'worm': 'r2l',
        # U2R attacks
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
        'rootkit': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r',
        'xterm': 'u2r'
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def print_step(step_num, text):
    """Print step information"""
    print(f"\n{'─'*80}")
    print(f"📍 Step {step_num}: {text}")
    print(f"{'─'*80}\n")


def create_directories():
    """Create output directories"""
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    print("✓ Directories created")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load training and test datasets"""
    print_step(1, "Loading Data")
    
    print("📥 Loading training data...")
    train_df = pd.read_csv(
        Config.TRAIN_DATA_PATH, 
        header=None, 
        names=Config.COLUMN_NAMES
    )
    print(f"✓ Training data loaded: {len(train_df):,} records")
    
    print("\n📥 Loading test data...")
    test_df = pd.read_csv(
        Config.TEST_DATA_PATH, 
        header=None, 
        names=Config.COLUMN_NAMES
    )
    print(f"✓ Test data loaded: {len(test_df):,} records")
    
    # Display info
    print(f"\n📊 Data Overview:")
    print(f"   Training samples: {len(train_df):,}")
    print(f"   Test samples: {len(test_df):,}")
    print(f"   Features: {len(Config.COLUMN_NAMES) - 2}")  # Exclude attack_type and difficulty
    
    print(f"\n🎯 Attack Distribution (Training):")
    attack_counts = train_df['attack_type'].value_counts()
    for attack, count in attack_counts.head(10).items():
        print(f"   {attack:20s}: {count:,}")
    
    return train_df, test_df


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(train_df, test_df, binary_classification=True):
    """Preprocess data: encoding, scaling, feature engineering"""
    print_step(2, "Data Preprocessing")
    
    # 1. Map attack categories
    print("🔄 Mapping attack categories...")
    train_df['attack_category'] = train_df['attack_type'].map(Config.ATTACK_MAPPING)
    test_df['attack_category'] = test_df['attack_type'].map(Config.ATTACK_MAPPING)
    
    # Handle unknown attacks
    train_df['attack_category'].fillna('unknown', inplace=True)
    test_df['attack_category'].fillna('unknown', inplace=True)
    print("✓ Attack categories mapped")
    
    # 2. Create binary labels
    if binary_classification:
        train_df['label'] = (train_df['attack_category'] != 'normal').astype(int)
        test_df['label'] = (test_df['attack_category'] != 'normal').astype(int)
        print("✓ Binary labels created (0=Normal, 1=Attack)")
    else:
        # Multi-class: encode attack categories
        label_encoder = LabelEncoder()
        combined_labels = pd.concat([
            train_df['attack_category'], 
            test_df['attack_category']
        ])
        label_encoder.fit(combined_labels)
        
        train_df['label'] = label_encoder.transform(train_df['attack_category'])
        test_df['label'] = label_encoder.transform(test_df['attack_category'])
        
        # Save label encoder
        with open(os.path.join(Config.MODEL_DIR, Config.LABEL_ENCODER_FILE), 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"✓ Multi-class labels created ({len(label_encoder.classes_)} classes)")
        print(f"   Classes: {list(label_encoder.classes_)}")
    
    # 3. Encode categorical features
    print("\n🔢 Encoding categorical features...")
    encoders = {}
    
    for col in Config.CATEGORICAL_COLUMNS:
        # Combine train and test for consistent encoding
        combined = pd.concat([train_df[col], test_df[col]])
        
        le = LabelEncoder()
        le.fit(combined)
        
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        
        encoders[col] = le
        print(f"   ✓ Encoded {col}: {len(le.classes_)} unique values")
    
    # Save encoders
    with open(os.path.join(Config.MODEL_DIR, Config.ENCODERS_FILE), 'wb') as f:
        pickle.dump(encoders, f)
    print("✓ Encoders saved")
    
    # 4. Separate features and labels
    print("\n🎯 Preparing features and labels...")
    
    columns_to_drop = ['attack_type', 'difficulty_level', 'attack_category', 'label']
    
    X_train = train_df.drop(columns_to_drop, axis=1)
    y_train = train_df['label']
    
    X_test = test_df.drop(columns_to_drop, axis=1)
    y_test = test_df['label']
    
    print(f"✓ Features shape: {X_train.shape}")
    print(f"✓ Label distribution:")
    print(f"   Training: {dict(y_train.value_counts())}")
    print(f"   Test: {dict(y_test.value_counts())}")
    
    # 5. Normalize features
    print("\n📊 Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open(os.path.join(Config.MODEL_DIR, Config.SCALER_FILE), 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Features normalized and scaler saved")
    
    return X_train_scaled, y_train, X_test_scaled, y_test


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train, X_test, y_test, binary_classification=True):
    """Train XGBoost model"""
    print_step(3, "Training XGBoost Model")
    
    # Select parameters
    if binary_classification:
        params = Config.XGBOOST_PARAMS.copy()
        print("📝 Training binary classification model")
    else:
        params = Config.XGBOOST_PARAMS_MULTICLASS.copy()
        print("📝 Training multi-class classification model")
    
    print(f"\n⚙️  Model Parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # Create model
    print("\n🏗️  Building model...")
    if binary_classification:
        model = xgb.XGBClassifier(**params)
    else:
        model = xgb.XGBClassifier(**params)
    
    # Train model with evaluation
    print("\n🏋️  Training started...")
    start_time = time.time()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return model


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, binary_classification=True):
    """Evaluate model performance"""
    print_step(4, "Model Evaluation")
    
    # Predictions
    print("🔮 Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training metrics
    print("\n📊 Training Set Performance:")
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"   Accuracy: {train_accuracy*100:.2f}%")
    
    # Test metrics
    print("\n📊 Test Set Performance:")
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='binary' if binary_classification else 'weighted')
    test_recall = recall_score(y_test, y_test_pred, average='binary' if binary_classification else 'weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='binary' if binary_classification else 'weighted')
    
    print(f"   Accuracy:  {test_accuracy*100:.2f}%")
    print(f"   Precision: {test_precision*100:.2f}%")
    print(f"   Recall:    {test_recall*100:.2f}%")
    print(f"   F1-Score:  {test_f1*100:.2f}%")
    
    # ROC-AUC for binary classification
    if binary_classification:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)
        print(f"   ROC-AUC:   {test_auc*100:.2f}%")
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    print("📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Save metrics
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm.tolist()
    }
    
    if binary_classification:
        metrics['test_auc'] = test_auc
    
    return metrics, y_test_pred


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(model, X_test, y_test, y_test_pred, metrics, binary_classification=True):
    """Create visualization plots"""
    print_step(5, "Creating Visualizations")
    
    # Create figure with subplots
    if binary_classification:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix
    print("📊 Plotting confusion matrix...")
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. Feature Importance
    print("📊 Plotting feature importance...")
    feature_importance = model.feature_importances_
    top_features = 20
    indices = np.argsort(feature_importance)[-top_features:]
    
    axes[0, 1].barh(range(top_features), feature_importance[indices])
    axes[0, 1].set_yticks(range(top_features))
    axes[0, 1].set_yticklabels([f'Feature {i}' for i in indices])
    axes[0, 1].set_title(f'Top {top_features} Feature Importance', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Importance')
    
    # 3. Metrics comparison
    print("📊 Plotting metrics...")
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [
        metrics['test_accuracy'],
        metrics['test_precision'],
        metrics['test_recall'],
        metrics['test_f1']
    ]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = axes[1, 0].bar(metrics_names, metrics_values, color=colors)
    axes[1, 0].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold')
    
    # 4. ROC Curve (binary only) or Class Distribution
    if binary_classification:
        print("📊 Plotting ROC curve...")
        y_test_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        
        axes[1, 1].plot(fpr, tpr, linewidth=2, label=f"AUC = {metrics['test_auc']:.3f}")
        axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[1, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Class distribution for multi-class
        unique, counts = np.unique(y_test, return_counts=True)
        axes[1, 1].bar(unique, counts)
        axes[1, 1].set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    
    # Save figure
    results_path = os.path.join(Config.RESULTS_DIR, 'xgboost_results.png')
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    print(f"✓ Results saved to {results_path}")
    
    plt.close()


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model(model):
    """Save trained model"""
    print_step(6, "Saving Model")
    
    model_path = os.path.join(Config.MODEL_DIR, Config.MODEL_FILE)
    
    # Save model in JSON format (recommended for XGBoost)
    model.save_model(model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Also save as pickle for compatibility
    pickle_path = os.path.join(Config.MODEL_DIR, 'xgboost_model.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model also saved as pickle to {pickle_path}")
    
    # Get model size
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"✓ Model size: {model_size:.2f} MB")


# ============================================================================
# INFERENCE TEST
# ============================================================================

def test_inference(model):
    """Test model inference with sample data"""
    print_step(7, "Testing Inference")
    
    # Load preprocessors
    with open(os.path.join(Config.MODEL_DIR, Config.SCALER_FILE), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(Config.MODEL_DIR, Config.ENCODERS_FILE), 'rb') as f:
        encoders = pickle.load(f)
    
    # Create sample normal traffic
    sample_normal = {
        'duration': 0,
        'protocol_type': 2,  # Already encoded
        'service': 20,
        'flag': 10,
        'src_bytes': 181,
        'dst_bytes': 5450,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 1,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 8,
        'srv_count': 8,
        'serror_rate': 0.0,
        'srv_serror_rate': 0.0,
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 9,
        'dst_host_srv_count': 9,
        'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 0.11,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }
    
    # Convert to array
    X_sample = np.array([list(sample_normal.values())])
    X_sample_scaled = scaler.transform(X_sample)
    
    # Predict
    start_time = time.time()
    prediction = model.predict(X_sample_scaled)[0]
    prediction_proba = model.predict_proba(X_sample_scaled)[0]
    inference_time = (time.time() - start_time) * 1000  # ms
    
    print(f"✓ Prediction: {'Attack' if prediction == 1 else 'Normal'}")
    print(f"✓ Confidence: {prediction_proba[prediction]*100:.2f}%")
    print(f"✓ Inference time: {inference_time:.2f} ms")
    print(f"✓ Model is ready for deployment!")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(binary_classification=True):
    """Main execution pipeline"""
    
    print_header("🛡️  Network Intrusion Detection System - XGBoost Training")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Mode: {'Binary Classification' if binary_classification else 'Multi-class Classification'}")
    
    # Create directories
    create_directories()
    
    # Step 1: Load data
    train_df, test_df = load_data()
    
    # Step 2: Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(
        train_df, test_df, binary_classification
    )
    
    # Step 3: Train model
    model = train_model(X_train, y_train, X_test, y_test, binary_classification)
    
    # Step 4: Evaluate model
    metrics, y_test_pred = evaluate_model(
        model, X_train, y_train, X_test, y_test, binary_classification
    )
    
    # Step 5: Plot results
    plot_results(model, X_test, y_test, y_test_pred, metrics, binary_classification)
    
    # Step 6: Save model
    save_model(model)
    
    # Step 7: Test inference
    test_inference(model)
    
    # Final summary
    print_header("✅ Training Complete!")
    print(f"📊 Final Results:")
    print(f"   Test Accuracy:  {metrics['test_accuracy']*100:.2f}%")
    print(f"   Test Precision: {metrics['test_precision']*100:.2f}%")
    print(f"   Test Recall:    {metrics['test_recall']*100:.2f}%")
    print(f"   Test F1-Score:  {metrics['test_f1']*100:.2f}%")
    
    if binary_classification:
        print(f"   ROC-AUC:        {metrics['test_auc']*100:.2f}%")
    
    print(f"\n📁 Saved Files:")
    print(f"   Model: {Config.MODEL_DIR}{Config.MODEL_FILE}")
    print(f"   Scaler: {Config.MODEL_DIR}{Config.SCALER_FILE}")
    print(f"   Encoders: {Config.MODEL_DIR}{Config.ENCODERS_FILE}")
    print(f"   Results: {Config.RESULTS_DIR}xgboost_results.png")
    
    print(f"\n✅ Model training complete!")
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train XGBoost model for Network Intrusion Detection'
    )
    parser.add_argument(
        '--multiclass',
        action='store_true',
        help='Use multi-class classification (5 classes) instead of binary'
    )
    
    args = parser.parse_args()
    
    # Run training
    main(binary_classification=not args.multiclass)