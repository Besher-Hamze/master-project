"""
Simple Inference Script for XGBoost Network Intrusion Detection
Load model and make predictions on new data
"""

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import time


class XGBoostDetector:
    """Network Intrusion Detector using XGBoost"""
    
    def __init__(self, model_path='models/xgboost_model.json',
                 scaler_path='models/scaler.pkl',
                 encoder_path='models/encoders.pkl'):
        """Initialize detector with trained model"""
        
        print("🔄 Loading XGBoost detector...")
        
        # Load model
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Scaler loaded")
        
        # Load encoders
        with open(encoder_path, 'rb') as f:
            self.encoders = pickle.load(f)
        print(f"✓ Encoders loaded")
        
        print("✅ Detector ready!\n")
    
    def predict(self, features, threshold=0.5):
        """
        Make prediction on network request
        
        Args:
            features: Dictionary or array with 41 features
            threshold: Classification threshold
            
        Returns:
            Dictionary with prediction results
        """
        # Convert dict to array if needed
        if isinstance(features, dict):
            features = np.array([list(features.values())])
        elif isinstance(features, list):
            features = np.array([features])
        
        # Ensure 2D shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        start_time = time.time()
        prediction = self.model.predict(features_scaled)[0]
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Prepare result
        result = {
            'is_attack': bool(prediction),
            'confidence': float(prediction_proba[prediction]),
            'attack_probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else float(prediction),
            'normal_probability': float(prediction_proba[0]) if len(prediction_proba) > 1 else float(1 - prediction),
            'inference_time_ms': round(inference_time, 2)
        }
        
        return result
    
    def predict_batch(self, features_list):
        """Make predictions on multiple samples"""
        results = []
        
        for features in features_list:
            result = self.predict(features)
            results.append(result)
        
        return results


def create_sample_request():
    """Create sample network request for testing"""
    return {
        'duration': 0,
        'protocol_type': 2,      # tcp (encoded)
        'service': 20,           # http (encoded)
        'flag': 10,              # SF (encoded)
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


def create_attack_request():
    """Create sample attack request for testing"""
    return {
        'duration': 0,
        'protocol_type': 1,      # udp (encoded)
        'service': 50,           # other (encoded)
        'flag': 11,              # REJ (encoded)
        'src_bytes': 0,
        'dst_bytes': 0,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 0,
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
        'count': 511,
        'srv_count': 511,
        'serror_rate': 1.0,
        'srv_serror_rate': 1.0,
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 255,
        'dst_host_srv_count': 255,
        'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 1.0,
        'dst_host_srv_serror_rate': 1.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }


def test_detector():
    """Test the detector with sample requests"""
    print("="*80)
    print("🧪 Testing XGBoost Network Intrusion Detector")
    print("="*80 + "\n")
    
    # Initialize detector
    detector = XGBoostDetector()
    
    # Test 1: Normal traffic
    print("📊 Test 1: Normal Traffic")
    print("-" * 80)
    normal_request = create_sample_request()
    result = detector.predict(normal_request)
    
    print(f"   Prediction: {'🔴 Attack' if result['is_attack'] else '🟢 Normal'}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    print(f"   Attack Probability: {result['attack_probability']*100:.2f}%")
    print(f"   Normal Probability: {result['normal_probability']*100:.2f}%")
    print(f"   Inference Time: {result['inference_time_ms']} ms")
    
    # Test 2: Attack traffic
    print("\n📊 Test 2: Attack Traffic (Simulated)")
    print("-" * 80)
    attack_request = create_attack_request()
    result = detector.predict(attack_request)
    
    print(f"   Prediction: {'🔴 Attack' if result['is_attack'] else '🟢 Normal'}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    print(f"   Attack Probability: {result['attack_probability']*100:.2f}%")
    print(f"   Normal Probability: {result['normal_probability']*100:.2f}%")
    print(f"   Inference Time: {result['inference_time_ms']} ms")
    
    # Test 3: Batch prediction
    print("\n📊 Test 3: Batch Prediction")
    print("-" * 80)
    batch_requests = [normal_request, attack_request, normal_request]
    
    start_time = time.time()
    results = detector.predict_batch(batch_requests)
    total_time = (time.time() - start_time) * 1000
    
    for i, result in enumerate(results, 1):
        status = '🔴 Attack' if result['is_attack'] else '🟢 Normal'
        print(f"   Request {i}: {status} (confidence: {result['confidence']*100:.1f}%)")
    
    print(f"   Total Time: {total_time:.2f} ms")
    print(f"   Avg Time per Request: {total_time/len(batch_requests):.2f} ms")
    
    print("\n" + "="*80)
    print("✅ Testing Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_detector()