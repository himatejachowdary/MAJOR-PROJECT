"""
Basic Model Training Script
A simplified version for quick testing and demonstration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


def quick_train():
    """
    Quick training function for demonstration
    """
    print("\n" + "="*60)
    print("QUICK MODEL TRAINING")
    print("="*60)
    
    # Check if dataset exists
    data_path = 'data/Crop_recommendation.csv'
    
    if not os.path.exists(data_path):
        print("\n✗ Dataset not found!")
        print("Please run: python download_dataset.py")
        return False
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(data_path)
    print(f"   ✓ Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Prepare features and target
    print("\n2. Preparing features...")
    X = df.drop('label', axis=1)
    y = df['label']
    print(f"   ✓ Features: {list(X.columns)}")
    print(f"   ✓ Crops: {y.nunique()} unique categories")
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Training: {len(X_train)} samples")
    print(f"   ✓ Testing: {len(X_test)} samples")
    
    # Scale features
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✓ Features scaled")
    
    # Train model
    print("\n5. Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("   ✓ Model trained")
    
    # Evaluate
    print("\n6. Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   ✓ Accuracy: {accuracy*100:.2f}%")
    
    # Save models
    print("\n7. Saving models...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/random_forest_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("   ✓ Model saved: models/random_forest_model.pkl")
    print("   ✓ Scaler saved: models/scaler.pkl")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n✓ Model Accuracy: {accuracy*100:.2f}%")
    print(f"✓ Model Type: Random Forest")
    print(f"✓ Training Samples: {len(X_train)}")
    print(f"✓ Test Samples: {len(X_test)}")
    print(f"✓ Features: {X_train.shape[1]}")
    print(f"✓ Crops: {y.nunique()}")
    
    print("\n📊 Next Steps:")
    print("1. Run the web app: streamlit run app/streamlit_app.py")
    print("2. Test predictions with different inputs")
    print("3. Review the documentation for more details")
    
    return True


def test_prediction():
    """
    Test the trained model with a sample prediction
    """
    print("\n" + "="*60)
    print("TESTING PREDICTION")
    print("="*60)
    
    # Load model and scaler
    try:
        model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("\n✓ Models loaded successfully")
    except:
        print("\n✗ Models not found. Please train first.")
        return
    
    # Sample input (Rice-friendly conditions)
    sample_input = {
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.8,
        'humidity': 82.0,
        'ph': 6.5,
        'rainfall': 202.9
    }
    
    print("\n📥 Sample Input:")
    for key, value in sample_input.items():
        print(f"   {key}: {value}")
    
    # Prepare features
    features = np.array([[
        sample_input['N'],
        sample_input['P'],
        sample_input['K'],
        sample_input['temperature'],
        sample_input['humidity'],
        sample_input['ph'],
        sample_input['rainfall']
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    # Get confidence
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = max(probabilities) * 100
    
    print("\n📤 Prediction:")
    print(f"   Crop: {prediction.upper()}")
    print(f"   Confidence: {confidence:.2f}%")
    
    print("\n✓ Prediction successful!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode
        test_prediction()
    else:
        # Training mode
        success = quick_train()
        
        if success:
            # Offer to test
            choice = input("\nWould you like to test a prediction? (y/n): ").lower()
            if choice == 'y':
                test_prediction()
