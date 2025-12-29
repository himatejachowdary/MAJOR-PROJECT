"""
Model Training Module
Trains and evaluates multiple ML models for crop recommendation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_PATH, RANDOM_STATE
from data_preprocessing import DataPreprocessor


class CropRecommendationModel:
    """
    Handles training and evaluation of multiple ML models
    """
    
    def __init__(self):
        """
        Initialize the model trainer
        """
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """
        Initialize all models to be trained
        """
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=20,
                min_samples_split=5,
                random_state=RANDOM_STATE
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=RANDOM_STATE
            ),
            'Naive Bayes': GaussianNB()
        }
        
        print(f"✓ Initialized {len(self.models)} models")
        
    def train_model(self, model_name, model, X_train, y_train):
        """
        Train a single model
        
        Args:
            model_name (str): Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        print(f"✓ {model_name} training complete")
        return model
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """
        Evaluate a trained model
        
        Args:
            model_name (str): Name of the model
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        self.results[model_name] = metrics
        
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        return metrics
    
    def cross_validate(self, model_name, model, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            model_name (str): Name of the model
            model: Model instance
            X: Features
            y: Target
            cv (int): Number of folds
            
        Returns:
            float: Mean CV score
        """
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        mean_cv_score = cv_scores.mean()
        
        print(f"\n{model_name} Cross-Validation (5-fold):")
        print(f"  CV Scores: {cv_scores}")
        print(f"  Mean CV Score: {mean_cv_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return mean_cv_score
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        self.initialize_models()
        
        for model_name, model in self.models.items():
            # Train
            trained_model = self.train_model(model_name, model, X_train, y_train)
            
            # Evaluate
            self.evaluate_model(model_name, trained_model, X_test, y_test)
            
            # Update models dict with trained model
            self.models[model_name] = trained_model
        
        # Find best model
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "="*60)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
        print("="*60)
    
    def display_comparison(self):
        """
        Display comparison of all models
        """
        if not self.results:
            print("No results to display. Train models first.")
            return
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        
        print("\n", comparison_df.to_string())
        
        return comparison_df
    
    def plot_confusion_matrix(self, model_name, model, X_test, y_test, save_path=None):
        """
        Plot confusion matrix for a model
        
        Args:
            model_name (str): Name of the model
            model: Trained model
            X_test: Test features
            y_test: Test target
            save_path (str): Path to save the plot
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=sorted(y_test.unique()),
                    yticklabels=sorted(y_test.unique()))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✓ Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance(self, model_name=None):
        """
        Get feature importance from tree-based models
        
        Args:
            model_name (str): Name of the model (default: best model)
            
        Returns:
            pd.DataFrame: Feature importance
        """
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n{model_name} - Feature Importance:")
            print(importance_df.to_string(index=False))
            
            return importance_df
        else:
            print(f"{model_name} does not support feature importance")
            return None
    
    def plot_feature_importance(self, model_name=None, save_path=None):
        """
        Plot feature importance
        
        Args:
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        importance_df = self.get_feature_importance(model_name)
        
        if importance_df is not None:
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importance - {model_name or self.best_model_name}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"✓ Feature importance plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
    
    def save_model(self, model_name=None, filename=None):
        """
        Save a trained model to disk
        
        Args:
            model_name (str): Name of the model to save (default: best model)
            filename (str): Custom filename
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        if filename is None:
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        
        os.makedirs(MODEL_PATH, exist_ok=True)
        filepath = os.path.join(MODEL_PATH, filename)
        
        joblib.dump(model, filepath)
        print(f"✓ {model_name} saved to {filepath}")
    
    def save_all_models(self):
        """
        Save all trained models
        """
        print("\nSaving all models...")
        for model_name in self.models.keys():
            self.save_model(model_name)
        print("✓ All models saved")
    
    def load_model(self, filename):
        """
        Load a saved model
        
        Args:
            filename (str): Name of the model file
            
        Returns:
            Loaded model
        """
        filepath = os.path.join(MODEL_PATH, filename)
        try:
            model = joblib.load(filepath)
            print(f"✓ Model loaded from {filepath}")
            return model
        except FileNotFoundError:
            print(f"✗ Model file not found at {filepath}")
            return None
    
    def generate_classification_report(self, model_name, model, X_test, y_test):
        """
        Generate detailed classification report
        
        Args:
            model_name (str): Name of the model
            model: Trained model
            X_test: Test features
            y_test: Test target
        """
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        print(f"\n{model_name} - Classification Report:")
        print(report)
        
        return report


def main():
    """
    Main function to demonstrate model training
    """
    print("\n" + "="*60)
    print("CROP RECOMMENDATION MODEL TRAINING")
    print("="*60)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess_pipeline()
    
    if result is None:
        print("\n✗ Preprocessing failed. Please check the data file.")
        return
    
    X_train_scaled, X_test_scaled, y_train, y_test = result
    
    # Train models
    trainer = CropRecommendationModel()
    trainer.train_all_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Display comparison
    trainer.display_comparison()
    
    # Feature importance
    trainer.get_feature_importance()
    
    # Save best model
    trainer.save_model()
    
    # Save Random Forest and XGBoost specifically
    trainer.save_model('Random Forest', 'random_forest_model.pkl')
    trainer.save_model('XGBoost', 'xgboost_model.pkl')
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
