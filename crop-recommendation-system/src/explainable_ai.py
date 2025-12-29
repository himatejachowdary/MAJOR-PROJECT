"""
Explainable AI Module
Uses SHAP (SHapley Additive exPlanations) to explain model predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from config import MODEL_PATH


class ExplainableAI:
    """
    Provides explainability for crop recommendation predictions using SHAP
    """
    
    def __init__(self, model=None, model_path=None):
        """
        Initialize explainable AI
        
        Args:
            model: Trained model (optional)
            model_path (str): Path to saved model file (optional)
        """
        self.model = model
        self.explainer = None
        self.feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path (str): Path to model file
        """
        try:
            self.model = joblib.load(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
    
    def create_explainer(self, X_train):
        """
        Create SHAP explainer
        
        Args:
            X_train: Training data for background distribution
        """
        try:
            # Use TreeExplainer for tree-based models
            if hasattr(self.model, 'estimators_') or 'XGB' in str(type(self.model)):
                self.explainer = shap.TreeExplainer(self.model)
                print("✓ TreeExplainer created")
            else:
                # Use KernelExplainer for other models (slower)
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(X_train, 100)
                )
                print("✓ KernelExplainer created")
        except Exception as e:
            print(f"✗ Error creating explainer: {str(e)}")
    
    def explain_prediction(self, X_sample, class_names=None):
        """
        Explain a single prediction
        
        Args:
            X_sample: Single sample to explain (1D array or 2D with 1 row)
            class_names: List of class names (optional)
            
        Returns:
            shap_values: SHAP values for the prediction
        """
        if self.explainer is None:
            print("Please create explainer first using create_explainer()")
            return None
        
        # Ensure X_sample is 2D
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        
        try:
            shap_values = self.explainer.shap_values(X_sample)
            return shap_values
        except Exception as e:
            print(f"✗ Error calculating SHAP values: {str(e)}")
            return None
    
    def plot_waterfall(self, X_sample, prediction_class=None, class_names=None, save_path=None):
        """
        Create waterfall plot for a single prediction
        
        Args:
            X_sample: Single sample (1D array)
            prediction_class: Index of predicted class (for multi-class)
            class_names: List of class names
            save_path: Path to save the plot
        """
        if self.explainer is None:
            print("Please create explainer first")
            return
        
        # Ensure X_sample is 2D
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        
        try:
            shap_values = self.explainer.shap_values(X_sample)
            
            # For multi-class, select the predicted class
            if isinstance(shap_values, list):
                if prediction_class is None:
                    prediction_class = 0
                shap_values_class = shap_values[prediction_class][0]
                expected_value = self.explainer.expected_value[prediction_class]
            else:
                shap_values_class = shap_values[0]
                expected_value = self.explainer.expected_value
            
            # Create explanation object
            explanation = shap.Explanation(
                values=shap_values_class,
                base_values=expected_value,
                data=X_sample[0],
                feature_names=self.feature_names
            )
            
            # Plot
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Waterfall plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"✗ Error creating waterfall plot: {str(e)}")
    
    def plot_force(self, X_sample, prediction_class=None, save_path=None):
        """
        Create force plot for a single prediction
        
        Args:
            X_sample: Single sample
            prediction_class: Index of predicted class
            save_path: Path to save the plot
        """
        if self.explainer is None:
            print("Please create explainer first")
            return
        
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        
        try:
            shap_values = self.explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                if prediction_class is None:
                    prediction_class = 0
                shap_values_class = shap_values[prediction_class]
                expected_value = self.explainer.expected_value[prediction_class]
            else:
                shap_values_class = shap_values
                expected_value = self.explainer.expected_value
            
            # Create force plot
            force_plot = shap.force_plot(
                expected_value,
                shap_values_class[0],
                X_sample[0],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Force plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"✗ Error creating force plot: {str(e)}")
    
    def plot_summary(self, X_test, max_display=10, save_path=None):
        """
        Create summary plot showing feature importance across all predictions
        
        Args:
            X_test: Test dataset
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if self.explainer is None:
            print("Please create explainer first")
            return
        
        try:
            shap_values = self.explainer.shap_values(X_test)
            
            plt.figure(figsize=(10, 8))
            
            if isinstance(shap_values, list):
                # For multi-class, show summary for all classes
                shap.summary_plot(
                    shap_values,
                    X_test,
                    feature_names=self.feature_names,
                    max_display=max_display,
                    show=False
                )
            else:
                shap.summary_plot(
                    shap_values,
                    X_test,
                    feature_names=self.feature_names,
                    max_display=max_display,
                    show=False
                )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Summary plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"✗ Error creating summary plot: {str(e)}")
    
    def plot_bar(self, X_test, max_display=10, save_path=None):
        """
        Create bar plot showing mean absolute SHAP values
        
        Args:
            X_test: Test dataset
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if self.explainer is None:
            print("Please create explainer first")
            return
        
        try:
            shap_values = self.explainer.shap_values(X_test)
            
            plt.figure(figsize=(10, 6))
            
            if isinstance(shap_values, list):
                # Average across all classes
                shap_values_avg = np.abs(np.array(shap_values)).mean(axis=0)
                shap.summary_plot(
                    shap_values_avg,
                    X_test,
                    feature_names=self.feature_names,
                    plot_type='bar',
                    max_display=max_display,
                    show=False
                )
            else:
                shap.summary_plot(
                    shap_values,
                    X_test,
                    feature_names=self.feature_names,
                    plot_type='bar',
                    max_display=max_display,
                    show=False
                )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Bar plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"✗ Error creating bar plot: {str(e)}")
    
    def get_feature_importance(self, X_test):
        """
        Get feature importance based on mean absolute SHAP values
        
        Args:
            X_test: Test dataset
            
        Returns:
            pd.DataFrame: Feature importance
        """
        if self.explainer is None:
            print("Please create explainer first")
            return None
        
        try:
            shap_values = self.explainer.shap_values(X_test)
            
            if isinstance(shap_values, list):
                # Average across all classes
                shap_values_avg = np.abs(np.array(shap_values)).mean(axis=0)
                mean_shap = np.abs(shap_values_avg).mean(axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance (based on SHAP values):")
            print(importance_df.to_string(index=False))
            
            return importance_df
            
        except Exception as e:
            print(f"✗ Error calculating feature importance: {str(e)}")
            return None
    
    def explain_prediction_text(self, X_sample, predicted_crop):
        """
        Generate text explanation for a prediction
        
        Args:
            X_sample: Single sample (1D array)
            predicted_crop: Name of predicted crop
            
        Returns:
            str: Text explanation
        """
        if self.explainer is None:
            return "Explainer not initialized"
        
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        
        try:
            shap_values = self.explainer.shap_values(X_sample)
            
            # Get SHAP values for the prediction
            if isinstance(shap_values, list):
                # For multi-class, use the first class (or you can specify)
                shap_vals = shap_values[0][0]
            else:
                shap_vals = shap_values[0]
            
            # Create feature-value pairs with SHAP values
            feature_impacts = []
            for i, (feature, value, shap_val) in enumerate(zip(self.feature_names, X_sample[0], shap_vals)):
                feature_impacts.append({
                    'feature': feature,
                    'value': value,
                    'shap_value': shap_val,
                    'impact': abs(shap_val)
                })
            
            # Sort by impact
            feature_impacts.sort(key=lambda x: x['impact'], reverse=True)
            
            # Generate explanation
            explanation = []
            explanation.append(f"Why {predicted_crop} was recommended:\n")
            
            # Top 3 most important features
            for i, impact in enumerate(feature_impacts[:3], 1):
                direction = "favors" if impact['shap_value'] > 0 else "discourages"
                explanation.append(
                    f"{i}. {impact['feature']} = {impact['value']:.2f} "
                    f"({direction} this crop)"
                )
            
            return "\n".join(explanation)
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"


def main():
    """
    Demonstration of explainable AI
    """
    import os
    
    print("\n" + "="*60)
    print("EXPLAINABLE AI DEMONSTRATION")
    print("="*60)
    
    # Check if model exists
    model_path = os.path.join(MODEL_PATH, 'random_forest_model.pkl')
    
    if not os.path.exists(model_path):
        print("\n✗ Model not found. Please train the model first.")
        print("Run: python src/model_training.py")
        return
    
    # Load model
    xai = ExplainableAI(model_path=model_path)
    
    # Create sample data
    print("\nCreating sample prediction...")
    sample = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]])
    
    # For demonstration, we need training data to create explainer
    print("\nNote: For full SHAP analysis, training data is required.")
    print("This is a simplified demonstration.")
    
    print("\n✓ Explainable AI module ready!")
    print("\nTo use in your application:")
    print("1. Create explainer with training data")
    print("2. Generate SHAP values for predictions")
    print("3. Create visualizations (waterfall, force, summary plots)")


if __name__ == "__main__":
    main()
