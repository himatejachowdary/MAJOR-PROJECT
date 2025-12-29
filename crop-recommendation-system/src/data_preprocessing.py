"""
Data Preprocessing Module
Handles data loading, cleaning, and preparation for model training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from config import DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE


class DataPreprocessor:
    """
    Handles all data preprocessing tasks for the crop recommendation system
    """
    
    def __init__(self, data_file='Crop_recommendation.csv'):
        """
        Initialize the preprocessor
        
        Args:
            data_file (str): Name of the CSV file containing crop data
        """
        self.data_file = os.path.join(DATA_PATH, data_file)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """
        Load the dataset from CSV file
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✓ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            print(f"✗ Error: File not found at {self.data_file}")
            print("Please download the dataset from Kaggle and place it in the data/ folder")
            return None
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None
    
    def explore_data(self):
        """
        Display basic information about the dataset
        """
        if self.df is None:
            print("Please load data first using load_data()")
            return
        
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        
        print(f"\nShape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        
        print("\n--- First 5 Rows ---")
        print(self.df.head())
        
        print("\n--- Data Types ---")
        print(self.df.dtypes)
        
        print("\n--- Missing Values ---")
        print(self.df.isnull().sum())
        
        print("\n--- Statistical Summary ---")
        print(self.df.describe())
        
        print("\n--- Target Distribution ---")
        print(self.df['label'].value_counts())
        
        print("\n--- Unique Crops ---")
        print(f"Total unique crops: {self.df['label'].nunique()}")
        print(f"Crops: {sorted(self.df['label'].unique())}")
        
    def check_data_quality(self):
        """
        Check for data quality issues
        
        Returns:
            dict: Dictionary containing data quality metrics
        """
        if self.df is None:
            return None
        
        quality_report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'unique_crops': self.df['label'].nunique()
        }
        
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        for key, value in quality_report.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return quality_report
    
    def prepare_features(self):
        """
        Prepare features and target variables
        
        Returns:
            tuple: (X, y) features and target
        """
        if self.df is None:
            print("Please load data first")
            return None, None
        
        # Separate features and target
        X = self.df.drop('label', axis=1)
        y = self.df['label']
        
        print(f"\n✓ Features prepared: {X.shape}")
        print(f"✓ Target prepared: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        """
        Split data into training and testing sets
        
        Args:
            X: Features
            y: Target
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Data split complete:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Testing set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            tuple: Scaled X_train, X_test
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n✓ Features scaled using StandardScaler")
        
        return X_train_scaled, X_test_scaled
    
    def save_scaler(self, filename='scaler.pkl'):
        """
        Save the fitted scaler to disk
        
        Args:
            filename (str): Name of the file to save
        """
        os.makedirs(MODEL_PATH, exist_ok=True)
        filepath = os.path.join(MODEL_PATH, filename)
        joblib.dump(self.scaler, filepath)
        print(f"✓ Scaler saved to {filepath}")
    
    def load_scaler(self, filename='scaler.pkl'):
        """
        Load a saved scaler from disk
        
        Args:
            filename (str): Name of the file to load
            
        Returns:
            StandardScaler: Loaded scaler
        """
        filepath = os.path.join(MODEL_PATH, filename)
        try:
            self.scaler = joblib.load(filepath)
            print(f"✓ Scaler loaded from {filepath}")
            return self.scaler
        except FileNotFoundError:
            print(f"✗ Scaler file not found at {filepath}")
            return None
    
    def get_feature_statistics(self):
        """
        Get statistical information about features
        
        Returns:
            pd.DataFrame: Feature statistics
        """
        if self.df is None:
            return None
        
        features = self.df.drop('label', axis=1)
        stats = pd.DataFrame({
            'mean': features.mean(),
            'std': features.std(),
            'min': features.min(),
            'max': features.max(),
            'median': features.median()
        })
        
        print("\n" + "="*60)
        print("FEATURE STATISTICS")
        print("="*60)
        print(stats)
        
        return stats
    
    def preprocess_pipeline(self):
        """
        Complete preprocessing pipeline
        
        Returns:
            tuple: X_train_scaled, X_test_scaled, y_train, y_test
        """
        print("\n" + "="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        self.load_data()
        if self.df is None:
            return None
        
        # Explore and check quality
        self.check_data_quality()
        
        # Prepare features
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Save scaler
        self.save_scaler()
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    """
    Main function to demonstrate preprocessing
    """
    preprocessor = DataPreprocessor()
    
    # Run complete pipeline
    result = preprocessor.preprocess_pipeline()
    
    if result is not None:
        X_train_scaled, X_test_scaled, y_train, y_test = result
        print(f"\n✓ Ready for model training!")
        print(f"  Training samples: {X_train_scaled.shape[0]}")
        print(f"  Test samples: {X_test_scaled.shape[0]}")
        print(f"  Features: {X_train_scaled.shape[1]}")


if __name__ == "__main__":
    main()
