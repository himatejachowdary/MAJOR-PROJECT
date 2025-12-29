"""
Dataset Downloader
Downloads the Crop Recommendation dataset from Kaggle
"""

import os
import urllib.request
import zipfile
from config import DATA_PATH


def download_dataset():
    """
    Download the crop recommendation dataset
    
    Note: This is a placeholder. For actual Kaggle download, you need:
    1. Kaggle API credentials
    2. kaggle package installed
    3. Run: kaggle datasets download -d atharvaingle/crop-recommendation-dataset
    """
    
    print("="*60)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    
    print("\nTo download the Crop Recommendation Dataset:")
    print("\nOption 1: Manual Download (Recommended)")
    print("-" * 40)
    print("1. Visit: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
    print("2. Click 'Download' button")
    print("3. Extract the ZIP file")
    print("4. Copy 'Crop_recommendation.csv' to the 'data/' folder")
    
    print("\nOption 2: Using Kaggle API")
    print("-" * 40)
    print("1. Install Kaggle: pip install kaggle")
    print("2. Set up API credentials from https://www.kaggle.com/settings")
    print("3. Run: kaggle datasets download -d atharvaingle/crop-recommendation-dataset")
    print("4. Extract and move to data/ folder")
    
    print("\nOption 3: Use Sample Dataset (For Testing)")
    print("-" * 40)
    print("Run: python create_sample_dataset.py")
    print("This will create a small sample dataset for testing")
    
    print("\n" + "="*60)


def create_sample_dataset():
    """
    Create a small sample dataset for testing purposes
    """
    import pandas as pd
    import numpy as np
    
    print("\nCreating sample dataset...")
    
    # Sample data for demonstration
    np.random.seed(42)
    
    crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
             'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
             'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
             'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    
    # Generate sample data
    n_samples = 100
    data = []
    
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        
        # Generate realistic values based on crop type
        if crop in ['rice', 'cotton', 'jute']:
            n = np.random.randint(60, 100)
            p = np.random.randint(30, 60)
            k = np.random.randint(30, 50)
            temp = np.random.uniform(20, 30)
            humidity = np.random.uniform(70, 90)
            ph = np.random.uniform(6.0, 7.5)
            rainfall = np.random.uniform(150, 250)
        elif crop in ['maize', 'wheat']:
            n = np.random.randint(70, 90)
            p = np.random.randint(35, 50)
            k = np.random.randint(15, 30)
            temp = np.random.uniform(18, 28)
            humidity = np.random.uniform(55, 75)
            ph = np.random.uniform(5.5, 7.0)
            rainfall = np.random.uniform(80, 150)
        else:
            n = np.random.randint(20, 80)
            p = np.random.randint(20, 80)
            k = np.random.randint(20, 80)
            temp = np.random.uniform(15, 35)
            humidity = np.random.uniform(40, 90)
            ph = np.random.uniform(5.0, 8.0)
            rainfall = np.random.uniform(50, 200)
        
        data.append({
            'N': n,
            'P': p,
            'K': k,
            'temperature': round(temp, 2),
            'humidity': round(humidity, 2),
            'ph': round(ph, 2),
            'rainfall': round(rainfall, 2),
            'label': crop
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(DATA_PATH, exist_ok=True)
    filepath = os.path.join(DATA_PATH, 'Crop_recommendation.csv')
    df.to_csv(filepath, index=False)
    
    print(f"✓ Sample dataset created: {filepath}")
    print(f"  Samples: {len(df)}")
    print(f"  Crops: {df['label'].nunique()}")
    print("\n⚠ Note: This is a SAMPLE dataset for testing only.")
    print("For production use, download the full dataset from Kaggle.")
    
    return df


def check_dataset():
    """
    Check if dataset exists
    """
    filepath = os.path.join(DATA_PATH, 'Crop_recommendation.csv')
    
    if os.path.exists(filepath):
        import pandas as pd
        df = pd.read_csv(filepath)
        print(f"\n✓ Dataset found: {filepath}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Crops: {df['label'].nunique()}")
        return True
    else:
        print(f"\n✗ Dataset not found at: {filepath}")
        return False


def main():
    """
    Main function
    """
    print("\n" + "="*60)
    print("CROP RECOMMENDATION DATASET SETUP")
    print("="*60)
    
    # Check if dataset exists
    if check_dataset():
        print("\n✓ Dataset is ready!")
        print("You can proceed with model training.")
    else:
        print("\n⚠ Dataset not found.")
        
        choice = input("\nCreate sample dataset for testing? (y/n): ").lower()
        
        if choice == 'y':
            create_sample_dataset()
            print("\n✓ Sample dataset created!")
            print("\nNext steps:")
            print("1. Run: python src/model_training.py")
            print("2. Run: streamlit run app/streamlit_app.py")
        else:
            download_dataset()


if __name__ == "__main__":
    main()
