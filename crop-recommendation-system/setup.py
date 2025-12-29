"""
Setup Script for Crop Recommendation System
Automates the initial setup process
"""

import os
import sys
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")


def print_step(step_num, text):
    """Print step information"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {text}")
    print(f"{'='*60}\n")


def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8 or higher is required")
        print("Please upgrade Python and try again")
        return False


def create_directories():
    """Create necessary directories"""
    print_step(2, "Creating Project Directories")
    
    directories = ['data', 'models', 'notebooks']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")
    
    return True


def install_dependencies():
    """Install required packages"""
    print_step(3, "Installing Dependencies")
    
    print("This may take a few minutes...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("\n✓ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Error installing dependencies")
        print("Please run manually: pip install -r requirements.txt")
        return False


def setup_environment():
    """Set up environment file"""
    print_step(4, "Setting Up Environment File")
    
    if os.path.exists('.env'):
        print("✓ .env file already exists")
        return True
    
    if os.path.exists('.env.example'):
        # Copy example to .env
        with open('.env.example', 'r') as f:
            content = f.read()
        
        with open('.env', 'w') as f:
            f.write(content)
        
        print("✓ Created .env file from template")
        print("\n⚠ Remember to add your OpenWeatherMap API key to .env file")
        return True
    else:
        print("⚠ .env.example not found, skipping")
        return True


def check_dataset():
    """Check if dataset exists"""
    print_step(5, "Checking Dataset")
    
    dataset_path = os.path.join('data', 'Crop_recommendation.csv')
    
    if os.path.exists(dataset_path):
        print(f"✓ Dataset found at: {dataset_path}")
        return True
    else:
        print(f"✗ Dataset not found at: {dataset_path}")
        print("\nOptions:")
        print("1. Download from Kaggle: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        print("2. Run: python download_dataset.py (to create sample dataset)")
        
        choice = input("\nCreate sample dataset now? (y/n): ").lower()
        
        if choice == 'y':
            try:
                from download_dataset import create_sample_dataset
                create_sample_dataset()
                return True
            except Exception as e:
                print(f"✗ Error creating sample dataset: {e}")
                return False
        else:
            print("\n⚠ Please download the dataset before training models")
            return False


def train_models():
    """Train ML models"""
    print_step(6, "Training Models")
    
    dataset_path = os.path.join('data', 'Crop_recommendation.csv')
    
    if not os.path.exists(dataset_path):
        print("✗ Dataset not found. Skipping model training.")
        print("Please set up the dataset first.")
        return False
    
    choice = input("Train models now? This may take a few minutes (y/n): ").lower()
    
    if choice == 'y':
        try:
            print("\nTraining models...")
            subprocess.check_call([sys.executable, 'src/model_training.py'])
            print("\n✓ Models trained successfully")
            return True
        except subprocess.CalledProcessError:
            print("\n✗ Error training models")
            print("You can train manually later: python src/model_training.py")
            return False
    else:
        print("\n⚠ Skipping model training")
        print("Remember to train models before running the app:")
        print("  python src/model_training.py")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print_header("SETUP COMPLETE!")
    
    print("🎉 Your Crop Recommendation System is ready!\n")
    
    print("📋 NEXT STEPS:\n")
    
    print("1. (Optional) Add OpenWeatherMap API Key:")
    print("   - Edit .env file")
    print("   - Add: OPENWEATHER_API_KEY=your_key_here")
    print("   - Get free key: https://openweathermap.org/api\n")
    
    print("2. If you haven't trained models yet:")
    print("   python src/model_training.py\n")
    
    print("3. Run the web application:")
    print("   streamlit run app/streamlit_app.py\n")
    
    print("4. Open in browser:")
    print("   http://localhost:8501\n")
    
    print("📚 For detailed instructions, see QUICKSTART.md\n")
    
    print("="*60)


def main():
    """Main setup function"""
    print_header("CROP RECOMMENDATION SYSTEM - SETUP")
    
    print("This script will help you set up the project.\n")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    create_directories()
    
    # Install dependencies
    install_choice = input("\nInstall dependencies now? (y/n): ").lower()
    if install_choice == 'y':
        install_dependencies()
    else:
        print("\n⚠ Skipping dependency installation")
        print("Remember to install manually: pip install -r requirements.txt")
    
    # Setup environment
    setup_environment()
    
    # Check/setup dataset
    check_dataset()
    
    # Train models
    model_path = os.path.join('models', 'random_forest_model.pkl')
    if not os.path.exists(model_path):
        train_models()
    else:
        print_step(6, "Models Already Trained")
        print("✓ Models found in models/ directory")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        print("You can run this script again anytime: python setup.py")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        print("Please check the error and try again")
