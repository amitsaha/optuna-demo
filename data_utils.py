"""
Utility functions for loading and preparing the UCI Adult Income dataset.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
import numpy as np


def load_adult_income_data(test_size=0.2, random_state=42):
    """
    Load and prepare the UCI Adult Income dataset.
    If the dataset cannot be fetched from UCI, generates a synthetic dataset
    with similar characteristics for demonstration purposes.
    
    Args:
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Split and preprocessed data
    """
    # Try to use ucimlrepo first, fall back to direct URL if not available
    try:
        from ucimlrepo import fetch_ucirepo
        adult = fetch_ucirepo(id=2)
        X = adult.data.features
        y = adult.data.targets
        
        # Encode categorical variables
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
            label_encoders[column] = le
        
        # Encode target variable (income)
        y_encoder = LabelEncoder()
        y_encoded = y_encoder.fit_transform(y.values.ravel())
        
    except Exception as e:
        print(f"Note: Could not fetch from UCI ML Repository ({type(e).__name__})")
        print("Using synthetic dataset with similar characteristics for demonstration...")
        
        # Load from UCI ML Repository directly via URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        try:
            df = pd.read_csv(url, names=column_names, sep=r',\s*', na_values='?', engine='python')
            # Drop rows with missing values
            df = df.dropna()
            
            # Split features and target
            X = df.drop('income', axis=1)
            y = df[['income']]
            
            # Encode categorical variables
            label_encoders = {}
            for column in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column].astype(str))
                label_encoders[column] = le
            
            # Encode target variable (income)
            y_encoder = LabelEncoder()
            y_encoded = y_encoder.fit_transform(y.values.ravel())
            
        except Exception as e2:
            print(f"Could not download from UCI repository: {type(e2).__name__}")
            print("Generating synthetic classification dataset...")
            
            # Generate synthetic data that mimics Adult Income dataset characteristics
            # 14 features, ~30,000 samples, binary classification with ~75/25 class split
            X, y_encoded = make_classification(
                n_samples=30000,
                n_features=14,
                n_informative=10,
                n_redundant=2,
                n_repeated=0,
                n_classes=2,
                weights=[0.75, 0.25],  # Imbalanced like the adult dataset
                flip_y=0.01,
                random_state=random_state
            )
            
            # Convert to DataFrame for consistency
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(14)])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test


def get_dataset_info():
    """
    Get information about the Adult Income dataset.
    
    Returns:
        dict: Dictionary containing dataset metadata
    """
    try:
        from ucimlrepo import fetch_ucirepo
        adult = fetch_ucirepo(id=2)
        
        return {
            'name': adult.metadata.name,
            'description': adult.metadata.abstract,
            'n_features': len(adult.data.features.columns),
            'n_samples': len(adult.data.features),
        }
    except Exception:
        return {
            'name': 'Adult Income (Synthetic)',
            'description': 'Census income classification dataset (synthetic version for demonstration)',
            'n_features': 14,
            'n_samples': 30000,
        }
