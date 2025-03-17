"""
Template for data science and data analysis papers.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path=None):
    """
    Load data from a file or generate sample data.
    
    Args:
        file_path (str, optional): Path to the data file
        
    Returns:
        pd.DataFrame: The loaded or generated data
    """
    if file_path:
        # Load data from file
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    else:
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Create features
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(0, 1, n_samples)
        x3 = np.random.normal(0, 1, n_samples)
        
        # Create target with some noise
        y = 2*x1 + 0.5*x2 - 1*x3 + np.random.normal(0, 0.5, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature1': x1,
            'feature2': x2,
            'feature3': x3,
            'target': y
        })
        
        return df

def explore_data(df):
    """
    Explore the data and generate summary statistics and visualizations.
    
    Args:
        df (pd.DataFrame): The data to explore
        
    Returns:
        dict: Summary statistics and insights
    """
    # Summary statistics
    print("Data shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['target'], kde=True)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    # Pairplot for key features
    plt.figure(figsize=(12, 10))
    sns.pairplot(df)
    plt.suptitle('Pairplot of Features', y=1.02)
    plt.tight_layout()
    plt.show()
    
    return {
        'shape': df.shape,
        'dtypes': df.dtypes,
        'summary': df.describe(),
        'missing_values': df.isnull().sum(),
        'correlation': df.corr()
    }

def preprocess_data(df, target_column='target'):
    """
    Preprocess the data for modeling.
    
    Args:
        df (pd.DataFrame): The data to preprocess
        target_column (str): The name of the target column
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    """
    Train multiple models on the data.
    
    Args:
        X_train (np.ndarray): The training features
        y_train (np.ndarray): The training target
        
    Returns:
        dict: The trained models
    """
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return {
        'linear_regression': lr_model,
        'random_forest': rf_model
    }

def evaluate_models(models, X_test, y_test):
    """
    Evaluate the models on the test data.
    
    Args:
        models (dict): The trained models
        X_test (np.ndarray): The test features
        y_test (np.ndarray): The test target
        
    Returns:
        dict: The evaluation metrics
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"\nModel: {name}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{name}: Actual vs Predicted')
        plt.tight_layout()
        plt.show()
    
    return results

def feature_importance(models, feature_names):
    """
    Plot feature importance for applicable models.
    
    Args:
        models (dict): The trained models
        feature_names (list): The names of the features
    """
    if 'random_forest' in models:
        rf_model = models['random_forest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance (Random Forest)')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    
    if 'linear_regression' in models:
        lr_model = models['linear_regression']
        coefficients = lr_model.coef_
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Coefficients (Linear Regression)')
        plt.bar(range(len(coefficients)), coefficients, align='center')
        plt.xticks(range(len(coefficients)), feature_names, rotation=90)
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to demonstrate the implementation.
    """
    parser = argparse.ArgumentParser(description="Data analysis template")
    parser.add_argument("--data", type=str, help="Path to the data file")
    parser.add_argument("--target", type=str, default="target", help="Name of the target column")
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.data)
    
    # Explore data
    explore_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, args.target)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Feature importance
    feature_importance(models, df.drop(columns=[args.target]).columns)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
