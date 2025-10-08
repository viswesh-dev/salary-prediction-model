# Core Data Handling
import pandas as pd  # Data manipulation
import numpy as np   # Numerical operations

# Visualization
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns            # Advanced visualizations
plt.style.use('ggplot')          # Professional styling

# Machine Learning Components
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import (                   # Feature engineering
    StandardScaler, OneHotEncoder, 
    FunctionTransformer, PowerTransformer
)
from sklearn.compose import ColumnTransformer         # Column-wise transformations
from sklearn.pipeline import Pipeline                # ML workflow chaining
from sklearn.impute import SimpleImputer             # Missing value handling

# Model Evaluation
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error
)

# Machine Learning Models
from sklearn.linear_model import LinearRegression       # Baseline model
from sklearn.ensemble import (                         # Ensemble methods
    RandomForestRegressor, 
    GradientBoostingRegressor
)
from xgboost import XGBRegressor                       # Optimized GBM
from catboost import CatBoostRegressor                 # Handles categoricals well

# Explainability & Fairness
import shap                                           # Model interpretation
from aif360.datasets import BinaryLabelDataset        # Bias detection
from aif360.metrics import BinaryLabelDatasetMetric   # Fairness metrics
from aif360.algorithms.preprocessing import Reweighing  # Bias mitigation

# Model Persistence & Deployment
import joblib    # Lightweight model saving
import pickle    # Alternative serialization

# System & Warnings
import warnings
warnings.filterwarnings('ignore')  # Cleaner output

def load_and_explore_data(filepath):
    """
    Loads the dataset and performs initial exploration.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    # Load data with error handling
    try:
        df = pd.read_csv(filepath)
        print("✅ Data loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Basic Exploration
    print("\n🔍 Dataset Overview:")
    print(f"📊 Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    
    print("\n📝 Data Types:")
    print(df.dtypes.value_counts())
    
    print("\n🧐 Missing Values:")
    print(df.isnull().sum().sort_values(ascending=False))
    
    return df

# Load the data
df = load_and_explore_data('expected_ctc.csv')

# Enhanced Data Summary
def detailed_summary(df):
    """Provides a comprehensive statistical summary"""
    print("\n📈 Numerical Features Summary:")
    print(df.describe(percentiles=[.01, .25, .5, .75, .99]).T)
    
    print("\n📊 Categorical Features Summary:")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True).head(10))

detailed_summary(df)

def plot_interactive_distributions(df):
    """Creates interactive distribution plots for all numerical features"""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    plt.figure(figsize=(20, 15))
    for i, col in enumerate(num_cols):
        plt.subplot(4, 4, i+1)
        sns.histplot(df[col], kde=True, color='dodgerblue')
        plt.title(f'Distribution of {col}', fontsize=10)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_interactive_distributions(df)

def plot_correlation_network(df):
    """Creates an advanced correlation network visualization"""
    corr = df.corr()
    plt.figure(figsize=(16, 12))
    sns.clustermap(corr, cmap='coolwarm', annot=True, 
                   fmt=".2f", figsize=(16, 12), 
                   dendrogram_ratio=0.1)
    plt.title('Feature Correlation Network', pad=20, fontsize=16)
    plt.show()

plot_correlation_network(df)

def create_advanced_features(df):
    """Generates sophisticated features for better prediction"""
    
    # Experience Features
    df['Experience_Ratio'] = (
        df['Total_Experience_in_field_applied'] / 
        (df['Total_Experience'] + 1e-5)  # Avoid division by zero
    )
    df['Experience_Discrepancy'] = (
        df['Total_Experience'] - 
        df['Total_Experience_in_field_applied']
    )
    
    # Education Features
    df['Highest_Education'] = df.apply(
        lambda x: 'PhD' if not pd.isna(x['PHD_Specialization']) 
        else 'PG' if not pd.isna(x['PG_Specialization']) 
        else 'UG', axis=1
    )
    
    # Career Progression Features
    df['Career_Growth_Rate'] = (
        df['Current_CTC'] / 
        (df['Total_Experience'] + 1e-5)
    )
    
    # Location Premium (hypothetical)
    premium_locations = ['Bangalore', 'Mumbai', 'Delhi']
    df['Location_Premium'] = df['Preferred_location'].apply(
        lambda x: 1 if x in premium_locations else 0
    )
    
    # Publication Impact
    df['Research_Impact'] = (
        df['Number_of_Publications'] * 
        (1 + df['Certifications'])
    )
    
    return df

df = create_advanced_features(df)

def build_preprocessing_pipeline():
    """Constructs a robust preprocessing pipeline"""
    
    # Numerical Features
    numerical_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()
    
    # Categorical Features
    categorical_cols = df.select_dtypes(
        include=['object']
    ).columns.tolist()
    
    # Numerical Pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier', FunctionTransformer(
            lambda x: np.clip(x, 
                x.quantile(0.01), 
                x.quantile(0.99)
            ))
        ),
        ('scaler', PowerTransformer(method='yeo-johnson'))
    ])
    
    # Categorical Pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])
    
    # Full Preprocessor
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    
    return preprocessor

preprocessor = build_preprocessing_pipeline()

def train_and_evaluate_models(X, y):
    """Trains and evaluates multiple models with tuning"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model Configurations
    models = {
        'XGBoost': {
            'model': XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1]
            }
        },
        'CatBoost': {
            'model': CatBoostRegressor(random_state=42, verbose=0),
            'params': {
                'iterations': [100, 200],
                'depth': [4, 6, 8]
            }
        }
    }
    
    results = {}
    
    for name, config in models.items():
        print(f"\n🚀 Training {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline,
            param_grid={f'model__{k}': v for k, v in config['params'].items()},
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': best_model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'best_params': grid_search.best_params_
        }
        
        print(f"✅ {name} Best Score: {-grid_search.best_score_:.2f}")
        print(f"🏆 Best Params: {grid_search.best_params_}")
    
    return results

results = train_and_evaluate_models(X, y)

def explain_model(model, X):
    """Provides comprehensive model explanations"""
    
    # Process data through pipeline
    processed_data = model.named_steps['preprocessor'].transform(X)
    
    # Get feature names
    num_features = model.named_steps['preprocessor'].transformers_[0][2]
    cat_features = model.named_steps['preprocessor'].transformers_[1][1]\
        .named_steps['encoder'].get_feature_names_out(
            input_features=model.named_steps['preprocessor']\
                .transformers_[1][2]
        )
    
    all_features = num_features.tolist() + cat_features.tolist()
    
    # SHAP Explanation
    explainer = shap.Explainer(model.named_steps['model'])
    shap_values = explainer(processed_data)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, processed_data, feature_names=all_features)
    plt.title('SHAP Feature Importance', fontsize=16)
    plt.show()
    
    # Traditional Feature Importance
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        feat_imp = pd.DataFrame({
            'Feature': all_features,
            'Importance': model.named_steps['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_imp.head(20))
        plt.title('Top 20 Important Features', fontsize=16)
        plt.show()

# Explain the best model
best_model = results['XGBoost']['model']
explain_model(best_model, X_train)

def explain_model(model, X):
    """Provides comprehensive model explanations"""
    
    # Process data through pipeline
    processed_data = model.named_steps['preprocessor'].transform(X)
    
    # Get feature names
    num_features = model.named_steps['preprocessor'].transformers_[0][2]
    cat_features = model.named_steps['preprocessor'].transformers_[1][1]\
        .named_steps['encoder'].get_feature_names_out(
            input_features=model.named_steps['preprocessor']\
                .transformers_[1][2]
        )
    
    all_features = num_features.tolist() + cat_features.tolist()
    
    # SHAP Explanation
    explainer = shap.Explainer(model.named_steps['model'])
    shap_values = explainer(processed_data)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, processed_data, feature_names=all_features)
    plt.title('SHAP Feature Importance', fontsize=16)
    plt.show()
    
    # Traditional Feature Importance
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        feat_imp = pd.DataFrame({
            'Feature': all_features,
            'Importance': model.named_steps['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_imp.head(20))
        plt.title('Top 20 Important Features', fontsize=16)
        plt.show()

# Explain the best model
best_model = results['XGBoost']['model']
explain_model(best_model, X_train)

def save_model_package(model, preprocessor, results, filename):
    """Saves all necessary components for production"""
    
    package = {
        'model': model,
        'preprocessor': preprocessor,
        'metadata': {
            'training_date': pd.Timestamp.now(),
            'performance': results,
            'feature_names': X.columns.tolist()
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(package, f)
    
    print(f"✅ Model package saved as {filename}")

save_model_package(
    best_model, 
    preprocessor, 
    results, 
    'salary_predictor_package.pkl'
)

def save_model_package(model, preprocessor, results, filename):
    """Saves all necessary components for production"""
    
    package = {
        'model': model,
        'preprocessor': preprocessor,
        'metadata': {
            'training_date': pd.Timestamp.now(),
            'performance': results,
            'feature_names': X.columns.tolist()
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(package, f)
    
    print(f"✅ Model package saved as {filename}")

save_model_package(
    best_model, 
    preprocessor, 
    results, 
    'salary_predictor_package.pkl'
)

def predict_salary(model_package, input_data):
    """
    Makes salary predictions on new data
    
    Args:
        model_package (dict): Loaded model package
        input_data (pd.DataFrame): New data to predict
        
    Returns:
        np.array: Predicted salaries
    """
    # Validate input
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Ensure all features are present
    missing_features = set(model_package['metadata']['feature_names']) - \
        set(input_data.columns)
    
    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )
    
    # Make prediction
    return model_package['model'].predict(input_data)

# Example usage
sample_input = X_train.iloc[:1].copy()
print("📊 Sample Input:")
print(sample_input)

print("\n💵 Predicted Salary:")
print(predict_salary(
    pickle.load(open('salary_predictor_package.pkl', 'rb')),
    sample_input
))