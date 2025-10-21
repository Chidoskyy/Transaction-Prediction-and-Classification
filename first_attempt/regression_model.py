#!/usr/bin/env python3
"""
Bank Statement Regression Model - Assignment 1.3
Machine Learning model to predict transaction amounts (no overfitting)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

class TransactionPredictor:
    def __init__(self, data_file: str = 'extracted_transactions.json'):
        """Initialize the transaction predictor"""
        self.data_file = data_file
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def load_and_prepare_data(self):
        """Load and prepare transaction data for ML"""
        print("Loading transaction data...")
        
        # Load data
        with open(self.data_file, 'r') as f:
            transactions = json.load(f)
        
        self.df = pd.DataFrame(transactions)
        print(f"Loaded {len(self.df)} transactions")
        
        # Convert dates
        self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
        self.df['transaction_postdate'] = pd.to_datetime(self.df['transaction_postdate'])
        
        # Remove payments for spending prediction (we want to predict spending amounts)
        self.df = self.df[self.df['spend_category'] != 'PAYMENT'].copy()
        print(f"After removing payments: {len(self.df)} transactions")
        
        return self.df
    
    def create_features(self):
        """Create comprehensive features using ALL database features with proper encoding"""
        print("Creating comprehensive features using ALL database features...")
        
        # ===== DATE-BASED FEATURES (from transaction_date and transaction_postdate) =====
        print("  Creating date-based features...")
        
        # Transaction date features
        self.df['day_of_week'] = self.df['transaction_date'].dt.dayofweek
        self.df['day_of_month'] = self.df['transaction_date'].dt.day
        self.df['month'] = self.df['transaction_date'].dt.month
        self.df['year'] = self.df['transaction_date'].dt.year
        self.df['quarter'] = self.df['transaction_date'].dt.quarter
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.df['is_month_start'] = (self.df['day_of_month'] <= 5).astype(int)
        self.df['is_month_end'] = (self.df['day_of_month'] >= 25).astype(int)
        
        # Transaction postdate features (processing delay)
        self.df['post_day_of_week'] = self.df['transaction_postdate'].dt.dayofweek
        self.df['post_day_of_month'] = self.df['transaction_postdate'].dt.day
        self.df['post_month'] = self.df['transaction_postdate'].dt.month
        self.df['post_year'] = self.df['transaction_postdate'].dt.year
        
        # Processing delay features
        self.df['processing_delay_days'] = (self.df['transaction_postdate'] - self.df['transaction_date']).dt.days
        self.df['is_same_day_post'] = (self.df['processing_delay_days'] == 0).astype(int)
        self.df['is_next_day_post'] = (self.df['processing_delay_days'] == 1).astype(int)
        
        # Time-based features
        self.df['days_since_start'] = (self.df['transaction_date'] - self.df['transaction_date'].min()).dt.days
        self.df['days_since_year_start'] = self.df['transaction_date'].dt.dayofyear
        self.df['week_of_year'] = self.df['transaction_date'].dt.isocalendar().week
        
        # ===== TRANSACTION DESCRIPTION FEATURES (comprehensive text analysis) =====
        print("  Creating transaction description features...")
        
        # Basic text features
        self.df['description_length'] = self.df['transaction_description'].str.len()
        self.df['word_count'] = self.df['transaction_description'].str.split().str.len()
        self.df['char_count_no_spaces'] = self.df['transaction_description'].str.replace(' ', '').str.len()
        self.df['avg_word_length'] = self.df['description_length'] / (self.df['word_count'] + 1)
        
        # Text pattern features
        self.df['has_numbers'] = self.df['transaction_description'].str.contains(r'\d+', na=False).astype(int)
        self.df['has_special_chars'] = self.df['transaction_description'].str.contains(r'[^a-zA-Z0-9\s]', na=False).astype(int)
        self.df['has_forward_slash'] = self.df['transaction_description'].str.contains('/', na=False).astype(int)
        self.df['has_hash'] = self.df['transaction_description'].str.contains('#', na=False).astype(int)
        self.df['has_dollar_sign'] = self.df['transaction_description'].str.contains('\$', na=False).astype(int)
        
        # Location features (Canadian provinces)
        canadian_provinces = ['MB', 'ON', 'AB', 'BC', 'SK', 'QC', 'NS', 'NB', 'NL', 'PE', 'YT', 'NT', 'NU']
        self.df['has_province'] = self.df['transaction_description'].str.contains('|'.join(canadian_provinces), na=False).astype(int)
        self.df['has_city'] = self.df['transaction_description'].str.contains('WINNIPEG|TORONTO|VANCOUVER|MONTREAL|CALGARY', case=False, na=False).astype(int)
        
        # Merchant type features (expanded patterns)
        merchant_patterns = {
            'grocery': ['SUPERSTORE', 'SOBEYS', 'GROCERY', 'MARKET', 'FOOD', 'FRESH', 'SAFEWAY', 'METRO'],
            'restaurant': ['TIM HORTONS', 'STARBUCKS', 'RESTAURANT', 'CAFE', 'COFFEE', 'PIZZA', 'SUBWAY', 'MCDONALD'],
            'gas': ['GAS', 'FUEL', 'SHELL', 'ESSO', 'PETRO', 'STATION', 'MOBIL', 'CHEVRON'],
            'pharmacy': ['SHOPPERS', 'PHARMACY', 'DRUG', 'MEDICAL', 'HEALTH', 'CLINIC'],
            'retail': ['STORE', 'SHOP', 'RETAIL', 'WALMART', 'AMAZON', 'BEST BUY', 'CANADIAN TIRE'],
            'service': ['SERVICE', 'CENTRE', 'CENTER', 'OFFICE', 'CLINIC', 'REPAIR', 'MAINTENANCE'],
            'entertainment': ['CINEMA', 'MOVIE', 'THEATRE', 'THEATER', 'ENTERTAINMENT', 'GAME'],
            'transportation': ['TAXI', 'UBER', 'LYFT', 'BUS', 'TRAIN', 'AIRPORT', 'PARKING'],
            'utilities': ['HYDRO', 'ELECTRIC', 'WATER', 'GAS', 'INTERNET', 'PHONE', 'CABLE'],
            'financial': ['BANK', 'ATM', 'WITHDRAWAL', 'DEPOSIT', 'TRANSFER', 'PAYMENT']
        }
        
        for category, patterns in merchant_patterns.items():
            pattern = '|'.join(patterns)
            self.df[f'is_{category}'] = self.df['transaction_description'].str.contains(pattern, case=False, na=False).astype(int)
        
        # Payment-specific features
        self.df['is_payment'] = self.df['transaction_description'].str.contains('PAYMENT|PAIEMENT', case=False, na=False).astype(int)
        self.df['is_thank_you'] = self.df['transaction_description'].str.contains('THANK YOU|MERCI', case=False, na=False).astype(int)
        
        # ===== SPEND CATEGORY FEATURES (comprehensive encoding) =====
        print("  Creating spend category features...")
        
        # Label encoding for categories
        le_category = LabelEncoder()
        self.df['category_encoded'] = le_category.fit_transform(self.df['spend_category'])
        
        # One-hot encoding for categories (if not too many unique values)
        unique_categories = self.df['spend_category'].nunique()
        if unique_categories <= 20:  # Only if reasonable number of categories
            category_dummies = pd.get_dummies(self.df['spend_category'], prefix='category')
            self.df = pd.concat([self.df, category_dummies], axis=1)
            print(f"    Created {unique_categories} one-hot encoded category features")
        
        # Category frequency features
        category_counts = self.df['spend_category'].value_counts()
        self.df['category_frequency'] = self.df['spend_category'].map(category_counts)
        self.df['category_frequency_log'] = np.log1p(self.df['category_frequency'])
        
        # ===== SOURCE FILE FEATURES (comprehensive encoding) =====
        print("  Creating source file features...")
        
        # Label encoding for source files
        le_source = LabelEncoder()
        self.df['source_encoded'] = le_source.fit_transform(self.df['source_file'])
        
        # Extract month/year from source file names
        self.df['source_month'] = pd.to_numeric(self.df['source_file'].str.extract(r'(\d+)')[0], errors='coerce').fillna(0)
        
        # Source file frequency features
        source_counts = self.df['source_file'].value_counts()
        self.df['source_frequency'] = self.df['source_file'].map(source_counts)
        
        # ===== AMOUNT-BASED FEATURES (comprehensive numerical features) =====
        print("  Creating amount-based features...")
        
        # Basic amount features
        self.df['amount_abs'] = abs(self.df['amount'])
        self.df['amount_log'] = np.log1p(self.df['amount_abs'])
        self.df['amount_sqrt'] = np.sqrt(self.df['amount_abs'])
        self.df['amount_rounded'] = round(self.df['amount'])
        self.df['amount_decimal'] = self.df['amount'] - self.df['amount'].astype(int)
        
        # Amount categorization
        self.df['is_small_amount'] = (self.df['amount_abs'] < 10).astype(int)
        self.df['is_medium_amount'] = ((self.df['amount_abs'] >= 10) & (self.df['amount_abs'] < 50)).astype(int)
        self.df['is_large_amount'] = (self.df['amount_abs'] >= 50).astype(int)
        
        # Amount percentiles
        self.df['amount_percentile'] = self.df['amount'].rank(pct=True)
        
        # Rolling statistics (if enough data)
        if len(self.df) > 10:
            self.df['amount_rolling_mean_7'] = self.df['amount'].rolling(window=7, min_periods=1).mean()
            self.df['amount_rolling_std_7'] = self.df['amount'].rolling(window=7, min_periods=1).std()
        
        # ===== INTERACTION FEATURES =====
        print("  Creating interaction features...")
        
        # Date-amount interactions
        self.df['weekend_amount'] = self.df['is_weekend'] * self.df['amount']
        self.df['month_end_amount'] = self.df['is_month_end'] * self.df['amount']
        
        # Category-amount interactions
        self.df['category_amount_interaction'] = self.df['category_encoded'] * self.df['amount']
        
        # Description-amount interactions
        self.df['description_length_amount'] = self.df['description_length'] * self.df['amount']
        
        print(f"Created {len(self.df.columns)} comprehensive features using ALL database fields")
        print(f"Features include: date features, text features, category features, source features, amount features, and interactions")
        return self.df
    
    def prepare_ml_data(self):
        """Prepare comprehensive ML data using ALL features"""
        print("Preparing comprehensive ML data...")
        
        # Define comprehensive feature columns (excluding target and original columns)
        exclude_columns = [
            'transaction_date', 'transaction_postdate', 'transaction_description', 
            'spend_category', 'amount', 'source_file', 'processed_description'
        ]
        
        # Get all feature columns (everything except excluded columns)
        feature_columns = [col for col in self.df.columns if col not in exclude_columns]
        
        print(f"Using {len(feature_columns)} features for ML:")
        print(f"  Date features: {len([col for col in feature_columns if any(x in col for x in ['day', 'month', 'year', 'week', 'quarter', 'weekend', 'processing'])])}")
        print(f"  Text features: {len([col for col in feature_columns if any(x in col for x in ['description', 'word', 'char', 'has_', 'is_'])])}")
        print(f"  Category features: {len([col for col in feature_columns if 'category' in col])}")
        print(f"  Source features: {len([col for col in feature_columns if 'source' in col])}")
        print(f"  Amount features: {len([col for col in feature_columns if 'amount' in col])}")
        print(f"  Interaction features: {len([col for col in feature_columns if 'interaction' in col or col.endswith('_amount')])}")
        
        # Prepare features and target
        X = self.df[feature_columns].copy()
        y = self.df['amount'].copy()
        
        # Handle any missing values
        X = X.fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {feature_columns[:10]}..." if len(feature_columns) > 10 else f"Feature columns: {feature_columns}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return X, y
    
    def train_models(self):
        """Train multiple regression models"""
        print("Training regression models...")
        
        # Define models to try
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        model_scores = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            # Store model and scores
            self.models[name] = model
            model_scores[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'overfitting': train_r2 - test_r2
            }
            
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Train RMSE: ${train_rmse:.2f}")
            print(f"  Test RMSE: ${test_rmse:.2f}")
            print(f"  Overfitting (R² diff): {train_r2 - test_r2:.4f}")
        
        return model_scores
    
    def prevent_overfitting(self):
        """Implement techniques to prevent overfitting"""
        print("\nImplementing overfitting prevention...")
        
        # Cross-validation for best model selection
        print("Performing cross-validation...")
        
        cv_scores = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            cv_scores[name] = {
                'mean_cv_score': scores.mean(),
                'std_cv_score': scores.std(),
                'scores': scores
            }
            print(f"{name}: CV R² = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Hyperparameter tuning for best models
        print("\nPerforming hyperparameter tuning...")
        
        # Random Forest tuning
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_params,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        rf_grid.fit(self.X_train, self.y_train)
        
        print(f"Best Random Forest params: {rf_grid.best_params_}")
        print(f"Best Random Forest CV score: {rf_grid.best_score_:.4f}")
        
        # Gradient Boosting tuning
        gb_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        gb_grid = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_params,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        gb_grid.fit(self.X_train, self.y_train)
        
        print(f"Best Gradient Boosting params: {gb_grid.best_params_}")
        print(f"Best Gradient Boosting CV score: {gb_grid.best_score_:.4f}")
        
        # Update models with best parameters
        self.models['Random Forest (Tuned)'] = rf_grid.best_estimator_
        self.models['Gradient Boosting (Tuned)'] = gb_grid.best_estimator_
        
        return cv_scores
    
    def select_best_model(self):
        """Select the best model based on performance and overfitting"""
        print("\nSelecting best model...")
        
        # Evaluate all models on test set
        model_performance = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            
            # Calculate overfitting (difference between train and test R²)
            y_pred_train = model.predict(self.X_train)
            train_r2 = r2_score(self.y_train, y_pred_train)
            overfitting = train_r2 - r2
            
            model_performance[name] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'overfitting': overfitting
            }
            
            print(f"{name}:")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAE: ${mae:.2f}")
            print(f"  Overfitting: {overfitting:.4f}")
        
        # Select best model (highest R² with low overfitting)
        best_model_name = max(model_performance.keys(), 
                            key=lambda x: model_performance[x]['r2'] - model_performance[x]['overfitting'])
        
        self.best_model = self.models[best_model_name]
        print(f"\nBest model: {best_model_name}")
        
        return model_performance
    
    def analyze_feature_importance(self):
        """Analyze feature importance for the best model"""
        print("\nAnalyzing feature importance...")
        
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = self.X_train.columns
            importance = self.best_model.feature_importances_
            
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            for i, row in self.feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.feature_importance
    
    def create_visualizations(self):
        """Create visualizations for model evaluation"""
        print("\nCreating visualizations...")
        
        # Predictions vs Actual
        y_pred = self.best_model.predict(self.X_test)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Visualizations', fontsize=16, fontweight='bold')
        
        # 1. Predictions vs Actual scatter plot
        ax1.scatter(self.y_test, y_pred, alpha=0.6, color='blue')
        ax1.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Amount ($)')
        ax1.set_ylabel('Predicted Amount ($)')
        ax1.set_title('Predictions vs Actual')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = self.y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Amount ($)')
        ax2.set_ylabel('Residuals ($)')
        ax2.set_title('Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature importance (if available)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            ax3.barh(range(len(top_features)), top_features['importance'], color='orange')
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'])
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 10 Feature Importance')
        
        # 4. Error distribution
        ax4.hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Residuals ($)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def make_predictions(self, sample_transactions=None):
        """Make predictions on new transactions"""
        print("\nMaking predictions on sample transactions...")
        
        if sample_transactions is None:
            # Use some test data as examples
            sample_indices = self.X_test.index[:5]
            sample_features = self.X_test.iloc[:5]
            sample_actual = self.y_test.iloc[:5]
        else:
            sample_features = sample_transactions
            sample_actual = None
        
        predictions = self.best_model.predict(sample_features)
        
        print("Sample Predictions:")
        print("-" * 60)
        for i, pred in enumerate(predictions):
            if sample_actual is not None:
                actual = sample_actual.iloc[i]
                error = abs(pred - actual)
                print(f"Transaction {i+1}: Predicted ${pred:.2f}, Actual ${actual:.2f}, Error ${error:.2f}")
            else:
                print(f"Transaction {i+1}: Predicted ${pred:.2f}")
        
        return predictions
    
    def run_complete_analysis(self):
        """Run the complete machine learning analysis"""
        print("="*80)
        print("BANK STATEMENT REGRESSION MODEL - ASSIGNMENT 1.3")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        self.create_features()
        self.prepare_ml_data()
        
        # Train models
        model_scores = self.train_models()
        
        # Prevent overfitting
        cv_scores = self.prevent_overfitting()
        
        # Select best model
        model_performance = self.select_best_model()
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Create visualizations
        self.create_visualizations()
        
        # Make sample predictions
        self.make_predictions()
        
        print("\n" + "="*80)
        print("REGRESSION MODEL ANALYSIS COMPLETE!")
        print("="*80)
        print("Key Results:")
        print(f"✓ Multiple models trained and evaluated")
        print(f"✓ Cross-validation performed to prevent overfitting")
        print(f"✓ Hyperparameter tuning completed")
        print(f"✓ Feature importance analyzed")
        print(f"✓ Model performance visualized")
        print("="*80)

def main():
    """Main execution function"""
    predictor = TransactionPredictor()
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main()

