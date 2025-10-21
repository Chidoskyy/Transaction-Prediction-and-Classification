#!/usr/bin/env python3
"""
Bank Statement Classification Model - Assignment 1.4
Machine Learning model to categorize transaction descriptions (no overfitting)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import re

class TransactionClassifier:
    def __init__(self, data_file: str = 'extracted_transactions.json'):
        """Initialize the transaction classifier"""
        self.data_file = data_file
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.label_encoder = LabelEncoder()
        self.vectorizer = None
        
    def load_and_prepare_data(self):
        """Load and prepare transaction data for classification"""
        print("Loading transaction data for classification...")
        
        # Load data
        with open(self.data_file, 'r') as f:
            transactions = json.load(f)
        
        self.df = pd.DataFrame(transactions)
        print(f"Loaded {len(self.df)} transactions")
        
        # Convert dates
        self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
        self.df['transaction_postdate'] = pd.to_datetime(self.df['transaction_postdate'])
        
        # Remove payments (we want to classify spending transactions)
        self.df = self.df[self.df['spend_category'] != 'PAYMENT'].copy()
        print(f"After removing payments: {len(self.df)} transactions")
        
        # Check category distribution
        print("\nCategory distribution:")
        category_counts = self.df['spend_category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count} transactions")
        
        return self.df
    
    def preprocess_text(self, text):
        """Preprocess transaction description text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def create_text_features(self):
        """Create comprehensive text-based features for classification using ALL database features"""
        print("Creating comprehensive text features...")
        
        # Preprocess transaction descriptions
        self.df['processed_description'] = self.df['transaction_description'].apply(self.preprocess_text)
        
        # ===== COMPREHENSIVE TEXT FEATURES =====
        print("  Creating comprehensive text features...")
        
        # Basic text features
        self.df['description_length'] = self.df['processed_description'].str.len()
        self.df['word_count'] = self.df['processed_description'].str.split().str.len()
        self.df['char_count_no_spaces'] = self.df['processed_description'].str.replace(' ', '').str.len()
        self.df['avg_word_length'] = self.df['description_length'] / (self.df['word_count'] + 1)
        
        # Text pattern features
        self.df['has_numbers'] = self.df['processed_description'].str.contains(r'\d+', na=False).astype(int)
        self.df['has_special_chars'] = self.df['processed_description'].str.contains(r'[^a-zA-Z0-9\s]', na=False).astype(int)
        self.df['has_forward_slash'] = self.df['processed_description'].str.contains('/', na=False).astype(int)
        self.df['has_hash'] = self.df['processed_description'].str.contains('#', na=False).astype(int)
        self.df['has_dollar_sign'] = self.df['processed_description'].str.contains('\$', na=False).astype(int)
        
        # Location features (Canadian provinces)
        canadian_provinces = ['mb', 'on', 'ab', 'bc', 'sk', 'qc', 'ns', 'nb', 'nl', 'pe', 'yt', 'nt', 'nu']
        self.df['has_province'] = self.df['processed_description'].str.contains('|'.join(canadian_provinces), na=False).astype(int)
        self.df['has_city'] = self.df['processed_description'].str.contains('winnipeg|toronto|vancouver|montreal|calgary', na=False).astype(int)
        
        # Expanded merchant keywords
        merchant_keywords = {
            'grocery': ['superstore', 'sobeys', 'grocery', 'market', 'food', 'fresh', 'safeway', 'metro'],
            'restaurant': ['tim hortons', 'starbucks', 'restaurant', 'cafe', 'coffee', 'pizza', 'subway', 'mcdonald'],
            'gas': ['gas', 'fuel', 'shell', 'esso', 'petro', 'station', 'mobil', 'chevron'],
            'pharmacy': ['shoppers', 'pharmacy', 'drug', 'medical', 'health', 'clinic'],
            'retail': ['store', 'shop', 'retail', 'walmart', 'amazon', 'best buy', 'canadian tire'],
            'service': ['service', 'centre', 'center', 'office', 'clinic', 'repair', 'maintenance'],
            'entertainment': ['cinema', 'movie', 'theatre', 'theater', 'entertainment', 'game'],
            'transportation': ['taxi', 'uber', 'lyft', 'bus', 'train', 'airport', 'parking'],
            'utilities': ['hydro', 'electric', 'water', 'gas', 'internet', 'phone', 'cable'],
            'financial': ['bank', 'atm', 'withdrawal', 'deposit', 'transfer', 'payment']
        }
        
        for category, keywords in merchant_keywords.items():
            pattern = '|'.join(keywords)
            self.df[f'has_{category}'] = self.df['processed_description'].str.contains(pattern, na=False).astype(int)
        
        # Payment-specific features
        self.df['is_payment'] = self.df['processed_description'].str.contains('payment|paiement', na=False).astype(int)
        self.df['is_thank_you'] = self.df['processed_description'].str.contains('thank you|merci', na=False).astype(int)
        
        print(f"Created {len([col for col in self.df.columns if col.startswith('has_')])} keyword features")
        
        return self.df
    
    def create_numerical_features(self):
        """Create comprehensive numerical features for classification using ALL database features"""
        print("Creating comprehensive numerical features...")
        
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
        
        # ===== INTERACTION FEATURES =====
        print("  Creating interaction features...")
        
        # Date-amount interactions
        self.df['weekend_amount'] = self.df['is_weekend'] * self.df['amount']
        self.df['month_end_amount'] = self.df['is_month_end'] * self.df['amount']
        
        # Description-amount interactions
        self.df['description_length_amount'] = self.df['description_length'] * self.df['amount']
        
        print(f"Created {len(self.df.columns)} total comprehensive features")
        
        return self.df
    
    def prepare_ml_data(self):
        """Prepare comprehensive ML data for classification using ALL features"""
        print("Preparing comprehensive ML data for classification...")
        
        # Define comprehensive feature columns (excluding target and original columns)
        exclude_columns = [
            'transaction_date', 'transaction_postdate', 'transaction_description', 
            'spend_category', 'amount', 'source_file', 'processed_description'
        ]
        
        # Get all feature columns (everything except excluded columns)
        all_feature_columns = [col for col in self.df.columns if col not in exclude_columns]
        
        # Separate text and numerical features
        text_features = ['processed_description']
        numerical_features = [col for col in all_feature_columns if col != 'processed_description']
        
        print(f"Using {len(all_feature_columns)} total features for classification:")
        print(f"  Text features: {len(text_features)}")
        print(f"  Numerical features: {len(numerical_features)}")
        print(f"  Date features: {len([col for col in numerical_features if any(x in col for x in ['day', 'month', 'year', 'week', 'quarter', 'weekend', 'processing'])])}")
        print(f"  Text pattern features: {len([col for col in numerical_features if any(x in col for x in ['description', 'word', 'char', 'has_', 'is_'])])}")
        print(f"  Amount features: {len([col for col in numerical_features if 'amount' in col])}")
        print(f"  Source features: {len([col for col in numerical_features if 'source' in col])}")
        print(f"  Interaction features: {len([col for col in numerical_features if 'interaction' in col or col.endswith('_amount')])}")
        
        # Prepare features
        X_text = self.df[text_features].copy()
        X_numerical = self.df[numerical_features].copy()
        
        # Prepare target
        y = self.df['spend_category'].copy()
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Text features shape: {X_text.shape}")
        print(f"Numerical features shape: {X_numerical.shape}")
        print(f"Target shape: {y_encoded.shape}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Split data
        X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
            X_text, X_numerical, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Vectorize text features with enhanced parameters
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased from 1000
            ngram_range=(1, 3),  # Increased from (1, 2)
            stop_words='english',
            min_df=1,  # Reduced from 2 to capture more patterns
            max_df=0.9,  # Reduced from 0.95
            lowercase=True,
            strip_accents='unicode'
        )
        
        X_text_train_vec = self.vectorizer.fit_transform(X_text_train['processed_description'])
        X_text_test_vec = self.vectorizer.transform(X_text_test['processed_description'])
        
        # Handle any missing values in numerical features
        X_num_train = X_num_train.fillna(0)
        X_num_test = X_num_test.fillna(0)
        
        # Handle infinite values
        X_num_train = X_num_train.replace([np.inf, -np.inf], 0)
        X_num_test = X_num_test.replace([np.inf, -np.inf], 0)
        
        # Convert all columns to numeric types to avoid object dtype issues
        for col in X_num_train.columns:
            X_num_train[col] = pd.to_numeric(X_num_train[col], errors='coerce')
            X_num_test[col] = pd.to_numeric(X_num_test[col], errors='coerce')
        
        # Fill any NaN values created by conversion
        X_num_train = X_num_train.fillna(0)
        X_num_test = X_num_test.fillna(0)
        
        # Ensure all data is float64 for scipy sparse compatibility
        X_num_train = X_num_train.astype(np.float64)
        X_num_test = X_num_test.astype(np.float64)
        
        # Combine text and numerical features
        from scipy.sparse import hstack
        
        X_train = hstack([X_text_train_vec, X_num_train.values])
        X_test = hstack([X_text_test_vec, X_num_test.values])
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Final training set: {X_train.shape}")
        print(f"Final test set: {X_test.shape}")
        print(f"Text features: {X_text_train_vec.shape[1]}")
        print(f"Numerical features: {X_num_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_classification_models(self):
        """Train multiple classification models"""
        print("Training classification models...")
        
        # Define models to try
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Naive Bayes': MultinomialNB()
        }
        
        # Train and evaluate each model
        model_scores = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                
                # Calculate metrics
                train_accuracy = accuracy_score(self.y_train, y_pred_train)
                test_accuracy = accuracy_score(self.y_test, y_pred_test)
                train_f1 = f1_score(self.y_train, y_pred_train, average='weighted')
                test_f1 = f1_score(self.y_test, y_pred_test, average='weighted')
                
                # Store model and scores
                self.models[name] = model
                model_scores[name] = {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_f1': train_f1,
                    'test_f1': test_f1,
                    'overfitting': train_accuracy - test_accuracy
                }
                
                print(f"  Train Accuracy: {train_accuracy:.4f}")
                print(f"  Test Accuracy: {test_accuracy:.4f}")
                print(f"  Train F1: {train_f1:.4f}")
                print(f"  Test F1: {test_f1:.4f}")
                print(f"  Overfitting (Acc diff): {train_accuracy - test_accuracy:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
        
        return model_scores
    
    def prevent_overfitting_classification(self):
        """Implement techniques to prevent overfitting in classification"""
        print("\nImplementing overfitting prevention for classification...")
        
        # Cross-validation for best model selection
        print("Performing cross-validation...")
        
        cv_scores = {}
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
                cv_scores[name] = {
                    'mean_cv_score': scores.mean(),
                    'std_cv_score': scores.std(),
                    'scores': scores
                }
                print(f"{name}: CV Accuracy = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                print(f"Error in CV for {name}: {e}")
        
        # Hyperparameter tuning for best models
        print("\nPerforming hyperparameter tuning...")
        
        # Random Forest tuning
        try:
            rf_params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            
            rf_grid = GridSearchCV(
                RandomForestClassifier(random_state=42),
                rf_params,
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            rf_grid.fit(self.X_train, self.y_train)
            
            print(f"Best Random Forest params: {rf_grid.best_params_}")
            print(f"Best Random Forest CV score: {rf_grid.best_score_:.4f}")
            
            # Update model with best parameters
            self.models['Random Forest (Tuned)'] = rf_grid.best_estimator_
            
        except Exception as e:
            print(f"Error in Random Forest tuning: {e}")
        
        # Logistic Regression tuning
        try:
            lr_params = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            lr_grid = GridSearchCV(
                LogisticRegression(random_state=42, max_iter=1000),
                lr_params,
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            lr_grid.fit(self.X_train, self.y_train)
            
            print(f"Best Logistic Regression params: {lr_grid.best_params_}")
            print(f"Best Logistic Regression CV score: {lr_grid.best_score_:.4f}")
            
            # Update model with best parameters
            self.models['Logistic Regression (Tuned)'] = lr_grid.best_estimator_
            
        except Exception as e:
            print(f"Error in Logistic Regression tuning: {e}")
        
        return cv_scores
    
    def select_best_classification_model(self):
        """Select the best classification model"""
        print("\nSelecting best classification model...")
        
        # Evaluate all models on test set
        model_performance = {}
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(self.X_test)
                
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                # Calculate overfitting
                y_pred_train = model.predict(self.X_train)
                train_accuracy = accuracy_score(self.y_train, y_pred_train)
                overfitting = train_accuracy - accuracy
                
                model_performance[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'overfitting': overfitting
                }
                
                print(f"{name}:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1: {f1:.4f}")
                print(f"  Overfitting: {overfitting:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        # Select best model (highest F1 with low overfitting)
        if model_performance:
            best_model_name = max(model_performance.keys(), 
                                key=lambda x: model_performance[x]['f1'] - model_performance[x]['overfitting'])
            
            self.best_model = self.models[best_model_name]
            print(f"\nBest model: {best_model_name}")
        
        return model_performance
    
    def analyze_classification_results(self):
        """Analyze classification results in detail"""
        print("\nAnalyzing classification results...")
        
        if self.best_model is None:
            print("No best model available for analysis")
            return
        
        # Get predictions
        y_pred = self.best_model.predict(self.X_test)
        
        # Classification report
        print("\nDetailed Classification Report:")
        print("-" * 60)
        target_names = self.label_encoder.classes_
        report = classification_report(self.y_test, y_pred, target_names=target_names)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Create visualizations
        self.create_classification_visualizations(y_pred, cm)
        
        return y_pred, cm
    
    def create_classification_visualizations(self, y_pred, cm):
        """Create visualizations for classification results"""
        print("\nCreating classification visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Classification Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_, ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. Class distribution
        class_counts = pd.Series(self.y_test).value_counts()
        ax2.bar(range(len(class_counts)), class_counts.values, color='skyblue')
        ax2.set_xticks(range(len(class_counts)))
        ax2.set_xticklabels([self.label_encoder.classes_[i] for i in class_counts.index], rotation=45)
        ax2.set_title('Test Set Class Distribution')
        ax2.set_ylabel('Count')
        
        # 3. Prediction accuracy by class
        class_accuracy = []
        for i in range(len(self.label_encoder.classes_)):
            mask = self.y_test == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == i).sum() / mask.sum()
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)
        
        ax3.bar(range(len(class_accuracy)), class_accuracy, color='lightgreen')
        ax3.set_xticks(range(len(class_accuracy)))
        ax3.set_xticklabels(self.label_encoder.classes_, rotation=45)
        ax3.set_title('Accuracy by Class')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        
        # 4. Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            # Get feature names
            feature_names = list(self.vectorizer.get_feature_names_out())
            numerical_features = [
                'description_length', 'word_count', 'char_count_no_spaces', 'avg_word_length',
                'has_numbers', 'has_special_chars', 'has_forward_slash', 'has_hash', 'has_dollar_sign',
                'has_province', 'has_city', 'has_grocery', 'has_restaurant', 'has_gas', 'has_pharmacy',
                'has_retail', 'has_service', 'has_entertainment', 'has_transportation', 'has_utilities',
                'has_financial', 'is_payment', 'is_thank_you', 'day_of_week', 'day_of_month', 'month',
                'year', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end', 'post_day_of_week',
                'post_day_of_month', 'post_month', 'post_year', 'processing_delay_days', 'is_same_day_post',
                'is_next_day_post', 'days_since_start', 'days_since_year_start', 'week_of_year',
                'amount_abs', 'amount_log', 'amount_sqrt', 'amount_rounded', 'amount_decimal',
                'is_small_amount', 'is_medium_amount', 'is_large_amount', 'amount_percentile',
                'amount_rolling_mean_7', 'amount_rolling_std_7', 'source_encoded', 'source_month',
                'source_frequency', 'weekend_amount', 'month_end_amount', 'description_length_amount'
            ]
            all_feature_names = feature_names + numerical_features
            
            importance = self.best_model.feature_importances_
            top_indices = np.argsort(importance)[-15:]  # Top 15 features
            
            ax4.barh(range(len(top_indices)), importance[top_indices], color='orange')
            ax4.set_yticks(range(len(top_indices)))
            ax4.set_yticklabels([all_feature_names[i] for i in top_indices])
            ax4.set_title('Top 15 Feature Importance')
            ax4.set_xlabel('Importance')
        
        plt.tight_layout()
        plt.show()
    
    def make_classification_predictions(self, sample_descriptions=None):
        """Make predictions on new transaction descriptions"""
        print("\nMaking classification predictions...")
        
        if sample_descriptions is None:
            # Use some test data as examples
            sample_indices = np.random.choice(len(self.X_test), 5, replace=False)
            sample_features = self.X_test[sample_indices]
            sample_actual = self.y_test[sample_indices]
        else:
            # Process new descriptions
            processed_samples = [self.preprocess_text(desc) for desc in sample_descriptions]
            sample_features = self.vectorizer.transform(processed_samples)
            sample_actual = None
        
        predictions = self.best_model.predict(sample_features)
        prediction_probs = self.best_model.predict_proba(sample_features)
        
        print("Sample Predictions:")
        print("-" * 60)
        for i, pred in enumerate(predictions):
            predicted_class = self.label_encoder.classes_[pred]
            confidence = prediction_probs[i][pred]
            
            if sample_actual is not None:
                actual_class = self.label_encoder.classes_[sample_actual[i]]
                correct = "✓" if pred == sample_actual[i] else "✗"
                print(f"Transaction {i+1}: {correct} Predicted: {predicted_class} (Confidence: {confidence:.3f}), Actual: {actual_class}")
            else:
                print(f"Transaction {i+1}: Predicted: {predicted_class} (Confidence: {confidence:.3f})")
        
        return predictions, prediction_probs
    
    def run_complete_classification_analysis(self):
        """Run the complete classification analysis"""
        print("="*80)
        print("BANK STATEMENT CLASSIFICATION MODEL - ASSIGNMENT 1.4")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        self.create_text_features()
        self.create_numerical_features()
        self.prepare_ml_data()
        
        # Train models
        model_scores = self.train_classification_models()
        
        # Prevent overfitting
        cv_scores = self.prevent_overfitting_classification()
        
        # Select best model
        model_performance = self.select_best_classification_model()
        
        # Analyze results
        y_pred, cm = self.analyze_classification_results()
        
        # Make sample predictions
        self.make_classification_predictions()
        
        print("\n" + "="*80)
        print("CLASSIFICATION MODEL ANALYSIS COMPLETE!")
        print("="*80)
        print("Key Results:")
        print(f"✓ Multiple classification models trained and evaluated")
        print(f"✓ Text features extracted using TF-IDF vectorization")
        print(f"✓ Cross-validation performed to prevent overfitting")
        print(f"✓ Hyperparameter tuning completed")
        print(f"✓ Classification performance analyzed")
        print(f"✓ Confusion matrix and visualizations created")
        print("="*80)

def main():
    """Main execution function"""
    classifier = TransactionClassifier()
    classifier.run_complete_classification_analysis()

if __name__ == "__main__":
    main()

