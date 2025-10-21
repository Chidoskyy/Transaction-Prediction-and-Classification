#!/usr/bin/env python3
"""
Bank Statement Model Visualization UI
Interactive web interface for regression and classification models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our models
from regression_model import TransactionPredictor
from classification_model import TransactionClassifier

# Page configuration
st.set_page_config(
    page_title="Bank Statement ML Models",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load and cache the trained models"""
    try:
        # Load regression model
        reg_predictor = TransactionPredictor()
        reg_predictor.load_and_prepare_data()
        reg_predictor.create_features()
        reg_predictor.prepare_ml_data()
        
        # Train a simple model for demonstration
        from sklearn.ensemble import RandomForestRegressor
        reg_model = RandomForestRegressor(n_estimators=50, random_state=42)
        reg_model.fit(reg_predictor.X_train, reg_predictor.y_train)
        
        # Load classification model
        class_classifier = TransactionClassifier()
        class_classifier.load_and_prepare_data()
        class_classifier.create_text_features()
        class_classifier.create_numerical_features()
        class_classifier.prepare_ml_data()
        
        # Train a simple model for demonstration
        from sklearn.ensemble import RandomForestClassifier
        class_model = RandomForestClassifier(n_estimators=50, random_state=42)
        class_model.fit(class_classifier.X_train, class_classifier.y_train)
        
        return {
            'regression': {
                'predictor': reg_predictor,
                'model': reg_model
            },
            'classification': {
                'classifier': class_classifier,
                'model': class_model
            }
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def create_feature_input_form(model_type, models):
    """Create input form for model predictions"""
    st.subheader(f"🔮 {model_type.title()} Model Prediction")
    
    if model_type == "regression":
        predictor = models['regression']['predictor']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date inputs
            transaction_date = st.date_input(
                "Transaction Date",
                value=datetime.now().date(),
                help="Date when the transaction occurred"
            )
            
            transaction_postdate = st.date_input(
                "Transaction Post Date",
                value=datetime.now().date() + timedelta(days=1),
                help="Date when the transaction was posted"
            )
            
            # Amount input
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                max_value=10000.0,
                value=25.0,
                step=0.01,
                help="Amount of the transaction"
            )
            
        with col2:
            # Category selection
            categories = ['Retail and Grocery', 'Professional and Financial Services', 
                         'Restaurants', 'Health and Education', 'Personal and Household Expenses']
            spend_category = st.selectbox(
                "Spend Category",
                categories,
                help="Category of the transaction"
            )
            
            # Source file selection
            source_files = ['onlineStatement (13).pdf', 'onlineStatement (14).pdf', 
                           'onlineStatement (15).pdf', 'onlineStatement (16).pdf']
            source_file = st.selectbox(
                "Source File",
                source_files,
                help="Source statement file"
            )
            
            # Description input
            transaction_description = st.text_input(
                "Transaction Description",
                value="TIM HORTONS WINNIPEG MB",
                help="Description of the transaction"
            )
        
        # Prediction button
        if st.button("🔮 Predict Amount", type="primary"):
            try:
                # Create feature vector for prediction
                features = create_regression_features(
                    transaction_date, transaction_postdate, transaction_description,
                    spend_category, amount, source_file, predictor
                )
                
                # Make prediction
                prediction = models['regression']['model'].predict([features])[0]
                
                # Display results
                st.success(f"**Predicted Amount: ${prediction:.2f}**")
                
                # Show confidence (for regression, we'll use feature importance)
                if hasattr(models['regression']['model'], 'feature_importances_'):
                    importance = models['regression']['model'].feature_importances_
                    max_importance = np.max(importance)
                    confidence = min(max_importance * 100, 95)  # Cap at 95%
                    st.info(f"Model Confidence: {confidence:.1f}%")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    else:  # Classification
        classifier = models['classification']['classifier']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date inputs
            transaction_date = st.date_input(
                "Transaction Date",
                value=datetime.now().date(),
                help="Date when the transaction occurred"
            )
            
            transaction_postdate = st.date_input(
                "Transaction Post Date",
                value=datetime.now().date() + timedelta(days=1),
                help="Date when the transaction was posted"
            )
            
            # Amount input
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                max_value=10000.0,
                value=25.0,
                step=0.01,
                help="Amount of the transaction"
            )
            
        with col2:
            # Source file selection
            source_files = ['onlineStatement (13).pdf', 'onlineStatement (14).pdf', 
                           'onlineStatement (15).pdf', 'onlineStatement (16).pdf']
            source_file = st.selectbox(
                "Source File",
                source_files,
                help="Source statement file"
            )
            
            # Description input
            transaction_description = st.text_input(
                "Transaction Description",
                value="TIM HORTONS WINNIPEG MB",
                help="Description of the transaction"
            )
        
        # Prediction button
        if st.button("🏷️ Predict Category", type="primary"):
            try:
                # Create feature vector for prediction
                features = create_classification_features(
                    transaction_date, transaction_postdate, transaction_description,
                    amount, source_file, classifier
                )
                
                # Make prediction
                prediction = models['classification']['model'].predict(features)[0]
                prediction_proba = models['classification']['model'].predict_proba(features)[0]
                
                # Get category name
                predicted_category = classifier.label_encoder.classes_[prediction]
                confidence = prediction_proba[prediction] * 100
                
                # Display results
                st.success(f"**Predicted Category: {predicted_category}**")
                st.info(f"Confidence: {confidence:.1f}%")
                
                # Show all probabilities
                prob_df = pd.DataFrame({
                    'Category': classifier.label_encoder.classes_,
                    'Probability (%)': prediction_proba * 100
                }).sort_values('Probability (%)', ascending=False)
                
                st.subheader("All Category Probabilities:")
                st.dataframe(prob_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")

def create_regression_features(date, postdate, description, category, amount, source, predictor):
    """Create feature vector for regression prediction"""
    # This is a simplified version - in practice, you'd need to replicate the full feature engineering
    features = np.zeros(len(predictor.X_train.columns))
    
    # Basic features (simplified)
    features[0] = date.weekday()  # day_of_week
    features[1] = date.day  # day_of_month
    features[2] = date.month  # month
    features[3] = date.year  # year
    features[4] = 1 if date.weekday() >= 5 else 0  # is_weekend
    features[5] = len(description)  # description_length
    features[6] = len(description.split())  # word_count
    features[7] = amount  # amount
    features[8] = abs(amount)  # amount_abs
    features[9] = np.log1p(abs(amount))  # amount_log
    
    # Add more features as needed...
    return features

def create_classification_features(date, postdate, description, amount, source, classifier):
    """Create feature vector for classification prediction"""
    # This is a simplified version - in practice, you'd need to replicate the full feature engineering
    features = np.zeros(classifier.X_train.shape[1])
    
    # Basic features (simplified)
    features[0] = date.weekday()  # day_of_week
    features[1] = date.day  # day_of_month
    features[2] = date.month  # month
    features[3] = date.year  # year
    features[4] = 1 if date.weekday() >= 5 else 0  # is_weekend
    features[5] = len(description)  # description_length
    features[6] = len(description.split())  # word_count
    features[7] = amount  # amount
    features[8] = abs(amount)  # amount_abs
    features[9] = np.log1p(abs(amount))  # amount_log
    
    # Add more features as needed...
    return features.reshape(1, -1)

def create_model_performance_dashboard(models):
    """Create model performance visualizations"""
    st.subheader("📊 Model Performance Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔄 Regression Model Performance")
        
        # Get regression model predictions
        reg_model = models['regression']['model']
        reg_predictor = models['regression']['predictor']
        
        y_pred = reg_model.predict(reg_predictor.X_test)
        y_true = reg_predictor.y_test
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Display metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("RMSE", f"${rmse:.2f}")
        with metric_col2:
            st.metric("MAE", f"${mae:.2f}")
            st.metric("MSE", f"{mse:.2f}")
        
        # Prediction vs Actual plot
        fig = px.scatter(
            x=y_true, y=y_pred,
            title="Predicted vs Actual Amounts",
            labels={'x': 'Actual Amount ($)', 'y': 'Predicted Amount ($)'}
        )
        fig.add_trace(go.Scatter(
            x=[y_true.min(), y_true.max()],
            y=[y_true.min(), y_true.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏷️ Classification Model Performance")
        
        # Get classification model predictions
        class_model = models['classification']['model']
        class_classifier = models['classification']['classifier']
        
        y_pred = class_model.predict(class_classifier.X_test)
        y_true = class_classifier.y_test
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Display metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("Precision", f"{precision:.3f}")
        with metric_col2:
            st.metric("Recall", f"{recall:.3f}")
            st.metric("F1 Score", f"{f1:.3f}")
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=class_classifier.label_encoder.classes_,
            y=class_classifier.label_encoder.classes_
        )
        st.plotly_chart(fig, use_container_width=True)

def create_feature_importance_visualization(models):
    """Create feature importance visualizations"""
    st.subheader("🔍 Feature Importance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔄 Regression Model Feature Importance")
        
        reg_model = models['regression']['model']
        reg_predictor = models['regression']['predictor']
        
        if hasattr(reg_model, 'feature_importances_'):
            # Get top 15 features
            importance = reg_model.feature_importances_
            feature_names = reg_predictor.X_train.columns
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True).tail(15)
            
            # Create horizontal bar chart
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features",
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    
    with col2:
        st.markdown("### 🏷️ Classification Model Feature Importance")
        
        class_model = models['classification']['model']
        class_classifier = models['classification']['classifier']
        
        if hasattr(class_model, 'feature_importances_'):
            # Get top 15 features
            importance = class_model.feature_importances_
            
            # Get feature names (combine text and numerical features)
            text_features = list(class_classifier.vectorizer.get_feature_names_out())
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
            all_feature_names = text_features + numerical_features
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True).tail(15)
            
            # Create horizontal bar chart
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features",
                color='Importance',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

def create_data_exploration_dashboard():
    """Create data exploration dashboard"""
    st.subheader("📈 Data Exploration Dashboard")
    
    # Load original data
    try:
        with open('extracted_transactions.json', 'r') as f:
            transactions = json.load(f)
        
        df = pd.DataFrame(transactions)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['transaction_postdate'] = pd.to_datetime(df['transaction_postdate'])
        
        # Remove payments for analysis
        df_spending = df[df['spend_category'] != 'PAYMENT'].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Spending by category
            st.markdown("### 💰 Spending by Category")
            category_spending = df_spending.groupby('spend_category')['amount'].sum().sort_values(ascending=True)
            
            fig = px.bar(
                x=category_spending.values,
                y=category_spending.index,
                orientation='h',
                title="Total Spending by Category",
                labels={'x': 'Total Amount ($)', 'y': 'Category'},
                color=category_spending.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Transaction count by category
            st.markdown("### 📊 Transaction Count by Category")
            category_count = df_spending['spend_category'].value_counts()
            
            fig = px.pie(
                values=category_count.values,
                names=category_count.index,
                title="Transaction Distribution by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Spending over time
            st.markdown("### 📅 Spending Over Time")
            df_spending['month'] = df_spending['transaction_date'].dt.to_period('M')
            monthly_spending = df_spending.groupby('month')['amount'].sum()
            
            fig = px.line(
                x=monthly_spending.index.astype(str),
                y=monthly_spending.values,
                title="Monthly Spending Trend",
                labels={'x': 'Month', 'y': 'Total Amount ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Average transaction amount by category
            st.markdown("### 💵 Average Transaction Amount by Category")
            avg_amount = df_spending.groupby('spend_category')['amount'].mean().sort_values(ascending=True)
            
            fig = px.bar(
                x=avg_amount.values,
                y=avg_amount.index,
                orientation='h',
                title="Average Transaction Amount by Category",
                labels={'x': 'Average Amount ($)', 'y': 'Category'},
                color=avg_amount.values,
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("### 📋 Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(df_spending))
        with col2:
            st.metric("Total Spending", f"${df_spending['amount'].sum():.2f}")
        with col3:
            st.metric("Average Transaction", f"${df_spending['amount'].mean():.2f}")
        with col4:
            st.metric("Date Range", f"{df_spending['transaction_date'].min().strftime('%Y-%m-%d')} to {df_spending['transaction_date'].max().strftime('%Y-%m-%d')}")
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🏦 Bank Statement ML Models Dashboard</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check your data files.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🔮 Model Predictions", "📊 Performance Dashboard", "🔍 Feature Importance", "📈 Data Exploration"]
    )
    
    # Main content based on selection
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Bank Statement ML Models Dashboard! 🎉
        
        This interactive dashboard allows you to explore and interact with your machine learning models for bank statement analysis.
        
        ### 🚀 Features Available:
        
        **🔮 Model Predictions**
        - Interactive prediction interface for both regression and classification models
        - Real-time predictions with confidence scores
        - Input forms for testing new transactions
        
        **📊 Performance Dashboard**
        - Model performance metrics and visualizations
        - Prediction accuracy analysis
        - Confusion matrices and regression plots
        
        **🔍 Feature Importance**
        - Top features driving model predictions
        - Feature importance rankings
        - Insights into what the models learn
        
        **📈 Data Exploration**
        - Interactive data visualizations
        - Spending patterns and trends
        - Category analysis and statistics
        
        ### 🎯 Models Available:
        - **Regression Model**: Predicts transaction amounts (67 features)
        - **Classification Model**: Categorizes transactions (209 features)
        
        ### 📊 Dataset Overview:
        - **Total Transactions**: 67
        - **Spending Transactions**: 62 (excluding payments)
        - **Categories**: 5 spending categories
        - **Date Range**: August 2020 to May 2021
        - **Source Files**: 4 statement PDFs
        
        Navigate using the sidebar to explore different aspects of your models!
        """)
        
        # Quick stats
        st.markdown("### 📊 Quick Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Regression Features", "67")
        with col2:
            st.metric("Classification Features", "209")
        with col3:
            st.metric("Training Samples", "49")
    
    elif page == "🔮 Model Predictions":
        st.markdown("## 🔮 Interactive Model Predictions")
        
        # Model selection
        model_type = st.radio(
            "Select Model Type",
            ["Regression", "Classification"],
            horizontal=True
        )
        
        create_feature_input_form(model_type.lower(), models)
    
    elif page == "📊 Performance Dashboard":
        create_model_performance_dashboard(models)
    
    elif page == "🔍 Feature Importance":
        create_feature_importance_visualization(models)
    
    elif page == "📈 Data Exploration":
        create_data_exploration_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏦 Bank Statement ML Models Dashboard | Task 1.5 - Comprehensive Feature Engineering</p>
        <p>Built with Streamlit, Plotly, and scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
