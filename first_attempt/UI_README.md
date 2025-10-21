# 🏦 Bank Statement ML Models UI

An interactive web-based dashboard for visualizing and interacting with your bank statement machine learning models.

## 🚀 Quick Start

### Option 1: Using the Launcher (Recommended)
```bash
python launch_ui.py
```

### Option 2: Manual Installation
```bash
# Install requirements
pip install -r requirements_ui.txt

# Launch the UI
streamlit run model_ui.py
```

The UI will open in your browser at `http://localhost:8501`

## 🎯 Features

### 🔮 Interactive Model Predictions
- **Regression Model**: Predict transaction amounts with 67 comprehensive features
- **Classification Model**: Categorize transactions with 209 features
- Real-time predictions with confidence scores
- Interactive input forms for testing new transactions

### 📊 Performance Dashboard
- Model performance metrics (R², RMSE, MAE for regression; Accuracy, Precision, Recall, F1 for classification)
- Prediction vs actual scatter plots
- Confusion matrices for classification
- Visual performance comparisons

### 🔍 Feature Importance Analysis
- Top 15 most important features for each model
- Interactive bar charts showing feature importance
- Insights into what drives model predictions
- Separate analysis for regression and classification models

### 📈 Data Exploration Dashboard
- Spending patterns by category
- Monthly spending trends
- Transaction distribution analysis
- Average transaction amounts by category
- Interactive charts and visualizations

## 🏗️ Architecture

### Models Used
- **Regression Model**: Random Forest Regressor with 67 features
- **Classification Model**: Random Forest Classifier with 209 features (151 text + 58 numerical)

### Technologies
- **Frontend**: Streamlit (Python web framework)
- **Visualizations**: Plotly (interactive charts)
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy

### Feature Engineering
The UI showcases the comprehensive feature engineering from Task 1.5:

**Date Features (21 features)**
- Day of week, month, year, quarter
- Weekend indicators, month start/end
- Processing delay features

**Text Features (151 features)**
- TF-IDF vectorization of transaction descriptions
- 1-3 gram analysis
- Merchant type pattern matching

**Numerical Features (58 features)**
- Amount transformations (log, sqrt, percentile)
- Category and source encodings
- Interaction features
- Frequency features

## 📱 UI Pages

### 🏠 Home
- Welcome page with overview
- Quick statistics
- Navigation guide

### 🔮 Model Predictions
- Interactive prediction interface
- Input forms for both models
- Real-time results with confidence scores

### 📊 Performance Dashboard
- Model performance metrics
- Visual performance analysis
- Comparison charts

### 🔍 Feature Importance
- Feature importance rankings
- Interactive importance charts
- Model insights

### 📈 Data Exploration
- Data visualization dashboard
- Spending pattern analysis
- Category and trend analysis

## 🛠️ Customization

### Adding New Features
1. Modify the feature creation functions in `model_ui.py`
2. Update the input forms in `create_feature_input_form()`
3. Adjust the feature importance visualization

### Styling
- Custom CSS in the main function
- Plotly chart customization
- Streamlit theme configuration

### Model Integration
- Replace the simple models with your trained models
- Update the model loading in `load_models()`
- Modify prediction functions as needed

## 📊 Sample Data

The UI works with your bank statement data:
- **67 total transactions**
- **62 spending transactions** (excluding payments)
- **5 spending categories**
- **4 source statement files**
- **Date range**: August 2020 to May 2021

## 🔧 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
streamlit run model_ui.py --server.port 8502
```

**Missing Dependencies**
```bash
pip install -r requirements_ui.txt
```

**Model Loading Errors**
- Ensure `extracted_transactions.json` exists
- Check that all model files are present
- Verify data format compatibility

### Performance Tips
- The UI caches models for better performance
- Large datasets may require longer loading times
- Consider reducing feature complexity for faster predictions

## 🎨 UI Screenshots

The UI provides:
- Clean, modern interface with banking theme
- Interactive charts and visualizations
- Responsive design for different screen sizes
- Intuitive navigation and user experience

## 📝 Notes

- The UI demonstrates Task 1.5's comprehensive feature engineering
- All 6 original database fields are utilized
- Multiple encoding strategies are showcased
- Feature expansion ratios: 11.2x (regression), 34.8x (classification)

## 🤝 Contributing

To extend the UI:
1. Add new visualization pages
2. Implement additional model types
3. Enhance the prediction interface
4. Add data export functionality

## 📄 License

This UI is part of the Database & Programming Essentials Assignment 4.
