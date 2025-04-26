# Banking ML Case Study: Early Loan Payoff Prediction

A case study analyzing banking data to predict early loan closings using machine learning techniques. This project is built as a React web application showcasing the entire data science workflow.

## Project Overview

This interactive web application demonstrates a complete machine learning workflow for predicting which loans will be paid off early by customers. Early loan closings impact a bank's expected interest income and can disrupt financial planning, making this prediction valuable for financial institutions.

## Features

- **Interactive Data Analysis**: Visualizations and explanations of the banking dataset
- **Multiple ML Models**: Comparison of Logistic Regression, Random Forest, XGBoost, and LightGBM
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Educational Interface**: Code snippets and explanations throughout the workflow

## Project Structure

- **Introduction**: Project goals and dataset overview
- **Data Preparation**: Loading, cleaning, and preprocessing the banking data
- **Exploratory Analysis**: Statistical analysis and data visualization
- **Feature Engineering**: Scaling, encoding, and balancing the dataset
- **Model Training**: Implementation and tuning of multiple ML algorithms
- **Evaluation**: Performance metrics and model comparison
- **Results Export**: Final model selection and implementation recommendations

## Key Findings

- The tuned LightGBM model emerged as the best performing model with:
  - 93% overall accuracy
  - 83% precision for early payoff predictions
  - 96% specificity (correctly identifying loans not paid off early)
  - Strong ROC-AUC score of 0.956

## Technologies Used

- **Frontend**: React, CSS
- **Data Analysis**: Python, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, SMOTE
- **Development**: Vite, ESLint

## Getting Started

1. Clone the repository
2. Install dependencies with `npm install`
3. Run the development server with `npm run dev`
4. Open the application in your browser at `http://localhost:5173`

## About This Project

This case study demonstrates the application of machine learning in banking to solve real-world financial challenges. By predicting early loan payoffs, banks can better:
- Understand customer repayment behavior
- Anticipate changes in cash flow
- Develop targeted strategies for customer retention
- Make more informed decisions about loan terms and interest rates
