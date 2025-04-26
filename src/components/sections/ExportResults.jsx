import React, { useState } from 'react';
import './sections.css';
import CodeBlock from '../common/CodeBlock';

const ExportResults = () => {
  // State to track which code sections are visible
  const [visibleSections, setVisibleSections] = useState({
    exportCode: false,
    saveModelCode: false,
  });

  // Toggle visibility for a specific section
  const toggleSection = (section) => {
    setVisibleSections(prevState => ({
      ...prevState,
      [section]: !prevState[section]
    }));
  };

  // Code content for export model results
  const exportCodeContent = `#Defining a function for calculating and returning all performance metrics
def get_model_performance(y_test, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = confusion_matrix(y_test, y_pred)[0, 0] / (confusion_matrix(y_test, y_pred)[0, 0] + confusion_matrix(y_test, y_pred)[0, 1])
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1]) 
    
    #Returning the calculated metrics
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Specificity': specificity,
        'ROC-AUC Score': roc_auc
    }

#Initializing an empty list for storing model results
model_results = []

# ----- Untuned Logistic Regression 
log_reg_metrics = get_model_performance(y_test, log_reg_y_pred, log_reg_y_pred_proba)  # Calculating performance metrics
model_results.append({
    'Model': 'Untuned Logistic Regression',
    **log_reg_metrics,
    'Best Parameters': None  # No hyperparameters for untuned model
})

# ----- Tuned Logistic Regression 
log_reg_tuned_metrics = get_model_performance(y_test, log_reg_tuned_y_pred, log_reg_tuned_y_pred_proba)  # Calculating metrics for tuned model
model_results.append({
    'Model': 'Tuned Logistic Regression',
    **log_reg_tuned_metrics,
    'Best Parameters': best_log_reg_model.get_params()  # Storing the best parameters
})

# ... (additional model evaluations)

#Converting the list of results into a DataFrame for easy handling
df_results = pd.DataFrame(model_results)

# Exporting the DataFrame to a CSV file for storing results
df_results.to_csv('model_performance_results.csv', index=False)

# Displaying the DataFrame for verification
print(df_results)`;

  return (
    <div className="section-content">
      <h2 id="VI.-Export-Results">VI. Export Results</h2>
      
      <div className="section-overview">
        <p>
          In this final section, we'll export all model performance metrics and parameters to a structured format.
          This will allow for easy comparison of results and documentation of the modeling process.
        </p>
      </div>
      
      <div className="subsection">
        <h3>a) Export Model Results</h3>
        <p>
          We'll create a comprehensive DataFrame containing the performance metrics and parameters for all models,
          then export it to a CSV file for future reference and analysis.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('exportCode')}
          >
            <span className="code-title">Export Model Results</span>
          </div>
          {visibleSections.exportCode && (
            <pre className="code-content">
              <code>{exportCodeContent}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Exported Model Results</span>
          </div>
          <div className="output-content">
            <pre>{`                         Model  Accuracy  Precision    Recall  F1 Score  \\
0  Untuned Logistic Regression  0.884605   0.675174  0.795625  0.730468   
1    Tuned Logistic Regression  0.909726   0.738684  0.836637  0.784615   
2              Untuned XGBoost  0.925578   0.819396  0.796992  0.808039   
3                Tuned XGBoost  0.918189   0.790872  0.793575  0.792221   
4        Untuned Random Forest  0.918995   0.785525  0.808612  0.796901   
5          Tuned Random Forest  0.919398   0.791357  0.801094  0.796196   
6             Untuned LightGBM  0.923160   0.808732  0.797676  0.803166   
7               Tuned LightGBM  0.925578   0.826743  0.786056  0.805886   

   Specificity  ROC-AUC Score  \\
0     0.906370       0.924613   
1     0.927604       0.948484   
2     0.957031       0.954599   
3     0.948671       0.956497   
4     0.945996       0.954813   
5     0.948336       0.954742   
6     0.953854       0.957223   
7     0.959706       0.955989   

                                     Best Parameters  
0                                               None  
1  {'C': 1, 'class_weight': None, 'dual': False, ...  
2                                               None  
3  {'objective': 'binary:logistic', 'use_label_en...  
4                                               None  
5  {'n_estimators': 100, 'min_samples_split': 5, ...  
6                                               None  
7  {'subsample': 0.8, 'num_leaves': 500, 'n_estim...  `}</pre>
          </div>
        </div>
        
       
      </div>
      
      <div className="subsection">
        <h3>b) Final Model Selection and Implementation</h3>
        <p>
          After comparing all model results, we can make a final decision on which model to select for implementation.
        </p>
        
        <div className="info-box conclusion">
          <h4>Model Selection Decision</h4>
          <p>
            Based on the comprehensive evaluation, <strong>Tuned LightGBM</strong> is selected as the final model for the following reasons:
          </p>
          <ul>
            <li><strong>Highest Precision (0.83)</strong> - It correctly identifies early loan payoffs with the highest precision, minimizing false positives</li>
            <li><strong>Highest Specificity (0.96)</strong> - It excels at correctly identifying loans that won't be paid off early</li>
            <li><strong>Strong Overall Accuracy (0.93)</strong> - Tied for the highest accuracy among all models</li>
            <li><strong>Excellent F1-Score (0.81)</strong> - Provides the best balance between precision and recall</li>
            <li><strong>Computational Efficiency</strong> - LightGBM is known for its fast training and inference times</li>
          </ul>
        </div>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('saveModelCode')}
          >
            <span className="code-title">Saving the Final Model</span>
          </div>
          {visibleSections.saveModelCode && (
            <pre className="code-content">
              <code>{`# Save the tuned LightGBM model to a file using joblib
import joblib

# Save the model
joblib.dump(lgb_random_search.best_estimator_, 'final_lightgbm_model.pkl')

# Code for later loading the model
# loaded_model = joblib.load('final_lightgbm_model.pkl')
# predictions = loaded_model.predict(new_data)`}</code>
            </pre>
          )}
        </div>
        
        <div className="info-box recommendations">
          <h4>Implementation Recommendations</h4>
          <p>
            When implementing the selected model in a production environment, consider the following:
          </p>
          <ul>
            <li><strong>Model Monitoring</strong> - Set up periodic evaluation to detect model drift over time</li>
            <li><strong>Feature Engineering Pipeline</strong> - Implement the same preprocessing steps used during training</li>
            <li><strong>Threshold Optimization</strong> - Consider adjusting the classification threshold based on business requirements (e.g., favoring recall over precision if needed)</li>
            <li><strong>Interpretability Tools</strong> - Implement tools like SHAP values to explain individual predictions</li>
            <li><strong>Regular Retraining</strong> - Set up a schedule for model retraining as new data becomes available</li>
          </ul>
        </div>
      </div>
      
      <div className="subsection">
        <h3>Project Summary</h3>
        <p>
          In this case study, we set out to predict which loans would be paid off early by customers.
          Through careful data preparation, feature engineering, and model evaluation, we've developed
          a high-performing predictive model that can help the bank anticipate early loan payoffs.
        </p>
        
        <div className="info-box conclusion">
          <h4>Key Achievements</h4>
          <ul>
            <li>Successfully addressed data quality issues, including missing values and outliers</li>
            <li>Created informative features through engineering and transformation</li>
            <li>Developed and compared multiple machine learning models</li>
            <li>Selected a final model with 93% accuracy and 83% precision for early payoff prediction</li>
            <li>Provided a comprehensive evaluation framework for model comparison</li>
            <li>Created an exportable, reproducible analysis with clear documentation</li>
          </ul>
        </div>
        
        <p>
          The insights and predictions generated by this model can help the bank:
        </p>
        <ul>
          <li>Better understand customer repayment behavior</li>
          <li>Anticipate changes in cash flow from early repayments</li>
          <li>Develop targeted strategies for customer retention and product offerings</li>
          <li>Make more informed decisions about loan terms and interest rates</li>
        </ul>
      </div>
    </div>
  );
};

export default ExportResults;
