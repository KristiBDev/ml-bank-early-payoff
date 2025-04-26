import React, { useState } from 'react';
import './sections.css';
import CodeBlock from '../common/CodeBlock';

const Evaluation = () => {
  // State to track which code sections are visible
  const [visibleSections, setVisibleSections] = useState({
    logRegConfMatrix: false,
    tunedLogRegConfMatrix: false,
    rfConfMatrix: false,
    tunedRfConfMatrix: false,
    xgbConfMatrix: false,
    tunedXgbConfMatrix: false,
    lgbmConfMatrix: false,
    tunedLgbmConfMatrix: false,
    metricsCode: false,
  });

  // Toggle visibility for a specific section
  const toggleSection = (section) => {
    setVisibleSections(prevState => ({
      ...prevState,
      [section]: !prevState[section]
    }));
  };

  return (
    <div className="section-content">
      <h2 id="V.-Evaluation">V. Evaluation</h2>
      
      <div className="section-overview">
        <p>
          In this section, we'll evaluate the performance of our models using confusion matrices and additional metrics.
          This will help us understand the strengths and weaknesses of each model in predicting early loan payoffs.
        </p>
      </div>
      
      <div className="subsection">
        <h3>a) Evaluating with Confusion Matrix</h3>
        <p>
          Let's analyze the confusion matrices for each model to understand how they perform in distinguishing between
          loans that were paid off early (Class 1) and those that were not (Class 0).
        </p>
        
        <h4>Logistic Regression</h4>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('logRegConfMatrix')}
          >
            <span className="code-title">Logistic Regression Confusion Matrix</span>
          </div>
          {visibleSections.logRegConfMatrix && (
            <pre className="code-content">
              <code>{`# Confusion matrix for Logistic Regression
conf_matrix_log_reg = confusion_matrix(y_test, log_reg_y_pred)
print("Confusion Matrix for Logistic Regression:")
print(conf_matrix_log_reg)

# Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_log_reg, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Logistic Regression Confusion Matrix</span>
          </div>
          <div className="output-content">
            <pre>{`Confusion Matrix for Logistic Regression:
[[5421  560]
 [ 299 1164]]`}</pre>
          </div>
        </div>
        
        <div className="visualization">
          <img src="/conf_mat_logreg.png" alt="Logistic Regression Confusion Matrix" className="analysis-image" />
          <p className="image-caption">
            Figure 13: Confusion Matrix for untuned Logistic Regression model.
          </p>
        </div>
        
        <h4>Tuned Logistic Regression</h4>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('tunedLogRegConfMatrix')}
          >
            <span className="code-title">Tuned Logistic Regression Confusion Matrix</span>
          </div>
          {visibleSections.tunedLogRegConfMatrix && (
            <pre className="code-content">
              <code>{`# Confusion matrix for Tuned Logistic Regression
conf_matrix_log_reg_tuned = confusion_matrix(y_test, y_pred_tuned)
print("Confusion Matrix for Tuned Logistic Regression:")
print(conf_matrix_log_reg_tuned)

# Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_log_reg_tuned, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Tuned Logistic Regression')
plt.show()`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Tuned Logistic Regression Confusion Matrix</span>
          </div>
          <div className="output-content">
            <pre>{`Confusion Matrix for Tuned Logistic Regression:
[[5547  434]
 [ 239 1224]]`}</pre>
          </div>
        </div>
        
        <div className="visualization">
          <img src="/conf_mat_tuned_logreg.png" alt="Tuned Logistic Regression Confusion Matrix" className="analysis-image" />
          <p className="image-caption">
            Figure 14: Confusion Matrix for tuned Logistic Regression model.
          </p>
        </div>
        
        <h4>Random Forest</h4>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('rfConfMatrix')}
          >
            <span className="code-title">Random Forest Confusion Matrix</span>
          </div>
          {visibleSections.rfConfMatrix && (
            <pre className="code-content">
              <code>{`# Confusion matrix for untuned Random Forest
conf_matrix_rf = confusion_matrix(y_test, rf_y_pred)
print("Confusion Matrix for Untuned Random Forest:")
print(conf_matrix_rf)

# Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Untuned Random Forest')
plt.show()`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Untuned Random Forest Confusion Matrix</span>
          </div>
          <div className="output-content">
            <pre>{`Confusion Matrix for Untuned Random Forest:
[[5658  323]
 [ 280 1183]]`}</pre>
          </div>
        </div>
        
        <div className="visualization">
          <img src="/conf_mat_rf.png" alt="Random Forest Confusion Matrix" className="analysis-image" />
          <p className="image-caption">
            Figure 15: Confusion Matrix for untuned Random Forest model.
          </p>
        </div>

        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('tunedRfConfMatrix')}
          >
            <span className="code-title">Tuned Random Forest Confusion Matrix</span>
          </div>
          {visibleSections.tunedRfConfMatrix && (
            <pre className="code-content">
              <code>{`# Confusion matrix for tuned Random Forest
conf_matrix_rf_tuned = confusion_matrix(y_test, rf_tuned_y_pred)
print("Confusion Matrix for Tuned Random Forest:")
print(conf_matrix_rf_tuned)

# Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_rf_tuned, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Tuned Random Forest')
plt.show()`}</code>
            </pre>
          )}
        </div>

        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Tuned Random Forest Confusion Matrix</span>
          </div>
          <div className="output-content">
            <pre>{`Confusion Matrix for Tuned Random Forest:
[[5672  309]
 [ 291 1172]]`}</pre>
          </div>
        </div>
        
        <div className="visualization">
          <img src="/conf_mat_tuned_rf.png" alt="Tuned Random Forest Confusion Matrix" className="analysis-image" />
          <p className="image-caption">
            Figure 16: Confusion Matrix for tuned Random Forest model.
          </p>
        </div>
        
        <h4>XGBoost</h4>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('xgbConfMatrix')}
          >
            <span className="code-title">XGBoost Confusion Matrix</span>
          </div>
          {visibleSections.xgbConfMatrix && (
            <pre className="code-content">
              <code>{`# Confusion matrix for untuned XGBoost
conf_matrix_xgb = confusion_matrix(y_test, xgb_y_pred)
print("Confusion Matrix for Untuned XGBoost:")
print(conf_matrix_xgb)

# Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_xgb, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Untuned XGBoost')
plt.show()`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Untuned XGBoost Confusion Matrix</span>
          </div>
          <div className="output-content">
            <pre>{`Confusion Matrix for Untuned XGBoost:
[[5724  257]
 [ 297 1166]]`}</pre>
          </div>
        </div>
        
        <div className="visualization">
          <img src="/conf_mat_xgb.png" alt="XGBoost Confusion Matrix" className="analysis-image" />
          <p className="image-caption">
            Figure 17: Confusion Matrix for untuned XGBoost model.
          </p>
        </div>

        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('tunedXgbConfMatrix')}
          >
            <span className="code-title">Tuned XGBoost Confusion Matrix</span>
          </div>
          {visibleSections.tunedXgbConfMatrix && (
            <pre className="code-content">
              <code>{`# Confusion matrix for tuned XGBoost
conf_matrix_xgb_tuned = confusion_matrix(y_test, xgb_tuned_y_pred)
print("Confusion Matrix for Tuned XGBoost:")
print(conf_matrix_xgb_tuned)

# Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_xgb_tuned, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Tuned XGBoost')
plt.show()`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Tuned XGBoost Confusion Matrix</span>
          </div>
          <div className="output-content">
            <pre>{`Confusion Matrix for Tuned XGBoost:
[[5674  307]
 [ 302 1161]]`}</pre>
          </div>
        </div>
        
        <div className="visualization">
          <img src="/conf_mat_tuned_xgb.png" alt="Tuned XGBoost Confusion Matrix" className="analysis-image" />
          <p className="image-caption">
            Figure 18: Confusion Matrix for tuned XGBoost model.
          </p>
        </div>
        
        <h4>LightGBM</h4>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('lgbmConfMatrix')}
          >
            <span className="code-title">LightGBM Confusion Matrix</span>
          </div>
          {visibleSections.lgbmConfMatrix && (
            <pre className="code-content">
              <code>{`# Confusion matrix for untuned LightGBM
conf_matrix_lgb = confusion_matrix(y_test, lgb_y_pred)
print("Confusion Matrix for Untuned LightGBM:")
print(conf_matrix_lgb)

# Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_lgb, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Untuned LightGBM')
plt.show()`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Untuned LightGBM Confusion Matrix</span>
          </div>
          <div className="output-content">
            <pre>{`Confusion Matrix for Untuned LightGBM:
[[5705  276]
 [ 296 1167]]`}</pre>
          </div>
        </div>
        
        <div className="visualization">
          <img src="/conf_mat_lgb.png" alt="LightGBM Confusion Matrix" className="analysis-image" />
          <p className="image-caption">
            Figure 19: Confusion Matrix for untuned LightGBM model.
          </p>
        </div>

        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('tunedLgbmConfMatrix')}
          >
            <span className="code-title">Tuned LightGBM Confusion Matrix</span>
          </div>
          {visibleSections.tunedLgbmConfMatrix && (
            <pre className="code-content">
              <code>{`# Confusion matrix for tuned LightGBM
conf_matrix_lgb_tuned = confusion_matrix(y_test, lgb_tuned_y_pred)
print("Confusion Matrix for Tuned LightGBM:")
print(conf_matrix_lgb_tuned)

# Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_lgb_tuned, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Tuned LightGBM')
plt.show()`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Tuned LightGBM Confusion Matrix</span>
          </div>
          <div className="output-content">
            <pre>{`Confusion Matrix for Tuned LightGBM:
[[5740  241]
 [ 313 1150]]`}</pre>
          </div>
        </div>
        
        <div className="visualization">
          <img src="/conf_mat_tuned_lgb.png" alt="Tuned LightGBM Confusion Matrix" className="analysis-image" />
          <p className="image-caption">
            Figure 20: Confusion Matrix for tuned LightGBM model.
          </p>
        </div>
        
        <div className="analysis-notes">
          <h4>Confusion Matrix Analysis:</h4>
          
          <h5>Logistic Regression</h5>
          <ul>
            <li><strong>Untuned:</strong>
              <p>True Negatives (TN): 5421, False Positives (FP): 560</p>
              <p>True Positives (TP): 1164, False Negatives (FN): 299</p>
              <p>The model correctly identifies most loans not paid off early, but it struggles with predicting early payoffs (high FN and FP).</p>
            </li>
            <li><strong>Tuned:</strong>
              <p>True Negatives (TN): 5547, False Positives (FP): 434</p>
              <p>True Positives (TP): 1224, False Negatives (FN): 239</p>
              <p>After tuning, the model reduced both false positives and false negatives, improving its ability to predict early payoffs.</p>
            </li>
          </ul>
          
          <h5>Random Forest</h5>
          <ul>
            <li><strong>Untuned:</strong>
              <p>True Negatives (TN): 5658, False Positives (FP): 323</p>
              <p>True Positives (TP): 1183, False Negatives (FN): 280</p>
              <p>Random Forest shows better performance than Logistic Regression, reducing false positives and false negatives.</p>
            </li>
            <li><strong>Tuned:</strong>
              <p>True Negatives (TN): 5672, False Positives (FP): 309</p>
              <p>True Positives (TP): 1172, False Negatives (FN): 291</p>
              <p>Tuning slightly reduced false positives but increased false negatives, indicating a trade-off.</p>
            </li>
          </ul>
          
          <h5>XGBoost</h5>
          <ul>
            <li><strong>Untuned:</strong>
              <p>True Negatives (TN): 5724, False Positives (FP): 257</p>
              <p>True Positives (TP): 1166, False Negatives (FN): 297</p>
              <p>XGBoost has the fewest false positives so far, making it strong at correctly predicting loans not paid off early.</p>
            </li>
            <li><strong>Tuned:</strong>
              <p>True Negatives (TN): 5674, False Positives (FP): 307</p>
              <p>True Positives (TP): 1161, False Negatives (FN): 302</p>
              <p>After tuning, false positives and false negatives slightly increased, indicating a minor decline in performance.</p>
            </li>
          </ul>
          
          <h5>LightGBM</h5>
          <ul>
            <li><strong>Untuned:</strong>
              <p>True Negatives (TN): 5705, False Positives (FP): 276</p>
              <p>True Positives (TP): 1167, False Negatives (FN): 296</p>
              <p>LightGBM showed strong overall performance with fewer false positives and false negatives compared to other models.</p>
            </li>
            <li><strong>Tuned:</strong>
              <p>True Negatives (TN): 5740, False Positives (FP): 241</p>
              <p>True Positives (TP): 1150, False Negatives (FN): 313</p>
              <p>The tuned model reduced false positives significantly but increased false negatives, improving in predicting loans not paid off early but slightly worsening for early payoffs.</p>
            </li>
          </ul>
          
          <h5>General Observations:</h5>
          <ul>
            <li><strong>Best Non-Early Payoff Prediction:</strong> Untuned XGBoost (TN = 5724) and Tuned LightGBM (TN = 5740) excel in predicting loans not paid off early.</li>
            <li><strong>Best Early Payoff Prediction:</strong> Tuned Logistic Regression (TP = 1224) captures the most early payoffs, followed by Tuned Random Forest.</li>
            <li><strong>Trade-offs:</strong> Tuning improved false positives in most models but increased false negatives in some, such as XGBoost and LightGBM.</li>
            <li><strong>Overall:</strong> Untuned XGBoost and Tuned LightGBM show strong overall performance, while Tuned Logistic Regression excels in predicting early payoffs.</li>
          </ul>
        </div>
      </div>
      
      <div className="subsection">
        <h3>b) Evaluating with Other Metrics</h3>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('metricsCode')}
          >
            <span className="code-title">Evaluation Metrics Code</span>
          </div>
          {visibleSections.metricsCode && (
            <pre className="code-content">
              <code>{`#Defining a function to calculate and display all the metrics
def evaluate_model_performance(model_name, y_test, y_pred, y_pred_proba):
    print(f"--- Metrics for {model_name} ---")
    
    #Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = confusion_matrix(y_test, y_pred)[0, 0] / (confusion_matrix(y_test, y_pred)[0, 0] + confusion_matrix(y_test, y_pred)[0, 1])
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1]) 
    
    #Printing results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\\n")


# ----- Untuned Logistic Regression 
log_reg_model = LogisticRegression(random_state=40, max_iter=5000)  
log_reg_model.fit(X_train_res, y_train_res)  # Fitting the model
log_reg_y_pred = log_reg_model.predict(X_test)
log_reg_y_pred_proba = log_reg_model.predict_proba(X_test)
evaluate_model_performance("Untuned Logistic Regression", y_test, log_reg_y_pred, log_reg_y_pred_proba)

# ----- Tuned Logistic Regression 
best_log_reg_model.fit(X_train_res, y_train_res) 
log_reg_tuned_y_pred = best_log_reg_model.predict(X_test)
log_reg_tuned_y_pred_proba = best_log_reg_model.predict_proba(X_test)
evaluate_model_performance("Tuned Logistic Regression", y_test, log_reg_tuned_y_pred, log_reg_tuned_y_pred_proba)


# ----- Untuned XGBoost 
xgb_model = XGBClassifier(random_state=40, eval_metric='logloss')
xgb_model.fit(X_train_res, y_train_res)  
xgb_y_pred = xgb_model.predict(X_test)
xgb_y_pred_proba = xgb_model.predict_proba(X_test)
evaluate_model_performance("Untuned XGBoost", y_test, xgb_y_pred, xgb_y_pred_proba)


# ----- Tuned XGBoost 
best_xgb_model.fit(X_train_res, y_train_res) 
y_pred_tuned = best_xgb_model.predict(X_test)
xgb_tuned_y_pred_proba = best_xgb_model.predict_proba(X_test)
evaluate_model_performance("Tuned XGBoost", y_test, y_pred_tuned, xgb_tuned_y_pred_proba)



# ----- Untuned Random Forest 
rf_model = RandomForestClassifier(random_state=40)
rf_model.fit(X_train_res, y_train_res) 
rf_y_pred = rf_model.predict(X_test)
rf_y_pred_proba = rf_model.predict_proba(X_test)
evaluate_model_performance("Untuned Random Forest", y_test, rf_y_pred, rf_y_pred_proba)


# ----- Tuned Random Forest 
rf_random_search.best_estimator_.fit(X_train_res, y_train_res) 
rf_tuned_y_pred = rf_random_search.best_estimator_.predict(X_test)
rf_tuned_y_pred_proba = rf_random_search.best_estimator_.predict_proba(X_test)
evaluate_model_performance("Tuned Random Forest", y_test, rf_tuned_y_pred, rf_tuned_y_pred_proba)


# ----- Untuned LightGBM 
lgb_model = lgb.LGBMClassifier(random_state=40, is_unbalance=True)
lgb_model.fit(X_train_res, y_train_res)  
lgb_y_pred = lgb_model.predict(X_test)
lgb_y_pred_proba = lgb_model.predict_proba(X_test)
evaluate_model_performance("Untuned LightGBM", y_test, lgb_y_pred, lgb_y_pred_proba)


# ----- Tuned LightGBM 
lgb_random_search.best_estimator_.fit(X_train_res, y_train_res)  
lgb_tuned_y_pred = lgb_random_search.best_estimator_.predict(X_test)
lgb_tuned_y_pred_proba = lgb_random_search.best_estimator_.predict_proba(X_test)
evaluate_model_performance("Tuned LightGBM", y_test, lgb_tuned_y_pred, lgb_tuned_y_pred_proba)`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Model Performance Metrics</span>
          </div>
          <div className="output-content">
            <pre>{`--- Metrics for Untuned Logistic Regression ---
Accuracy: 0.8846
Precision: 0.6752
Recall: 0.7956
F1 Score: 0.7305
Specificity: 0.9064
ROC-AUC Score: 0.9246


--- Metrics for Tuned Logistic Regression ---
Accuracy: 0.9097
Precision: 0.7387
Recall: 0.8366
F1 Score: 0.7846
Specificity: 0.9276
ROC-AUC Score: 0.9485


--- Metrics for Untuned XGBoost ---
Accuracy: 0.9256
Precision: 0.8194
Recall: 0.7970
F1 Score: 0.8080
Specificity: 0.9570
ROC-AUC Score: 0.9546


--- Metrics for Tuned XGBoost ---
Accuracy: 0.9182
Precision: 0.7909
Recall: 0.7936
F1 Score: 0.7922
Specificity: 0.9487
ROC-AUC Score: 0.9565


--- Metrics for Untuned Random Forest ---
Accuracy: 0.9190
Precision: 0.7855
Recall: 0.8086
F1 Score: 0.7969
Specificity: 0.9460
ROC-AUC Score: 0.9548


--- Metrics for Tuned Random Forest ---
Accuracy: 0.9194
Precision: 0.7914
Recall: 0.8011
F1 Score: 0.7962
Specificity: 0.9483
ROC-AUC Score: 0.9547


--- Metrics for Untuned LightGBM ---
Accuracy: 0.9232
Precision: 0.8087
Recall: 0.7977
F1 Score: 0.8032
Specificity: 0.9539
ROC-AUC Score: 0.9572


--- Metrics for Tuned LightGBM ---
Accuracy: 0.9256
Precision: 0.8267
Recall: 0.7861
F1 Score: 0.8059
Specificity: 0.9597
ROC-AUC Score: 0.9560`}</pre>
          </div>
        </div>
        
        <div className="analysis-notes">
          <h4>Logistic Regression</h4>
          <p><strong>Untuned:</strong></p>
          <ul>
            <li>Accuracy: 0.88, Precision: 0.67, Recall: 0.80, F1 Score: 0.73</li>
            <li>ROC-AUC: 0.92, Specificity: 0.91</li>
            <li>Decent baseline, but lower precision shows it struggles with predicting loans paid off early.</li>
          </ul>
          <p><strong>Tuned:</strong></p>
          <ul>
            <li>Accuracy: 0.91, Precision: 0.74, Recall: 0.84, F1 Score: 0.78</li>
            <li>ROC-AUC: 0.95, Specificity: 0.93</li>
            <li>Improvement in all metrics, especially ROC-AUC and recall for class 1.</li>
          </ul>
          
          <h4>XGBoost</h4>
          <p><strong>Untuned:</strong></p>
          <ul>
            <li>Accuracy: 0.93, Precision: 0.82, Recall: 0.80, F1 Score: 0.81</li>
            <li>ROC-AUC: 0.95, Specificity: 0.96</li>
            <li>Strong out-of-the-box performance, balanced across all metrics.</li>
          </ul>
          <p><strong>Tuned:</strong></p>
          <ul>
            <li>Accuracy: 0.92, Precision: 0.79, Recall: 0.79, F1 Score: 0.79</li>
            <li>ROC-AUC: 0.96, Specificity: 0.95</li>
            <li>Marginal changes in metrics after tuning, with a slight improvement in ROC-AUC.</li>
          </ul>
          
          <h4>Random Forest</h4>
          <p><strong>Untuned:</strong></p>
          <ul>
            <li>Accuracy: 0.92, Precision: 0.79, Recall: 0.81, F1 Score: 0.80</li>
            <li>ROC-AUC: 0.95, Specificity: 0.95</li>
            <li>Strong results, especially in recall and ROC-AUC, showing its ability to capture early payoffs.</li>
          </ul>
          <p><strong>Tuned:</strong></p>
          <ul>
            <li>Accuracy: 0.92, Precision: 0.79, Recall: 0.80, F1 Score: 0.80</li>
            <li>ROC-AUC: 0.95, Specificity: 0.95</li>
            <li>No major improvements however maintains strong performance.</li>
          </ul>
          
          <h4>LightGBM</h4>
          <p><strong>Untuned:</strong></p>
          <ul>
            <li>Accuracy: 0.92, Precision: 0.81, Recall: 0.80, F1 Score: 0.80</li>
            <li>ROC-AUC: 0.96, Specificity: 0.95</li>
            <li>LightGBM shows strong performance without tuning, with balanced precision and recall for both classes.</li>
          </ul>
          <p><strong>Tuned:</strong></p>
          <ul>
            <li>Accuracy: 0.93, Precision: 0.83, Recall: 0.79, F1 Score: 0.81</li>
            <li>ROC-AUC: 0.96, Specificity: 0.96</li>
            <li>Tuned LightGBM has better precision and the highest specificity reached, further improving its performance over XGBoost.</li>
          </ul>
          
          <h4>Conclusion:</h4>
          <ul>
            <li>LightGBM tuned model outperforms others in precision and specificity.</li>
            <li>Overall, both XGBoost and LightGBM show strong performance, with LightGBM offering faster training and marginally better precision, specificity and a ROC-AUC score.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Evaluation;

