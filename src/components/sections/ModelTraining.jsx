import React, { useState } from 'react';
import './sections.css';
import CodeBlock from '../common/CodeBlock';

const ModelTraining = () => {
  // State to track which code sections are visible
  const [visibleSections, setVisibleSections] = useState({
    logisticRegressionCode: false,
    tuningLogRegCode: false,
    randomForestCode: false,
    tuningRFCode: false,
    xgboostCode: false,
    tuningXGBCode: false,
    lightgbmCode: false,
    tuningLGBMCode: false
  });

  // Toggle visibility for a specific section
  const toggleSection = (section) => {
    setVisibleSections(prevState => ({
      ...prevState,
      [section]: !prevState[section]
    }));
  };

  // Example content for Logistic Regression implementation
  const logisticRegressionCode = `#Setting up logistic regression model with random state and iterations
log_reg_model = LogisticRegression(random_state=40, max_iter=5000)

#Training the logistic regression model on resampled training data
log_reg_model.fit(X_train_res, y_train_res)

#Making predictions on the test data
log_reg_y_pred = log_reg_model.predict(X_test)

#Displaying classification report for logistic regression
print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_reg_y_pred))

#Calculating and printing the ROC-AUC score for logistic regression
log_reg_roc_auc = roc_auc_score(y_test, log_reg_model.predict_proba(X_test)[:, 1])
print(f"Logistic Regression ROC-AUC Score: {log_reg_roc_auc}")`;

  // Example content for tuning Logistic Regression
  const tuningLogRegCode = `#Defining parameter grid for tuning logistic regression with solvers and penalties
param_grid = {
    'penalty': ['l1', 'l2'],  # Regularization types: L1 for Lasso (feature selection), L2 for Ridge (shrinkage)
    'C': [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength (smaller values specify stronger regularization)
    'solver': ['liblinear', 'saga'],  # Solvers for optimization: 'liblinear' for small datasets, 'saga' for larger datasets
    'max_iter': [10, 100, 500]  # Maximum number of iterations for the solver to converge
}

#Setting up GridSearchCV with cross-validation for logistic regression
log_reg_model = LogisticRegression()
grid_search = GridSearchCV(estimator=log_reg_model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1, error_score='raise')

#Running GridSearchCV on the resampled training data
grid_search.fit(X_train_res, y_train_res)

#Displaying the best parameters found by GridSearchCV
print("Best parameters found:", grid_search.best_params_)

#Using the best logistic regression model to make predictions on the test data
best_log_reg_model = grid_search.best_estimator_
y_pred_tuned = best_log_reg_model.predict(X_test)

#Displaying the classification report for the tuned logistic regression model
print("Classification Report after tuning:")
print(classification_report(y_test, y_pred_tuned))

#Calculating and printing the ROC-AUC score for the tuned model
roc_auc_tuned = roc_auc_score(y_test, best_log_reg_model.predict_proba(X_test)[:, 1])
print(f"Tuned ROC-AUC Score: {roc_auc_tuned}")`;

  // Example content for Random Forest implementation
  const randomForestCode = `#Initializing the Random Forest classifier with a random state for reproducibility
rf_model = RandomForestClassifier(random_state=40)

#Training the Random Forest model on the resampled training data
rf_model.fit(X_train_res, y_train_res)

#Making predictions on the test data
rf_y_pred = rf_model.predict(X_test)

#Displaying the classification report to evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, rf_y_pred))

#Calculating and printing the ROC-AUC score for the Random Forest model
rf_roc_auc_score = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score: {rf_roc_auc_score}")`;

  // Example content for tuning Random Forest
  const tuningRFCode = `#Defining the hyperparameters to search for Random Forest
param_dist = {
    'n_estimators': [100, 300, 500],  # Number of trees in the forest
    'max_depth': [10, 30, 50, None],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4, 10],  # Minimum number of samples required at each leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}

#Initializing the Random Forest classifier with a random state for reproducibility
rf_model = RandomForestClassifier(random_state=40)

#Setting up RandomizedSearchCV for hyperparameter tuning with Random Forest
rf_random_search = RandomizedSearchCV(
    estimator=rf_model, 
    param_distributions=param_dist, 
    n_iter=10,  # Limiting the number of iterations for faster execution
    cv=5,  # 5-fold cross-validation
    verbose=2,  
    random_state=40, 
    n_jobs=-1  # Using all available processors
)

#Running RandomizedSearchCV on the resampled training data 
print("Starting RandomizedSearchCV...")
rf_random_search.fit(X_train_res, y_train_res)

#Displaying the best hyperparameters found by RandomizedSearchCV
print("Best parameters found:", rf_random_search.best_params_)

#Using the best Random Forest model to make predictions on the test data
rf_tuned_y_pred = rf_random_search.best_estimator_.predict(X_test)

#Displaying the classification report for the tuned Random Forest model
print("Classification Report:")
print(classification_report(y_test, rf_tuned_y_pred))

#Calculating and printing the ROC-AUC score for the tuned Random Forest model
rf_tuned_roc_auc_score = roc_auc_score(y_test, rf_random_search.best_estimator_.predict_proba(X_test)[:, 1])
print(f"Tuned ROC-AUC Score: {rf_tuned_roc_auc_score}")`;

  // Example content for XGBoost implementation
  const xgboostCode = `#Initializing the XGBoost classifier with a random state for reproducibility

xgb_model = XGBClassifier(random_state=40, eval_metric='logloss') #logloss specifies the evaluation metric (logarithmic loss)

#Training the XGBoost model on the resampled training data
xgb_model.fit(X_train_res, y_train_res)

#Making predictions on the test data using the trained XGBoost model
xgb_y_pred = xgb_model.predict(X_test)

#Displaying the classification report to evaluate XGBoost model's performance
print("Classification Report:")
print(classification_report(y_test, xgb_y_pred))

#Calculating and printing the ROC-AUC score for XGBoost model
xgb_roc_auc_score = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
print(f"XGBoost ROC-AUC Score: {xgb_roc_auc_score}")`;

  // Example content for tuning XGBoost
  const tuningXGBCode = `#Setting up the parameter grid for tuning XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees (estimators) to build
    'max_depth': [3, 6, 9],  # Maximum depth of each tree (controls model complexity)
    'learning_rate': [0.01, 0.1, 0.2],  # Step size for updating weights (controls how fast the model learns)
    'subsample': [0.6, 0.8, 1.0],  # Subsampling ratio of the training instances (to prevent overfitting)
    'colsample_bytree': [0.6, 0.8, 1.0],  # Subsampling ratio of features for each tree (to prevent overfitting)
    'gamma': [0, 0.1, 0.3]  # Minimum loss reduction required to make a split (controls regularization)
}

#Initializing the XGBoost classifier with a random state and evaluation metric
xgb_model = XGBClassifier(random_state=40, eval_metric='logloss')

#Setting up RandomizedSearchCV for hyperparameter tuning of XGBoost
random_search = RandomizedSearchCV(
    estimator=xgb_model, 
    param_distributions=param_grid, 
    n_iter=10,  # Number of random combinations to try
    cv=3,  # 3-fold cross-validation to assess model performance
    verbose=1,  # Enabling verbose to show progress during the search
    random_state=40, 
    n_jobs=-1  # Utilizing all available processors to speed up the search
)

#Running RandomizedSearchCV on the resampled training data
random_search.fit(X_train_res, y_train_res)

#Displaying the best hyperparameters found by RandomizedSearchCV
print("Best parameters found:", random_search.best_params_)

#Using the best XGBoost model to make predictions on the test data
best_xgb_model = random_search.best_estimator_
xgb_tuned_y_pred = best_xgb_model.predict(X_test)

#Displaying the classification report for the tuned XGBoost model
print("Classification Report after tuning:")
print(classification_report(y_test, xgb_tuned_y_pred))

#Calculating and printing the ROC-AUC score for the tuned XGBoost model
xgb_tuned_roc_auc_score = roc_auc_score(y_test, best_xgb_model.predict_proba(X_test)[:, 1])
print(f"Tuned ROC-AUC Score: {xgb_tuned_roc_auc_score}")`;

  // Example content for LightGBM implementation
  const lightgbmCode = `#Initializing the LightGBM classifier with a random state for reproducibility
#'is_unbalance=True' helps to handle imbalanced datasets by adjusting class weights
lgb_model = lgb.LGBMClassifier(random_state=40, is_unbalance=True)  
#Training the LightGBM model on the resampled training data (balanced with SMOTE)
lgb_model.fit(X_train_res, y_train_res)

#Making predictions on the test data using the trained LightGBM model
lgb_y_pred = lgb_model.predict(X_test)

#Displaying the classification report to evaluate LightGBM model's performance
print("Classification Report:")
print(classification_report(y_test, lgb_y_pred))

#Calculating and printing the ROC-AUC score for LightGBM model
lgb_roc_auc_score = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score: {lgb_roc_auc_score}")`;

  // Example content for tuning LightGBM
  const tuningLGBMCode = `#Defining the hyperparameter grid for LightGBM
param_grid = {
    'num_leaves': [50, 100, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [-1, 10, 20],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

#Initializing the LightGBM Classifier
lgb_model = lgb.LGBMClassifier(random_state=40, is_unbalance=True, verbose=-1)

#Initializing RandomizedSearchCV with the hyperparameter grid
lgb_random_search = RandomizedSearchCV(
    estimator=lgb_model, 
    param_distributions=param_grid, 
    n_iter=10,  # Number of parameter settings that are sampled
    cv=3,  # Cross-validation
    verbose=-1, 
    random_state=40, 
    n_jobs=-1  # Use all available processors
)

# Fiting RandomizedSearchCV to the resampled training data
print("Starting RandomizedSearchCV for LightGBM...")
lgb_random_search.fit(X_train_res, y_train_res)

#Printing the best parameters found
print("Best parameters found for LightGBM:", lgb_random_search.best_params_)

#Using the best estimator to make predictions on the test data
lgb_tuned_y_pred = lgb_random_search.best_estimator_.predict(X_test)

#Evaluating the model's performance
print("LightGBM Classification Report after tuning:")
print(classification_report(y_test, lgb_tuned_y_pred))

#Calculating the ROC-AUC score for the tuned model
lgb_tuned_roc_auc_score = roc_auc_score(y_test, lgb_random_search.best_estimator_.predict_proba(X_test)[:, 1])
print(f"Tuned LightGBM ROC-AUC Score: {lgb_tuned_roc_auc_score}")`;

  return (
    <div className="section-content">
      <h2 id="IV.-Modeling">IV. Modeling</h2>
      
      <div className="section-overview">
        <p>
          After preparing our data and engineering features, we can now build and evaluate 
          machine learning models to predict early loan payoffs. We'll try several classifiers 
          and tune their parameters for optimal performance.
        </p>
      </div>
      
      <div className="subsection">
        <h3>a) Modeling with Classifiers</h3>
        <p>
          For modeling, I selected four models: Logistic Regression, Random Forest, XGBoost, and LightGBM.
        </p>
        
        <h4>Logistic Regression</h4>
        <p>
          Logistic Regression is one of the most common and well-known classification models. It's often used as a baseline model for comparison.
          I wanted to see how Logistic Regression performs on this dataset compared to other models that may handle skewed data better, like tree-based models.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('logisticRegressionCode')}
          >
            <span className="code-title">Logistic Regression Implementation</span>
          </div>
          {visibleSections.logisticRegressionCode && (
            <pre className="code-content">
              <code>{logisticRegressionCode}</code>
            </pre>
          )}
        </div>
        
        <div className="results-box">
          <h4>Logistic Regression Results</h4>
          <div className="output-box">
  <div className="output-header">
    <span className="output-title">Output</span>
  </div>
  <div className="output-content">
    <pre>
{`Logistic Regression Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.91      0.93      5981
           1       0.68      0.80      0.73      1463

    accuracy                           0.88      7444
   macro avg       0.81      0.85      0.83      7444
weighted avg       0.89      0.88      0.89      7444

Logistic Regression ROC-AUC Score: 0.9246129489795836`}
    </pre>
  </div>
</div>

        
          <h5>Results Analysis:</h5>
          <p><strong>Precision, Recall, and F1-Score:</strong></p>
          <ul>
            <li>Class 0 (loans not paid off early) performed well with a high precision (0.95) and recall (0.91).</li>
            <li>Class 1 (loans paid off early) showed lower performance, particularly in precision (0.68), but recall is decent (0.80), indicating that the model is catching most early-paid loans but with more false positives.</li>
          </ul>
          <p><strong>Overall Accuracy:</strong> The model achieved 88% accuracy, which is strong but not definitive since accuracy alone may not reflect true performance on an unbalanced dataset.</p>
          <p><strong>ROC-AUC Score:</strong> The ROC-AUC score of 0.924 suggests the model can distinguish between the two classes fairly well.</p>
          <p>These results provide a solid baseline, but more sophisticated models may handle the skewed data better and improve performance, especially for class 1.</p>
        </div>
        
        <h4>Tuned Logistic Regression</h4>
        <p>
          GridSearchCV was chosen to systematically test different combinations of hyperparameters, to find the best setup for Logistic Regression.
          Since Logistic Regression has fewer hyperparameters compared to more complex models, it's feasible to exhaustively search the parameter space using GridSearchCV.
        </p>
        
        <div className="explanation-box">
          <h4>Explanation of Parameters Used</h4>
          <ul>
            <li><strong>Penalty (l1, l2):</strong>
              <ul>
                <li>L1 (Lasso): Performs feature selection by shrinking less important feature weights to zero, which can help in reducing overfitting and simplifying the model.</li>
                <li>L2 (Ridge): Shrinks coefficients but doesn't eliminate them, helping to avoid overfitting while keeping all features.</li>
              </ul>
            </li>
            <li><strong>Regularization Strength (C):</strong>
              <ul>
                <li>This controls the trade-off between fitting the data well and regularizing to prevent overfitting. Smaller values (closer to 0) impose stronger regularization.</li>
                <li>The best value found was C = 1, indicating a moderate amount of regularization was most effective.</li>
              </ul>
            </li>
            <li><strong>Solver (liblinear, saga):</strong>
              <ul>
                <li>liblinear: Suitable for small datasets and works with both L1 and L2 penalties.</li>
                <li>saga: More scalable for larger datasets but was not selected in this case.</li>
                <li>The best solver found was liblinear, likely due to the dataset size and efficiency with L1 regularization.</li>
              </ul>
            </li>
          </ul>
        </div>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('tuningLogRegCode')}
          >
            <span className="code-title">Tuning Logistic Regression</span>
          </div>
          {visibleSections.tuningLogRegCode && (
            <pre className="code-content">
              <code>{tuningLogRegCode}</code>
            </pre>
          )}
        </div>
        
        <div className="results-box">
          <h4>Tuned Logistic Regression Results</h4>
          <div className="output-box">
  <div className="output-header">
    <span className="output-title">Output</span>
  </div>
  <div className="output-content">
    <pre>
{`Fitting 3 folds for each of 60 candidates, totalling 180 fits
Best parameters found: {'C': 1, 'max_iter': 500, 'penalty': 'l1', 'solver': 'liblinear'}
Classification Report after tuning:
              precision    recall  f1-score   support

           0       0.96      0.93      0.94      5981
           1       0.74      0.84      0.78      1463

    accuracy                           0.91      7444
   macro avg       0.85      0.88      0.86      7444
weighted avg       0.92      0.91      0.91      7444

Tuned ROC-AUC Score: 0.9484819952177109`}
    </pre>
  </div>
</div>

          <h5>Tuned Results Analysis:</h5>
          <p><strong>Best Parameters:</strong> C=1, max_iter=500, penalty=l1, solver=liblinear</p>
          <p><strong>Class 0 (Not Paid Off Early):</strong></p>
          <ul>
            <li>Precision improved to 0.96, and recall increased to 0.93, indicating strong performance in correctly identifying loans that were not paid off early.</li>
          </ul>
          <p><strong>Class 1 (Paid Off Early):</strong></p>
          <ul>
            <li>Precision improved significantly to 0.74 (from 0.68) and recall to 0.84 (from 0.80), both showing significant improvement after tuning. The model is now better at correctly identifying loans paid off early while reducing false positives.</li>
          </ul>
          <p><strong>Overall Accuracy:</strong> Increased to 91% (from 88%), which shows clear improvement.</p>
          <p><strong>ROC-AUC Score:</strong> The ROC-AUC score improved to 0.948 (from 0.924), indicating that the tuned model is better at distinguishing between the two classes.</p>
          <p>The hyperparameter tuning successfully improved the model's performance across all metrics, with the most notable improvement being in the precision for class 1, which indicates fewer false positives when predicting early loan payoffs.</p>
        </div>
        
        <h4>Random Forest</h4>
        <p>
          Random Forest is a powerful method that builds multiple decision trees and averages their results. This helps improve model stability and accuracy, particularly with complex data.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('randomForestCode')}
          >
            <span className="code-title">Random Forest Implementation</span>
          </div>
          {visibleSections.randomForestCode && (
            <pre className="code-content">
              <code>{randomForestCode}</code>
            </pre>
          )}
        </div>
        
        <div className="results-box">
          <h4>Random Forest Results</h4>
          <div className="output-box">
  <div className="output-header">
    <span className="output-title">Output</span>
  </div>
  <div className="output-content">
    <pre>
{`Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.95      0.95      5981
           1       0.79      0.81      0.80      1463

    accuracy                           0.92      7444
   macro avg       0.87      0.88      0.87      7444
weighted avg       0.92      0.92      0.92      7444

ROC-AUC Score: 0.9548134483279987`}
    </pre>
  </div>
</div>

          <h5>Untuned Random Forest Insights:</h5>
          <p><strong>Class 0 (Not Paid Off Early):</strong></p>
          <ul>
            <li>Precision, recall, and F1-score are all 0.95, showing excellent performance in identifying loans not paid off early.</li>
          </ul>
          <p><strong>Class 1 (Paid Off Early):</strong></p>
          <ul>
            <li>Precision is 0.79, meaning 79% of loans predicted as paid off early were correct.</li>
            <li>Recall is 0.81, indicating that the model captured 81% of actual early loan payoffs.</li>
          </ul>
          <p><strong>Accuracy:</strong> The model achieved 92% accuracy, which is strong but doesn't fully reflect performance on the minority class (1).</p>
          <p><strong>ROC-AUC Score:</strong> The untuned model has an ROC-AUC score of 0.95, showing very good ability to distinguish between the two classes.</p>
          <p>Even without tuning, Random Forest outperforms the tuned Logistic Regression model, particularly in its ability to correctly identify class 1 (early payoffs) with fewer false positives.</p>
        </div>
        
        <h4>Tuned Random Forest</h4>
        <p>
          RandomizedSearchCV was chosen over GridSearchCV because Random Forest has many hyperparameters, and searching all combinations exhaustively would be too computationally expensive.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('tuningRFCode')}
          >
            <span className="code-title">Tuning Random Forest</span>
          </div>
          {visibleSections.tuningRFCode && (
            <pre className="code-content">
              <code>{tuningRFCode}</code>
            </pre>
          )}
        </div>
        
        <div className="results-box">
          <h4>Tuned Random Forest Results</h4>
          <div className="output-box">
  <div className="output-header">
    <span className="output-title">Output</span>
  </div>
  <div className="output-content">
    <pre>
{`Starting RandomizedSearchCV...
Fitting 5 folds for each of 10 candidates, totalling 50 fits
Best parameters found: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 50, 'bootstrap': False}
Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.95      0.95      5981
           1       0.79      0.80      0.80      1463

    accuracy                           0.92      7444
   macro avg       0.87      0.87      0.87      7444
weighted avg       0.92      0.92      0.92      7444

Tuned ROC-AUC Score: 0.9547423071213319`}
    </pre>
  </div>
</div>

          <h5>Tuned Random Forest Insights:</h5>
          <p><strong>Best Parameters:</strong></p>
          <ul>
            <li>n_estimators: 100, min_samples_split: 5, min_samples_leaf: 2, max_features: 'sqrt', max_depth: 50, bootstrap: False.</li>
          </ul>
          <p><strong>Class 0 (Not Paid Off Early):</strong></p>
          <ul>
            <li>Precision and recall remain at 0.95, indicating consistent performance for this class.</li>
          </ul>
          <p><strong>Class 1 (Paid Off Early):</strong></p>
          <ul>
            <li>Precision is 0.79 (same as untuned), and recall is 0.80 (slightly lower than untuned's 0.81), but the overall F1-score remains unchanged.</li>
          </ul>
          <p><strong>Accuracy:</strong> The model's accuracy is still 92%, showing no significant gain in this metric post-tuning.</p>
          <p><strong>ROC-AUC Score:</strong> The ROC-AUC score after tuning is 0.9547, essentially the same as the untuned model (0.9548).</p>
          <p><strong>Comparison:</strong> The tuned model didn't result in a major improvement, likely due to the already strong performance of the untuned Random Forest. However, the tuning process did identify optimal parameters, confirming that the model was already close to its optimal configuration.</p>
        </div>
        
        <h4>XGBoost</h4>
        <p>
          XGBoost is a gradient boosting algorithm that is highly effective at handling structured/tabular data, making it ideal for this dataset.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('xgboostCode')}
          >
            <span className="code-title">XGBoost Implementation</span>
          </div>
          {visibleSections.xgboostCode && (
            <pre className="code-content">
              <code>{xgboostCode}</code>
            </pre>
          )}
        </div>
        
        <div className="results-box">
          <h4>XGBoost Results</h4>
          <div className="output-box">
          <div className="output-header">
            <span className="output-title">Output</span>
          </div>
          <div className="output-content">
            <pre>
        {`Classification Report:
                      precision    recall  f1-score   support
        
                   0       0.95      0.96      0.95      5981
                   1       0.82      0.80      0.81      1463
        
            accuracy                           0.93      7444
           macro avg       0.89      0.88      0.88      7444
        weighted avg       0.92      0.93      0.93      7444
        
        XGBoost ROC-AUC Score: 0.954598881877369`}
            </pre>
          </div>
        </div>
        
          <h5>Untuned XGBoost Results:</h5>
          <p><strong>Class 0 (Not Paid Off Early):</strong></p>
          <ul>
            <li>Precision is 0.95 and recall is 0.96, indicating strong performance in identifying loans that were not paid off early.</li>
          </ul>
          <p><strong>Class 1 (Paid Off Early):</strong></p>
          <ul>
            <li>Precision is 0.82, meaning 82% of the loans predicted as paid off early were correct, which is higher than both Logistic Regression and Random Forest.</li>
            <li>Recall is 0.80, showing that the model captured 80% of actual early loan payoffs.</li>
          </ul>
          <p><strong>Overall Accuracy:</strong> The model achieved 93% accuracy, reflecting strong overall performance and improvement over previous models.</p>
          <p><strong>ROC-AUC Score:</strong> The ROC-AUC score of 0.9546 suggests the model is very good at distinguishing between loans paid off early and those that were not.</p>
          <p>XGBoost outperforms both Logistic Regression and Random Forest in terms of precision for class 1, showing fewer false positives when predicting early loan payoffs. It also slightly outperforms in terms of overall accuracy.</p>
        </div>
        
        <h4>Tuned XGBoost</h4>
        <p>
          XGBoost tuned with RandomizedSearchCV was once again chosen due to the large number of hyperparameters and their wide range of values.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('tuningXGBCode')}
          >
            <span className="code-title">Tuning XGBoost</span>
          </div>
          {visibleSections.tuningXGBCode && (
            <pre className="code-content">
              <code>{tuningXGBCode}</code>
            </pre>
          )}
        </div>
        
        <div className="results-box">
          <h4>Tuned XGBoost Results</h4>
          <div className="output-box">
  <div className="output-header">
    <span className="output-title">Output</span>
  </div>
  <div className="output-content">
    <pre>
{`Fitting 3 folds for each of 10 candidates, totalling 30 fits
Best parameters found: {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.3, 'colsample_bytree': 1.0}
Classification Report after tuning:
              precision    recall  f1-score   support

           0       0.95      0.95      0.95      5981
           1       0.79      0.79      0.79      1463

    accuracy                           0.92      7444
   macro avg       0.87      0.87      0.87      7444
weighted avg       0.92      0.92      0.92      7444

Tuned ROC-AUC Score: 0.9564974664016366`}
    </pre>
  </div>
</div>

          <h5>Tuned XGBoost Insights:</h5>
          <p><strong>Best Parameters:</strong></p>
          <ul>
            <li>'subsample': 0.8, 'n_estimators': 50, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.3, 'colsample_bytree': 1.0.</li>
          </ul>
          <p><strong>Class 0 (Not Paid Off Early):</strong></p>
          <ul>
            <li>The precision remains at 0.95, but recall decreased from 0.96 to 0.95 compared to the untuned model.</li>
          </ul>
          <p><strong>Class 1 (Paid Off Early):</strong></p>
          <ul>
            <li>After tuning, precision dropped from 0.82 to 0.79, and recall decreased slightly from 0.80 to 0.79. This indicates a reduction in performance for class 1.</li>
          </ul>
          <p><strong>Overall Accuracy:</strong> The accuracy decreased slightly from 93% to 92%.</p>
          <p><strong>ROC-AUC Score:</strong> After tuning, the ROC-AUC score improved slightly to 0.9565 (from 0.9546), indicating a marginal improvement in distinguishing between early and non-early payoffs despite the drop in other metrics.</p>
          <p><strong>Comparison:</strong></p>
          <ul>
            <li>Surprisingly, the untuned model performs better in terms of precision and recall for both classes.</li>
            <li>The slight improvement in ROC-AUC score suggests that the tuned model may generate better probability estimates, even though its classification performance decreased.</li>
            <li>This outcome highlights the importance of evaluating models on multiple metrics and not just optimizing for a single value like ROC-AUC.</li>
          </ul>
        </div>
        
        <h4>LightGBM</h4>
        <p>
          LightGBM and XGBoost are both gradient boosting algorithms, but LightGBM is known for its faster performance.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('lightgbmCode')}
          >
            <span className="code-title">LightGBM Implementation</span>
          </div>
          {visibleSections.lightgbmCode && (
            <pre className="code-content">
              <code>{lightgbmCode}</code>
            </pre>
          )}
        </div>
        
        <div className="results-box">
          <h4>LightGBM Results</h4>
          <div className="output-box">
  <div className="output-header">
    <span className="output-title">Output</span>
  </div>
  <div className="output-content">
    <pre>
{`[LightGBM] [Info] Number of positive: 24024, number of negative: 24024
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.002426 seconds.
You can set \`force_row_wise=true\` to remove the overhead.
And if memory is not enough, you can set \`force_col_wise=true\`.
[LightGBM] [Info] Total Bins 2307
[LightGBM] [Info] Number of data points in the train set: 48048, number of used features: 26
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.95      0.95      5981
           1       0.81      0.80      0.80      1463

    accuracy                           0.92      7444
   macro avg       0.88      0.88      0.88      7444
weighted avg       0.92      0.92      0.92      7444

ROC-AUC Score: 0.9572228781435128`}
    </pre>
  </div>
</div>

          <h5>Untuned LightGBM results:</h5>
          <p><strong>Class 0 (Not Paid Off Early):</strong></p>
          <ul>
            <li>Precision and recall are both 0.95, indicating strong performance in identifying loans that were not paid off early.</li>
          </ul>
          <p><strong>Class 1 (Paid Off Early):</strong></p>
          <ul>
            <li>Precision is 0.81, meaning that 81% of the loans predicted as paid off early were correct.</li>
            <li>Recall is 0.80, showing that the model successfully captured 80% of actual early loan payoffs.</li>
          </ul>
          <p><strong>Overall Accuracy:</strong> The model achieved 92% accuracy, reflecting strong overall performance.</p>
          <p><strong>ROC-AUC Score:</strong> The untuned model has an excellent ROC-AUC score of 0.9572, suggesting that it is very good at distinguishing between early and non-early loan payoffs and slightly outperforms XGBoost in this metric.</p>
          <p>LightGBM shows performance similar to XGBoost for class 0 but with slightly worse precision for class 1. However, it achieves the highest ROC-AUC score among all models, indicating superior ability to rank predictions correctly.</p>
        </div>
        
        <h4>Tuned LightGBM</h4>
        <p>
          I used RandomizedSearchCV again for tuning LightGBM, similar to how I tuned XGBoost. This allows for a direct comparison between the two models.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('tuningLGBMCode')}
          >
            <span className="code-title">Tuning LightGBM</span>
          </div>
          {visibleSections.tuningLGBMCode && (
            <pre className="code-content">
              <code>{tuningLGBMCode}</code>
            </pre>
          )}
        </div>
        
        <div className="results-box">
          <h4>Tuned LightGBM Results</h4>
          <div className="output-box">
  <div className="output-header">
    <span className="output-title">Output</span>
  </div>
  <div className="output-content">
    <pre>
{`Starting RandomizedSearchCV for LightGBM...
Best parameters found for LightGBM: {'subsample': 0.8, 'num_leaves': 500, 'n_estimators': 100, 'min_child_samples': 20, 'max_depth': -1, 'learning_rate': 0.1, 'colsample_bytree': 0.8}
LightGBM Classification Report after tuning:
              precision    recall  f1-score   support

           0       0.95      0.96      0.95      5981
           1       0.83      0.79      0.81      1463

    accuracy                           0.93      7444
   macro avg       0.89      0.87      0.88      7444
weighted avg       0.92      0.93      0.92      7444

Tuned LightGBM ROC-AUC Score: 0.9559887924885857`}
    </pre>
  </div>
</div>

          <h5>Tuned LightGBM results:</h5>
          <p><strong>Best Parameters:</strong></p>
          <ul>
            <li>'subsample': 0.8, 'num_leaves': 500, 'n_estimators': 100, 'min_child_samples': 20, 'max_depth': -1, 'learning_rate': 0.1, 'colsample_bytree': 0.8.</li>
          </ul>
          <p><strong>Class 0 (Not Paid Off Early):</strong></p>
          <ul>
            <li>Precision remains high at 0.95, and recall improves slightly to 0.96, maintaining strong performance for this class.</li>
          </ul>
          <p><strong>Class 1 (Paid Off Early):</strong></p>
          <ul>
            <li>Precision improved to 0.83 (from 0.81), meaning a higher percentage of loans predicted as early payoffs were correct.</li>
            <li>However, recall slightly decreased to 0.79 (from 0.80), indicating the model caught slightly fewer actual early payoffs after tuning.</li>
          </ul>
          <p><strong>Overall Accuracy:</strong> Accuracy improved to 93% (from 92%), showing a slight gain after tuning.</p>
          <p><strong>ROC-AUC Score:</strong> The tuned ROC-AUC score is 0.9560, a small decrease from the untuned version (0.9572), but still indicating strong performance.</p>
          <p><strong>Comparison:</strong></p>
          <ul>
            <li>The tuned model has a small improvement in precision for class 1 but a slight decrease in recall.</li>
            <li>Overall, the performance remains strong, with the tuned model achieving a marginally higher accuracy but a slightly lower ROC-AUC score than the untuned version.</li>
            <li>Among all models tested, the tuned LightGBM has the highest precision for class 1 (0.83), making it potentially the best model when false positives are a concern.</li>
          </ul>
        </div>
        
        <div className="results-box">
          <h4>Model Comparison Summary</h4>
          <p>After evaluating the four different models, here's a comparison of their performance:</p>
          <table className="model-comparison-table">
            <thead>
            
            <tr>
              <th>Model</th>
              <th>Precision (Class 1)</th>
              <th>Recall (Class 1)</th>
              <th>Accuracy</th>
              <th>ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Logistic Regression</td>
              <td>0.68</td>
              <td>0.80</td>
              <td>0.88</td>
              <td>0.9246</td>
            </tr>
            <tr>
              <td>Tuned Logistic Regression</td>
              <td>0.74</td>
              <td>0.84</td>
              <td>0.91</td>
              <td>0.9485</td>
            </tr>
            <tr>
              <td>Random Forest</td>
              <td>0.79</td>
              <td>0.81</td>
              <td>0.92</td>
              <td>0.9548</td>
            </tr>
            <tr>
              <td>Tuned Random Forest</td>
              <td>0.79</td>
              <td>0.80</td>
              <td>0.92</td>
              <td>0.9547</td>
            </tr>
            <tr>
              <td>XGBoost</td>
              <td>0.82</td>
              <td>0.80</td>
              <td>0.93</td>
              <td>0.9546</td>
            </tr>
            <tr>
              <td>Tuned XGBoost</td>
              <td>0.79</td>
              <td>0.79</td>
              <td>0.92</td>
              <td>0.9565</td>
            </tr>
            <tr>
              <td>LightGBM</td>
              <td>0.81</td>
              <td>0.80</td>
              <td>0.92</td>
              <td>0.9572</td>
            </tr>
            <tr>
              <td>Tuned LightGBM</td>
              <td>0.83</td>
              <td>0.79</td>
              <td>0.93</td>
              <td>0.9560</td>
        
              </tr>
            </tbody>
          </table>
          <p><strong>Key Findings:</strong></p>
          <ul>
            <li>The tuned LightGBM model achieved the best precision for class 1 (0.83), making it the best choice when false positives are a concern.</li>
            <li>The tuned Logistic Regression model achieved the best recall for class 1 (0.84), making it the best at identifying actual early payoffs.</li>
            <li>XGBoost and tuned LightGBM tied for the best overall accuracy (0.93%).</li>
            <li>The untuned LightGBM model had the best ROC-AUC score (0.9572), indicating the best ability to rank predictions correctly.</li>
            <li>Tree-based models (Random Forest, XGBoost, LightGBM) consistently outperformed Logistic Regression in overall accuracy and precision for class 1.</li>
          </ul>
          <p><strong>Final Recommendation:</strong> The tuned LightGBM model offers the best balance of precision, accuracy, and ROC-AUC score, making it the recommended model for predicting early loan payoffs.</p>
        </div>
      </div>
    </div>
  );
};

export default ModelTraining;
