import React, { useState } from 'react';
import './sections.css';
import CodeBlock from '../common/CodeBlock';

const Introduction = () => {
  // State to track which code sections are visible
  const [visibleSections, setVisibleSections] = useState({
    pythonImports: false,
    helperFunctions: false
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
      <h2>Introduction</h2>
      
      <div className="project-goals">
        <h3>Project Goals</h3>
        <ul>
          <li>Understand factors that influence early loan closings</li>
          <li>Build a predictive model to identify customers likely to close loans early</li>
          <li>Provide actionable insights for the bank to mitigate financial impacts</li>
          <li>Compare multiple machine learning algorithms to find the optimal solution</li>
        </ul>
      </div>
      
      <div className="intro-overview">
        <p>
          In this banking case study, we're building a machine learning solution to predict <strong>early loan closing</strong>. 
          Early loan closings impact a bank's expected interest income and can disrupt financial planning. By identifying 
          customers likely to close loans early, banks can develop targeted retention strategies or adjust pricing models.
        </p>
        
        <p>
          We'll go through the entire data science workflow: data exploration, feature engineering, model building, 
          and evaluation of multiple algorithms to find the most effective predictor.
        </p>
      </div>
      
      <div className="use-case-box">
        <h3>Use Case: Early Loan Closing Prediction</h3>
      </div>
      
      <div className="dataset-section">
        <h3>1) The Data</h3>
        
        <div className="dataset-description">
          <h4>Dataset: use_case_customer_data.csv</h4>
          <p>9561 rows x 7 columns</p>
          
          <table className="data-table">
            <thead>
              <tr>
                <th>Variable</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>CUSTOMER_ID</td>
                <td>Customer ID</td>
              </tr>
              <tr>
                <td>SEX</td>
                <td>Gender of the customer (M: MAN, W: WOMAN)</td>
              </tr>
              <tr>
                <td>AGE</td>
                <td>Age of the customer</td>
              </tr>
              <tr>
                <td>ANNUAL_INCOME</td>
                <td>Annual salary value of the customer</td>
              </tr>
              <tr>
                <td>NUMBER_OF_MONTHS</td>
                <td>Number of months the salary is paid</td>
              </tr>
              <tr>
                <td>MARITAL_STATUS</td>
                <td>Marital status of the customer (D: DIVORCED, G: SINGLE, C: COHABITANT, J: CONJUGATE, S: SEPARATED, W: WIDOW(ER), X: OTHER)</td>
              </tr>
              <tr>
                <td>LEASE</td>
                <td>Type of customer lease (P: PROPERTY, E: AT THE EMPLOYER, R: RENT, A: PARENTS/RELATIVES, T: THIRD PARTIES, X: OTHER)</td>
              </tr>
            </tbody>
          </table>
        </div>
        
        <div className="dataset-description">
          <h4>Dataset: use_case_loans_data.csv</h4>
          <p>37291 rows x 12 variables</p>
          
          <table className="data-table">
            <thead>
              <tr>
                <th>Variable</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>CUSTOMER_ID</td>
                <td>Customer ID</td>
              </tr>
              <tr>
                <td>STATUS</td>
                <td>Loan status (target): CONCLUDED REGULARLY or EARLY EXPIRED (phenomenon of interest)</td>
              </tr>
              <tr>
                <td>SECTOR_TYPE</td>
                <td>Type of loan (CL: CAR LOAN, FL: FINALIZED LOAN, PL: PERSONAL LOAN)</td>
              </tr>
              <tr>
                <td>GOOD_VALUE</td>
                <td>Value of the mortgaged property</td>
              </tr>
              <tr>
                <td>ADVANCE_VALUE</td>
                <td>Advance paid</td>
              </tr>
              <tr>
                <td>LOAN_VALUE</td>
                <td>Value of the loan</td>
              </tr>
              <tr>
                <td>INSTALLMENT_VALUE</td>
                <td>Value of the installment</td>
              </tr>
              <tr>
                <td>NUMBER_INSTALLMENT</td>
                <td>Number of installments</td>
              </tr>
              <tr>
                <td>GAPR</td>
                <td>Gross Annual Percentage Rate</td>
              </tr>
              <tr>
                <td>NIR</td>
                <td>Nominal Interest Rate</td>
              </tr>
              <tr>
                <td>REFINANCED</td>
                <td>Loan subject to refinancing (Y / N)</td>
              </tr>
              <tr>
                <td>FROM_REFINANCE</td>
                <td>Loan from a refinancing (Y / N)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      
      <div className="imports-section">
        <h3>Libraries and Tools</h3>
        <CodeBlock 
          title="Python Imports"
          codeContent={`import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from scipy import stats`}
        />
        
        <CodeBlock 
          title="Helper Function"
          codeContent={`# Separator line function for outputs
def separate():
    print("\\n" + "-" * 50 + "\\n")`}
        />
      </div>
    </div>
  );
};

export default Introduction;
