import React, { useState } from 'react';
import './sections.css';
import CodeBlock from '../common/CodeBlock';

const FeatureEngineering = () => {
  // State to track which code sections are visible
  const [visibleSections, setVisibleSections] = useState({
    scalingCode: false,
    encodingCode: false,
    balancingCode: false,
    smoteResults: false
  });

  // Toggle visibility for a specific section
  const toggleSection = (section) => {
    setVisibleSections(prevState => ({
      ...prevState,
      [section]: !prevState[section]
    }));
  };
  
  // Sample code content for one of the sections
  const scalingCodeContent = `#Im using MinMaxScaler for everything as only AGE is normally distributed
#The rest of the data is skewed, which MinMaxScaler can handle better

merged_data = pd.merge(cust_data_clean, loan_data, on='CUSTOMER_ID', how='inner')

minmax_columns = [
    'AGE', 'ANNUAL_INCOME', 'LOAN_VALUE', 'GOOD_VALUE', 
    'ADVANCE_VALUE', 'NUMBER_OF_MONTHS', 'NUMBER_INSTALLMENT', 'GAPR', 'NIR', 'INSTALLMENT_VALUE'
]

#Initializing the Scaler
minmax_scaler = MinMaxScaler()

#Scaling the data using MinMaxScaler
minmax_scaled_data = minmax_scaler.fit_transform(merged_data[minmax_columns])

#Converting the scaled data back to a DataFrame
minmax_scaled_df = pd.DataFrame(minmax_scaled_data, columns=minmax_columns)

#Verifying results
print(minmax_scaled_df.head())`;

  return (
    <div className="section-content">
      <h2 id="III.-Feature-Engineering">III. Feature Engineering</h2>
      
      <div className="section-overview">
        <p>
          Before training our model, we need to prepare the features properly. This includes scaling numerical data, 
          encoding categorical variables, and addressing the class imbalance in our dataset.
        </p>
      </div>
      
      <div className="subsection">
        <h3>a) Scaling numerical data</h3>
        <p>
          In this stage, I'm applying MinMaxScaler to scale all features.
          AGE is the only feature that is normally distributed, while the rest of the data is skewed.
          MinMaxScaler is well-suited for this scenario because it scales features to a range between 0 and 1, preserving the shape of the distribution, which works better with skewed data.
        </p>
        
        <CodeBlock 
          title="Scaling Numerical Features"
          codeContent={scalingCodeContent}
        />
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">MinMaxScaled Data</span>
          </div>
          <div className="output-content">
            <pre>{`        AGE  ANNUAL_INCOME  LOAN_VALUE  GOOD_VALUE  ADVANCE_VALUE  \\
0  0.058824       0.008182    0.002478    0.003086       0.000952   
1  0.058824       0.008182    0.005724    0.005799       0.000000   
2  0.235294       0.008640    0.002635    0.002502       0.000000   
3  0.235294       0.008640    0.007747    0.007848       0.000000   
4  0.235294       0.008640    0.175273    0.195169       0.048276   

   NUMBER_OF_MONTHS  NUMBER_INSTALLMENT      GAPR       NIR  INSTALLMENT_VALUE  
0               1.0            0.050279  0.492768  0.198546           0.003742  
1               1.0            0.061453  0.213676  0.090532           0.006517  
2               1.0            0.027933  0.185827  0.000547           0.006501  
3               1.0            0.106145  0.000402  0.000189           0.004562  
4               1.0            0.463687  0.152636  0.057766           0.032587`}</pre>
          </div>
        </div>
      </div>
      
      <div className="subsection">
        <h3>b) Encoding categorical data</h3>
        <p>
          Categorical data needs to be converted to numeric format before it can be used in machine learning models. 
          I'll use different encoding strategies based on the nature of each categorical variable.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('encodingCode')}
          >
            <span className="code-title">Encoding Categorical Features</span>
          </div>
          {visibleSections.encodingCode && (
            <pre className="code-content">
              <code>{`#Initializing LabelEncoder for binary columns (SEX, REFINANCED, FROM_REFINANCE)
le = LabelEncoder()

#Applying label encoding to binary columns 
merged_data['SEX'] = le.fit_transform(merged_data['SEX']) # M/W -> 0/1
merged_data['REFINANCED'] = le.fit_transform(merged_data['REFINANCED']) # Y/N -> 0/1
merged_data['FROM_REFINANCE'] = le.fit_transform(merged_data['FROM_REFINANCE']) # Y/N -> 0/1

#Printing the first few rows to check label encoding results
print(merged_data[['SEX', 'REFINANCED', 'FROM_REFINANCE']].head())

#Applying one hot encoding (dropping the first category to avoid multicollinearity-can be detected by absence)
merged_data = pd.get_dummies(merged_data, columns=['MARITAL_STATUS', 'LEASE', 'STATUS', 'SECTOR_TYPE'], drop_first=True)

#Printing the first few rows to check the one hot encoding result
print(merged_data.head())`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Label Encoded Binary Features</span>
          </div>
          <div className="output-content">
            <pre>{`   SEX  REFINANCED  FROM_REFINANCE
0    0           0               0
1    0           0               0
2    1           0               0
3    1           0               0
4    1           1               0`}</pre>
          </div>
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">One-Hot Encoded Features</span>
          </div>
          <div className="output-content">
            <pre>{`   CUSTOMER_ID  SEX  AGE  ANNUAL_INCOME  NUMBER_OF_MONTHS  GOOD_VALUE  \\
0         1088    0   36       25200.00                14      469.00   
1         1088    0   36       25200.00                14      794.30   
2         1097    1   45       26610.78                14      399.00   
3         1097    1   45       26610.78                14     1039.98   
4         1097    1   45       26610.78                14    23500.00   

   ADVANCE_VALUE  LOAN_VALUE  INSTALLMENT_VALUE  NUMBER_INSTALLMENT  ...  \\
0           69.0      400.00              44.45                  10  ...   
1            0.0      794.30              70.00                  12  ...   
2            0.0      419.00              69.85                   6  ...   
3            0.0     1039.98              52.00                  20  ...   
4         3500.0    21387.86             310.00                  84  ...   

   MARITAL_STATUS_W  MARITAL_STATUS_X  LEASE_E  LEASE_P  LEASE_R  LEASE_T  \\
0                 0                 0        0        1        0        0   
1                 0                 0        0        1        0        0   
2                 0                 0        0        1        0        0   
3                 0                 0        0        1        0        0   
4                 0                 0        0        1        0        0   

   LEASE_X  STATUS_EARLY EXPIRED  SECTOR_TYPE_FL  SECTOR_TYPE_PL  
0        0                     0               1               0  
1        0                     0               1               0  
2        0                     0               1               0  
3        0                     0               1               0  
4        0                     1               0               0  

[5 rows x 28 columns]`}</pre>
          </div>
        </div>
      </div>
      
      <div className="subsection">
        <h3>c) Dealing with Unbalanced Dataset</h3>
        <p>
          I started by setting up the features (X) and the target variable (y), where the target is STATUS_EARLY EXPIRED (whether a loan was paid off early).
          Looking at the distribution of the target variable, the dataset is unbalanced, with many more instances of class 0 (loans that were not paid off early) than class 1.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('balancingCode')}
          >
            <span className="code-title">Checking Class Distribution</span>
          </div>
          {visibleSections.balancingCode && (
            <pre className="code-content">
              <code>{`#Checking the distribution of the target variable (STATUS_EARLY EXPIRED)
print(merged_data['STATUS_EARLY EXPIRED'].value_counts())`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Target Distribution</span>
          </div>
          <div className="output-content">
            <pre>{`0    30005
1     7214
Name: STATUS_EARLY EXPIRED, dtype: int64`}</pre>
          </div>
        </div>
        
        <p>
          Full dataset: 0: 30,005 vs 1: 7,214
        </p>
        <p>
          I then split the data into training and testing sets, keeping 80% for training and 20% for testing.
          Before SMOTE, the training set was still unbalanced: 0: 24,024 vs 1: 5,751.
          SMOTE generates synthetic examples for the minority class (in this case, 1), balancing the training set.
          After applying SMOTE, both classes have an equal number of instances (24,024 for each), making the data more balanced for training.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('smoteResults')}
          >
            <span className="code-title">Applying SMOTE</span>
          </div>
          {visibleSections.smoteResults && (
            <pre className="code-content">
              <code>{`#Setting up features (X) and the target (y)
X = merged_data.drop(columns=['STATUS_EARLY EXPIRED'])  # Features
y = merged_data['STATUS_EARLY EXPIRED']  # Target

#Displaying the target variable count
print("Full dataset:", y.value_counts())

#Splitting the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
separate()
#Checking class distribution in training data before resampling
print("Before SMOTE:", y_train.value_counts())

#Balancing the training set with SMOTE
smote = SMOTE(random_state=40)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
separate()
#Checking class distribution after SMOTE
print("After SMOTE:", y_train_res.value_counts())`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">SMOTE Results</span>
          </div>
          <div className="output-content">
            <pre>{`Full dataset: 0    30005
1     7214
Name: STATUS_EARLY EXPIRED, dtype: int64

--------------------------------------------------

Before SMOTE: 0    24024
1     5751
Name: STATUS_EARLY EXPIRED, dtype: int64

--------------------------------------------------

After SMOTE: 1    24024
0    24024
Name: STATUS_EARLY EXPIRED, dtype: int64`}</pre>
          </div>
        </div>
        
        <div className="analysis-note">
          <h4>Why I Used SMOTE</h4>
          <ul>
            <li>SMOTE (Synthetic Minority Over-sampling Technique) was chosen because it creates synthetic examples of the minority class (1: loans paid off early) rather than just duplicating existing ones.</li>
            <li>Compared to other methods like random oversampling (which can lead to overfitting by duplicating data), SMOTE introduces variability by generating new, realistic instances based on existing data.</li>
            <li>I avoided undersampling the majority class (removing instances of class 0) because it could result in losing important information from the majority class, reducing the overall amount of training data.</li>
            <li>SMOTE helps balance the data without discarding any valuable information, making it a good choice for this unbalanced dataset.</li>
          </ul>
        </div>
      </div>
      
      <div className="next-steps">
        <h3>Next Steps</h3>
        <p>
          Now that our features are properly engineered, we can proceed to:
        </p>
        <ul>
          <li>Train different machine learning models on our processed dataset</li>
          <li>Evaluate model performance using appropriate metrics</li>
          <li>Fine-tune the best performing model through hyperparameter optimization</li>
          <li>Interpret the final model results and draw business insights</li>
        </ul>
      </div>
    </div>
  );
};

export default FeatureEngineering;
