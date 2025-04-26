import React, { useState } from 'react';
import './sections.css';
import CodeBlock from '../common/CodeBlock';

const DataPreparation = () => {
  // State to track which code sections are visible
  const [visibleSections, setVisibleSections] = useState({
    dataLoadingCode: false,
    dataCleaningCode: false,
    missingValuesCode: false,
    mergeDataCode: false,
    dataTypeCode: false
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
      <h2 id="I.-Data-Preparation">I. Data Preparation</h2>
      
      <div className="subsection">
        <h3>a) Preparing the dataset</h3>
        <p>
          We start by importing the necessary libraries and loading the data from two separate CSV files:
          one for customer data and one for loan data.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('dataLoadingCode')}
          >
            <span className="code-title">Loading Data</span>
          </div>
          {visibleSections.dataLoadingCode && (
            <pre className="code-content">
              <code>{`#Loading data files
cust_data = pd.read_csv('data/use_case_customer_data.csv')
loan_data = pd.read_csv('data/use_case_loans_data.csv')`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Data Preview</span>
          </div>
          <div className="output-content">
            <pre>{`#Confirming data is loaded
cust_data.head()
CUSTOMER_ID	SEX	AGE	ANNUAL_INCOME	NUMBER_OF_MONTHS	MARITAL_STATUS	LEASE
0	1088	M	36	25200.00	14	J	P
1	1097	W	45	26610.78	14	J	P
2	1102	M	49	24700.00	13	J	P
3	1104	W	45	15951.00	13	J	P
4	1106	M	47	28114.45	13	J	P

loan_data.head()
CUSTOMER_ID	STATUS	SECTOR_TYPE	GOOD_VALUE	ADVANCE_VALUE	LOAN_VALUE	INSTALLMENT_VALUE	NUMBER_INSTALLMENT	GAPR	NIR	REFINANCED	FROM_REFINANCE
0	1088	CONCLUDED REGULARLY	FL	469.00	69.0	400.00	44.45	10	21.74882	19.84123	N	N
1	1088	CONCLUDED REGULARLY	FL	794.30	0.0	794.30	70.00	12	9.42200	9.03804	N	N
2	1097	CONCLUDED REGULARLY	FL	399.00	0.0	419.00	69.85	6	8.19200	0.03800	N	N
3	1097	CONCLUDED REGULARLY	FL	1039.98	0.0	1039.98	52.00	20	0.00220	0.00220	N	N
4	1097	EARLY EXPIRED	CL	23500.00	3500.0	21387.86	310.00	84	6.72602	5.76090	Y	N`}</pre>
          </div>
        </div>
      </div>
      
      <div className="subsection">
        <h3>b) Dealing with missing values</h3>
        <p>Let's examine the datasets to understand their characteristics and clean any issues.</p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('dataCleaningCode')}
          >
            <span className="code-title">Data Exploration</span>
          </div>
          {visibleSections.dataCleaningCode && (
            <pre className="code-content">
              <code>{`#Inspecting the data, checking columns, data types and missing values
cust_data.info()
separate()
loan_data.info()
separate()
print(cust_data.isnull().sum())
separate()
print(loan_data.isnull().sum())`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Data Information</span>
          </div>
          <div className="output-content">
            <pre>{`<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9561 entries, 0 to 9560
Data columns (total 7 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   CUSTOMER_ID       9561 non-null   int64  
 1   SEX               9561 non-null   object 
 2   AGE               9561 non-null   int64  
 3   ANNUAL_INCOME     9561 non-null   float64
 4   NUMBER_OF_MONTHS  9561 non-null   int64  
 5   MARITAL_STATUS    9546 non-null   object 
 6   LEASE             9559 non-null   object 
dtypes: float64(1), int64(3), object(3)
memory usage: 523.0+ KB

--------------------------------------------------

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 37291 entries, 0 to 37290
Data columns (total 12 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   CUSTOMER_ID         37291 non-null  int64  
 1   STATUS              37291 non-null  object 
 2   SECTOR_TYPE         37291 non-null  object 
 3   GOOD_VALUE          37291 non-null  float64
 4   ADVANCE_VALUE       37291 non-null  float64
 5   LOAN_VALUE          37291 non-null  float64
 6   INSTALLMENT_VALUE   37291 non-null  float64
 7   NUMBER_INSTALLMENT  37291 non-null  int64  
 8   GAPR                37291 non-null  float64
 9   NIR                 37291 non-null  float64
 10  REFINANCED          37291 non-null  object 
 11  FROM_REFINANCE      37291 non-null  object 
dtypes: float64(6), int64(2), object(4)
memory usage: 3.4+ MB

--------------------------------------------------

CUSTOMER_ID          0
SEX                  0
AGE                  0
ANNUAL_INCOME        0
NUMBER_OF_MONTHS     0
MARITAL_STATUS      15
LEASE                2
dtype: int64

--------------------------------------------------

CUSTOMER_ID           0
STATUS                0
SECTOR_TYPE           0
GOOD_VALUE            0
ADVANCE_VALUE         0
LOAN_VALUE            0
INSTALLMENT_VALUE     0
NUMBER_INSTALLMENT    0
GAPR                  0
NIR                   0
REFINANCED            0
FROM_REFINANCE        0
dtype: int64`}</pre>
          </div>
        </div>
        
        <div className="analysis-notes">
          <h4>Dropping Missing Data</h4>
          <p>
            The missing data accounts for only 15 rows (missing leases were also missing marital status) out of 37,000, which is a negligible amount (~0.04% of the data).
            The missing values relate to personal information like marital status, which cannot be reliably inferred or filled.
            Given the small proportion and the nature of the data, dropping these rows won't significantly affect the analysis.
          </p>
        </div>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('missingValuesCode')}
          >
            <span className="code-title">Handling Missing Values</span>
          </div>
          {visibleSections.missingValuesCode && (
            <pre className="code-content">
              <code>{`#Dropping rows with missing data
cust_data_clean = cust_data.dropna(subset=['MARITAL_STATUS', 'LEASE'])

# Confirming rows were dropped
print(cust_data_clean.isnull().sum())

print("Original shape:", cust_data.shape)
print("Cleaned shape:", cust_data_clean.shape)`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Missing Values Check</span>
          </div>
          <div className="output-content">
            <pre>{`CUSTOMER_ID         0
SEX                 0
AGE                 0
ANNUAL_INCOME       0
NUMBER_OF_MONTHS    0
MARITAL_STATUS      0
LEASE               0
dtype: int64
Original shape: (9561, 7)
Cleaned shape: (9546, 7)`}</pre>
          </div>
        </div>
      </div>
      
      <div className="subsection">
        <h3>c) Merging the Datasets</h3>
        <p>
          Now we'll merge the customer and loan datasets to create a comprehensive dataset for our analysis.
          We'll merge on the CUSTOMER_ID field, which is present in both datasets.
        </p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('mergeDataCode')}
          >
            <span className="code-title">Merging Data</span>
          </div>
          {visibleSections.mergeDataCode && (
            <pre className="code-content">
              <code>{`#Inner join
merged_data = pd.merge(cust_data_clean, loan_data, on='CUSTOMER_ID', how='inner')

print(merged_data.head())
separate()
print(merged_data.info())
separate()
print(merged_data.isnull().sum())`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Merged Data Preview</span>
          </div>
          <div className="output-content">
            <pre>{`   CUSTOMER_ID SEX  AGE  ANNUAL_INCOME  NUMBER_OF_MONTHS MARITAL_STATUS LEASE  \\
0         1088   M   36       25200.00                14              J     P   
1         1088   M   36       25200.00                14              J     P   
2         1097   W   45       26610.78                14              J     P   
3         1097   W   45       26610.78                14              J     P   
4         1097   W   45       26610.78                14              J     P   

                STATUS SECTOR_TYPE  GOOD_VALUE  ADVANCE_VALUE  LOAN_VALUE  \\
0  CONCLUDED REGULARLY          FL      469.00           69.0      400.00   
1  CONCLUDED REGULARLY          FL      794.30            0.0      794.30   
2  CONCLUDED REGULARLY          FL      399.00            0.0      419.00   
3  CONCLUDED REGULARLY          FL     1039.98            0.0     1039.98   
4        EARLY EXPIRED          CL    23500.00         3500.0    21387.86   

   INSTALLMENT_VALUE  NUMBER_INSTALLMENT      GAPR       NIR REFINANCED  \\
0              44.45                  10  21.74882  19.84123          N   
1              70.00                  12   9.42200   9.03804          N   
2              69.85                   6   8.19200   0.03800          N   
3              52.00                  20   0.00220   0.00220          N   
4             310.00                  84   6.72602   5.76090          Y   

  FROM_REFINANCE  
0              N  
1              N  
2              N  
3              N  
4              N  

--------------------------------------------------

<class 'pandas.core.frame.DataFrame'>
Int64Index: 37219 entries, 0 to 37218
Data columns (total 18 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   CUSTOMER_ID         37219 non-null  int64  
 1   SEX                 37219 non-null  object 
 2   AGE                 37219 non-null  int64  
 3   ANNUAL_INCOME       37219 non-null  float64
 4   NUMBER_OF_MONTHS    37219 non-null  int64  
 5   MARITAL_STATUS      37219 non-null  object 
 6   LEASE               37219 non-null  object 
 7   STATUS              37219 non-null  object 
 8   SECTOR_TYPE         37219 non-null  object 
 9   GOOD_VALUE          37219 non-null  float64
 10  ADVANCE_VALUE       37219 non-null  float64
 11  LOAN_VALUE          37219 non-null  float64
 12  INSTALLMENT_VALUE   37219 non-null  float64
 13  NUMBER_INSTALLMENT  37219 non-null  int64  
 14  GAPR                37219 non-null  float64
 15  NIR                 37219 non-null  float64
 16  REFINANCED          37219 non-null  object 
 17  FROM_REFINANCE      37219 non-null  object 
dtypes: float64(7), int64(4), object(7)
memory usage: 5.4+ MB
None

--------------------------------------------------

CUSTOMER_ID           0
SEX                   0
AGE                   0
ANNUAL_INCOME         0
NUMBER_OF_MONTHS      0
MARITAL_STATUS        0
LEASE                 0
STATUS                0
SECTOR_TYPE           0
GOOD_VALUE            0
ADVANCE_VALUE         0
LOAN_VALUE            0
INSTALLMENT_VALUE     0
NUMBER_INSTALLMENT    0
GAPR                  0
NIR                   0
REFINANCED            0
FROM_REFINANCE        0
dtype: int64`}</pre>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataPreparation;
