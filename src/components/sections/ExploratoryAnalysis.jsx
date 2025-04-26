import React, { useState } from 'react';
import './sections.css';
import CodeBlock from '../common/CodeBlock';

const ExploratoryAnalysis = () => {
  // State to track which code sections are visible
  const [visibleSections, setVisibleSections] = useState({
    dataLoadingCode: false,
    descriptiveStatsCode: false,
    distributionCode: false,
    logTransform: false,
    incomeBins: false,
    boxcoxTransform: false,
    installmentTransform: false,
    correlationMatrix: false,
    correlationHeatmap: false
  });

  // Toggle visibility for a specific section
  const toggleSection = (section) => {
    setVisibleSections(prevState => ({
      ...prevState,
      [section]: !prevState[section]
    }));
  };
  
  // Example of code content for distribution analysis
  const distributionCodeContent = `#List of columns to plot
columns_to_plot = [
    'AGE', 'ANNUAL_INCOME', 'LOAN_VALUE', 'INSTALLMENT_VALUE', 'GOOD_VALUE', 
    'ADVANCE_VALUE',  'NUMBER_INSTALLMENT'
]

#Creating a function for setting the title and labels for each subplot
def set_plot_details(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

#Function to plot distributions for multiple columns
def plot_distributions(columns):
    # Setting the size of the plots
    plt.figure(figsize=(16, 12))
    
    #Looping through each column to plot its distribution
    for i, column in enumerate(columns, 1):  # Using enumerate to index subplots
        plt.subplot(2, 4, i)  #Adjusting layout for up to 8 plots (2 rows, 4 columns)
        sns.histplot(merged_data[column], bins=30, kde=True)  #Plotting distribution
        set_plot_details(f'Distribution of {column}', column, 'Frequency')  #Setting labels
    
    #Adjusting layout for spacing
    plt.tight_layout()

    # Showing the plots
    plt.show()

#Plotting the distributions for all columns in the list
plot_distributions(columns_to_plot)`;

  return (
    <div className="section-content">
      <h2 id="II.-Exploratory-Data-Analysis-Data-Visualization">II. Exploratory Data Analysis &amp; Data Visualization</h2>
      
      <div className="subsection">
        <h3>a) Analyzing Distributions</h3>
        <p><em>hint: use histograms or the function <code>distplot</code> of the library <code>seaborn</code></em></p>
        
        <div className="analysis-notes">
          <ul>
            <li>First, I get a brief overview of the most relevant columns.</li>
            <li>From this, I can see that:
              <ul>
                <li><strong>AGE</strong> is normally distributed.</li>
                <li>The rest of the columns (<strong>ANNUAL_INCOME</strong>, <strong>LOAN_VALUE</strong>, <strong>GOOD_VALUE</strong>, etc.) are right-skewed, with most values concentrated in the lower ranges.</li>
              </ul>
            </li>
          </ul>
        </div>
        
        <CodeBlock 
          title="Distribution Analysis"
          codeContent={distributionCodeContent}
        />
        
        <div className="visualization">
          <img src={`${import.meta.env.BASE_URL}key-num-feat.jpg`} alt="Distribution of key numeric features" className="analysis-image" />
        </div>
        
        <div className="analysis-notes">
          <p>Next, I took a closer look at ANNUAL_INCOME, LOAN_VALUE, and INSTALLMENT_VALUE due to their significant right skew.</p>
          <p>These columns represent key financial metrics, and normalizing them could improve model performance and interpretability.</p>
          <p>I applied log or Box Cox transformations to these variables to see if I could achieve a more normal distribution, which often helps with modeling by reducing skewness and making patterns easier to detect.</p>
        </div>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('logTransform')}
          >
            <span className="code-title">Log Transform Annual Income</span>
          </div>
          {visibleSections.logTransform && (
            <pre className="code-content">
              <code>{`#Log-transforming annual income to normalize skewed data, making it easier to analyze
merged_data['LOG_ANNUAL_INCOME'] = np.log1p(merged_data['ANNUAL_INCOME'])

#Plotting the log-transformed annual income distribution to view the normalization effect
plt.figure(figsize=(10, 6))
sns.histplot(merged_data['LOG_ANNUAL_INCOME'], bins=30, kde=True)
plt.title('Log-Transformed Distribution of Annual Income')
plt.xlabel('Log(Annual Income)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()`}</code>
            </pre>
          )}
        </div>
        
        <div className="visualization">
          <img src={`${import.meta.env.BASE_URL}log-tran-dist.jpg`} alt="Log-Transformed Distribution of Annual Income" className="analysis-image" />
        </div>
        
        <div className="analysis-notes">
          <p><strong>Log-Transformed Distribution of Annual Income:</strong></p>
          <ul>
            <li>Log transformation has reduced skewness, bringing the distribution closer to normal.</li>
            <li>Most values now fall between 9 and 11 on the log scale, indicating that most customers have very low to mid income after transformation.</li>
            <li>The log transformation could help improve model performance by normalizing the data, if used for my models.</li>
          </ul>
        </div>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('incomeBins')}
          >
            <span className="code-title">Income Binning</span>
          </div>
          {visibleSections.incomeBins && (
            <pre className="code-content">
              <code>{`#Creating income bins for easier categorization of raw annual income
bins = [0, 25000, 50000, 75000, 100000, 500000]  # Defining the ranges for income categories
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']  # Labeling the income categories
merged_data['INCOME_BIN'] = pd.cut(merged_data['ANNUAL_INCOME'], bins=bins, labels=labels)

#Plotting the binned income data to see distribution across income categories
plt.figure(figsize=(10, 6))
sns.countplot(x='INCOME_BIN', data=merged_data)
plt.title('Binned Distribution of Annual Income')
plt.xlabel('Income Bins')
plt.ylabel('Count')
plt.grid(True)
plt.show()`}</code>
            </pre>
          )}
        </div>
        
        <div className="visualization">
          <img src={`${import.meta.env.BASE_URL}bin-ann-inc.jpg`} alt="Binned Distribution of Annual Income" className="analysis-image" />
        </div>
        
        <div className="analysis-notes">
          <p><strong>Binned Distribution of Annual Income:</strong></p>
          <ul>
            <li>A large number of customers fall into the "Very Low" and "Low" income categories.</li>
            <li>Very few customers fall into the "Medium," "High," or "Very High" income brackets, highlighting income disparity.</li>
            <li>The data is heavily skewed towards lower income levels, which could affect customer financial behavior or loan outcomes.</li>
          </ul>
        </div>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('boxcoxTransform')}
          >
            <span className="code-title">Box-Cox Transform Loan Value</span>
          </div>
          {visibleSections.boxcoxTransform && (
            <pre className="code-content">
              <code>{`#Applying Box-Cox transformation to Loan Value to reduce skewness
#Since Box-Cox requires strictly positive values, adding 1 to avoid zero values
loan_value_boxcox, _ = stats.boxcox(merged_data['LOAN_VALUE'] + 1)

#Plotting the Box-Cox transformed loan value distribution to visualize the normalization effect
plt.figure(figsize=(10, 6))
sns.histplot(loan_value_boxcox, bins=30, kde=True)
plt.title('Box-Cox Transformed Distribution of Loan Value')
plt.xlabel('Box-Cox Loan Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()`}</code>
            </pre>
          )}
        </div>
        
        <div className="visualization">
          <img src={`${import.meta.env.BASE_URL}box-cox-transformed.jpg`} alt="Box-Cox Transformed Distribution of Loan Value" className="analysis-image" />
        </div>
        
        <div className="analysis-notes">
          <p><strong>Box-Cox Transformation of Loan Value:</strong></p>
          <ul>
            <li>The transformation reduced the skewness present in the original loan value data.</li>
            <li>The resulting distribution is more symmetric, centered around 3.2 on the transformed scale.</li>
            <li>There is still some variation, but the extreme values have been reduced, making the spread of values more even.</li>
            <li>The Box-Cox transformation helped to stabilize variance and made the data more suitable for further analysis.</li>
          </ul>
        </div>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('installmentTransform')}
          >
            <span className="code-title">Log Transform Installment Value</span>
          </div>
          {visibleSections.installmentTransform && (
            <pre className="code-content">
              <code>{`#Log transforming Installment Value to normalize the distribution for better analysis
merged_data['LOG_INSTALLMENT_VALUE'] = np.log1p(merged_data['INSTALLMENT_VALUE'])

# Plotting the log transformed installment value distribution to check the transformation result
plt.figure(figsize=(10, 6))
sns.histplot(merged_data['LOG_INSTALLMENT_VALUE'], bins=30, kde=True)
plt.title('Log Transformed Distribution of Installment Value')
plt.xlabel('Log(Installment Value)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()`}</code>
            </pre>
          )}
        </div>
       
        <div className="visualization">
          <img src={`${import.meta.env.BASE_URL}log_int_val.jpg`} alt="Log Transformed Distribution of Installment Value" className="analysis-image" />
        </div>
        
        <div className="analysis-notes">
          <p><strong>Log Transformation of Installment Value:</strong></p>
          <ul>
            <li>The log transformation addressed the heavy right-skew present in the raw installment value data.</li>
            <li>The majority of values are now centered between 4 and 5 on the log scale.</li>
            <li>The transformation compressed the higher values, reducing the effect of outliers and producing a smoother distribution.</li>
            <li>This transformation helped in visualizing the data more clearly by bringing most values closer to a normal distribution.</li>
          </ul>
          <p>Even with transformations like log and Box-Cox, the data remains somewhat skewed and can't be fully normalized.</p>
          <p>This persistent skewness will be considered during modeling, as certain models are more robust to skewed data than others.</p>
        </div>
      </div>
      
      <div className="subsection">
        <h3>b) Analyzing Correlation</h3>
        <p><em>hint: calculate the correlation matrix and then visualize it with the <code>heatmap</code> of the library <code>seaborn</code></em></p>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('correlationMatrix')}
          >
            <span className="code-title">Correlation Matrix</span>
          </div>
          {visibleSections.correlationMatrix && (
            <pre className="code-content">
              <code>{`#Calculating the correlation matrix for the numeric columns
correlation_matrix = merged_data.corr()

print(correlation_matrix)`}</code>
            </pre>
          )}
        </div>
        
        <div className="output-box">
          <div className="output-header">
            <span className="output-title">Correlation Matrix Output</span>
          </div>
          <div className="output-content">
            <pre>{`                       CUSTOMER_ID       AGE  ANNUAL_INCOME  NUMBER_OF_MONTHS  \\
CUSTOMER_ID               1.000000  0.102233      -0.004614         -0.020530   
AGE                       0.102233  1.000000      -0.013549         -0.074464   
ANNUAL_INCOME            -0.004614 -0.013549       1.000000          0.023618   
NUMBER_OF_MONTHS         -0.020530 -0.074464       0.023618          1.000000   
GOOD_VALUE               -0.005764  0.028629       0.003103         -0.055829   
ADVANCE_VALUE             0.000232 -0.004615       0.002866         -0.020284   
LOAN_VALUE               -0.006967  0.031042       0.002442         -0.053898   
INSTALLMENT_VALUE        -0.005151  0.011715       0.006970         -0.032103   
NUMBER_INSTALLMENT       -0.003922  0.033594      -0.004772         -0.041107   
GAPR                      0.012051  0.018042      -0.001767         -0.002710   
NIR                      -0.006089 -0.019574      -0.002653         -0.013224   
LOG_ANNUAL_INCOME        -0.005763 -0.036118       0.296843          0.165117   
LOG_INSTALLMENT_VALUE    -0.005548  0.054616       0.007168         -0.058253   

                       GOOD_VALUE  ADVANCE_VALUE  LOAN_VALUE  \\
CUSTOMER_ID             -0.005764       0.000232   -0.006967   
AGE                      0.028629      -0.004615    0.031042   
ANNUAL_INCOME            0.003103       0.002866    0.002442   
NUMBER_OF_MONTHS        -0.055829      -0.020284   -0.053898   
GOOD_VALUE               1.000000       0.382018    0.972256   
ADVANCE_VALUE            0.382018       1.000000    0.158834   
LOAN_VALUE               0.972256       0.158834    1.000000   
INSTALLMENT_VALUE        0.349132       0.107375    0.342729   
NUMBER_INSTALLMENT       0.822444       0.104608    0.858213   
GAPR                     0.022054      -0.008466    0.027717   
NIR                      0.099696       0.009873    0.104201   
LOG_ANNUAL_INCOME       -0.022568       0.021723   -0.027978   
LOG_INSTALLMENT_VALUE    0.667849       0.223691    0.652447   

                       INSTALLMENT_VALUE  NUMBER_INSTALLMENT      GAPR  \\
CUSTOMER_ID                    -0.005151           -0.003922  0.012051   
AGE                             0.011715            0.033594  0.018042   
ANNUAL_INCOME                   0.006970           -0.004772 -0.001767   
NUMBER_OF_MONTHS               -0.032103           -0.041107 -0.002710   
GOOD_VALUE                      0.349132            0.822444  0.022054   
ADVANCE_VALUE                   0.107375            0.104608 -0.008466   
LOAN_VALUE                      0.342729            0.858213  0.027717   
INSTALLMENT_VALUE               1.000000            0.194244  0.018029   
NUMBER_INSTALLMENT              0.194244            1.000000  0.054867   
GAPR                            0.018029            0.054867  1.000000   
NIR                             0.306102            0.095839  0.332345   
LOG_ANNUAL_INCOME               0.012326           -0.037699 -0.023341   
LOG_INSTALLMENT_VALUE           0.626674            0.504953  0.025302   

                            NIR  LOG_ANNUAL_INCOME  LOG_INSTALLMENT_VALUE  
CUSTOMER_ID           -0.006089          -0.005763              -0.005548  
AGE                   -0.019574          -0.036118               0.054616  
ANNUAL_INCOME         -0.002653           0.296843               0.007168  
NUMBER_OF_MONTHS      -0.013224           0.165117              -0.058253  
GOOD_VALUE             0.099696          -0.022568               0.667849  
ADVANCE_VALUE          0.009873           0.021723               0.223691  
LOAN_VALUE             0.104201          -0.027978               0.652447  
INSTALLMENT_VALUE      0.306102           0.012326               0.626674  
NUMBER_INSTALLMENT     0.095839          -0.037699               0.504953  
GAPR                   0.332345          -0.023341               0.025302  
NIR                    1.000000          -0.008070               0.241094  
LOG_ANNUAL_INCOME     -0.008070           1.000000              -0.000467  
LOG_INSTALLMENT_VALUE  0.241094          -0.000467               1.000000`}</pre>
          </div>
        </div>
        
        <div className="code-container">
          <div 
            className="code-header"
            onClick={() => toggleSection('correlationHeatmap')}
          >
            <span className="code-title">Correlation Heatmap</span>
          </div>
          {visibleSections.correlationHeatmap && (
            <pre className="code-content">
              <code>{`#Setting the figure size for the correlation heatmap
plt.figure(figsize=(20, 16))

#Plotting the heatmap for the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix Heatmap')

#Displaying the plot
plt.show()`}</code>
            </pre>
          )}
        </div>
        
        <div className="visualization">
          <img src={`${import.meta.env.BASE_URL}corr-mat.png`} alt="Correlation Matrix Heatmap" className="analysis-image" />
        </div>
        
        <div className="analysis-notes">
          <h4>Correlation Matrix Insights</h4>
          <ul>
            <li><strong>LOAN_VALUE and GOOD_VALUE (0.97):</strong>
              <p>A very high correlation, which makes sense as the loan value is typically tied to the value of the asset being mortgaged (like a house or car).</p>
            </li>
            <li><strong>NUMBER_INSTALLMENT and LOAN_VALUE (0.86):</strong>
              <p>Strong positive correlation. Larger loans usually have more installments for repayment, so the connection is expected.</p>
            </li>
            <li><strong>GOOD_VALUE and NUMBER_INSTALLMENT (0.82):</strong>
              <p>Higher asset values usually lead to larger loans, which require more installments to pay off, explaining the strong correlation.</p>
            </li>
            <li><strong>GOOD_VALUE and INSTALLMENT_VALUE (0.35):</strong>
              <p>A moderate correlation. More valuable assets usually result in larger installments, although other factors also influence this.</p>
            </li>
            <li><strong>ADVANCE_VALUE and GOOD_VALUE (0.38):</strong>
              <p>Moderate positive correlation. Higher asset values tend to lead to larger advances (down payments).</p>
            </li>
            <li><strong>AGE:</strong>
              <p>Age has a very weak or no correlation with financial variables like loan value or income. This suggests that age doesn't significantly influence these factors in this dataset.</p>
            </li>
          </ul>
          
          <h4>Correlation Matrix after Feature Engineering</h4>
          <p>This matrix represents the data after feature engineering, where all variables have been converted to numerical form.</p>
          
          <div className="visualization">
            <img src={`${import.meta.env.BASE_URL}corr-mat-after.png`} alt="Correlation Matrix After Feature Engineering" className="analysis-image" />
          </div>
          
          <h4>Key Correlations with STATUS_EARLY_EXPIRED:</h4>
          <ul>
            <li><strong>GOOD_VALUE (0.58):</strong>
              <p>Higher asset values (such as properties or cars) are positively correlated with early loan payoffs. This suggests that more valuable collateral might encourage borrowers to pay off loans sooner.</p>
            </li>
            <li><strong>LOAN_VALUE (0.61):</strong>
              <p>There is a strong positive correlation between higher loan values and early payoff. Larger loans may be repaid faster, possibly due to more financially stable borrowers taking them.</p>
            </li>
            <li><strong>NUMBER_INSTALLMENT (0.69):</strong>
              <p>A strong positive correlation indicates that loans with a higher number of installments are more likely to be paid off early. This could be due to borrowers accelerating repayment to reduce long-term interest costs.</p>
            </li>
            <li><strong>REFINANCED (0.72):</strong>
              <p>Refinanced loans are highly correlated with early payoffs. This suggests that refinancing often leads to more favorable terms that allow borrowers to settle loans earlier.</p>
            </li>
            <li><strong>FROM_REFINANCE (0.47):</strong>
              <p>Loans originating from a refinance also show a positive correlation with early payoff, reinforcing the idea that refinancing plays a key role in concluding loans early.</p>
            </li>
            <li><strong>SECTOR_TYPE_FL (-0.59) (Finalized Loan):</strong>
              <p>Finalized loans are negatively correlated with early payoff, meaning they are less likely to be concluded early. This might be due to fixed terms that are more rigid compared to other loan types.</p>
            </li>
            <li><strong>SECTOR_TYPE_PL (0.63) (Personal Loan):</strong>
              <p>Personal loans have a strong positive correlation with early payoff, indicating they are more likely to be settled before the end of the loan term. This might be because personal loans often have higher interest rates, motivating quicker payoffs.</p>
            </li>
            <li><strong>Marital Status:</strong>
              <p>Different marital statuses seem to have a very minor variation in their effect on early loan payoff, indicating that marital status does not play any significant role.</p>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ExploratoryAnalysis;
