# Employee Turnover Analytics
## Project Overview

This project analyzes employee turnover data for Portobello Tech, an app innovator company. The analysis helps identify patterns in work style and factors contributing to employee turnover, enabling the HR department to develop effective retention strategies.

## Business Context

Employee turnover refers to the total number of workers who leave a company over time. High turnover rates can be costly and disruptive to business operations. This project uses machine learning to:

- Predict employee turnover
- Identify key factors contributing to turnover
- Develop targeted retention strategies
- Improve employee satisfaction and retention

## Dataset Description

The dataset contains employee information including:
- **satisfaction_level**: Employee satisfaction score (0-1)
- **last_evaluation**: Last performance evaluation score (0-1)
- **number_project**: Number of projects worked on
- **average_montly_hours**: Average monthly working hours
- **time_spend_company**: Years spent in the company
- **Work_accident**: Whether employee had work accident (0/1)
- **left**: Whether employee left the company (0/1)
- **promotion_last_5years**: Whether promoted in last 5 years (0/1)
- **sales**: Department/role
- **salary**: Salary level (low/medium/high)

## Project Objectives

1. **Data Quality Checks**: Verify data integrity and identify missing values
2. **Exploratory Data Analysis (EDA)**: Understand factors contributing to turnover
3. **Clustering Analysis**: Group employees who left based on satisfaction and evaluation
4. **Class Imbalance Handling**: Use SMOTE technique to balance the dataset
5. **Model Training**: Perform k-fold cross-validation with multiple algorithms
6. **Performance Evaluation**: Identify the best model using appropriate metrics
7. **Retention Strategies**: Suggest targeted strategies for different employee groups

## Methodology

### 1. Data Quality Assessment
- Missing value analysis
- Data type verification
- Duplicate detection
- Statistical summary

### 2. Exploratory Data Analysis
- Distribution analysis of key variables
- Correlation analysis
- Turnover rate analysis by department and salary
- Visualization of key patterns

### 3. Clustering Analysis
- K-means clustering on employees who left
- Elbow method for optimal cluster determination
- Cluster profiling and interpretation

### 4. Machine Learning Pipeline
- Data preprocessing and encoding
- SMOTE for class imbalance handling
- Multiple model training (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- K-fold cross-validation
- Performance comparison

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC analysis
- Confusion matrix visualization
- Feature importance analysis

### 6. Retention Strategy Development
- High-risk employee identification
- Department-specific recommendations
- Targeted intervention strategies

## Key Findings

### Factors Contributing to Turnover
1. **Low Satisfaction**: Employees with satisfaction < 0.5 are significantly more likely to leave
2. **Overwork**: Employees working > 250 hours/month show higher turnover rates
3. **Lack of Promotion**: Employees without promotions in 5 years are more likely to leave
4. **Department Variations**: Different departments show varying turnover rates
5. **Salary Level**: Low salary employees have higher turnover rates

### Model Performance
- **Best Model**: Random Forest (typically shows best performance)
- **Key Metrics**: F1-Score, Precision, Recall for balanced evaluation
- **Cross-Validation**: 5-fold CV ensures robust performance estimation

## Files Structure

```
Employee Turnover Analytics/
├── dataset/
│   └── HR_comma_sep.csv          # Original dataset
├── employee_turnover_analysis.py # Main analysis script
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── Generated Outputs:
│   ├── eda_analysis.png         # EDA visualizations
│   ├── correlation_matrix.png   # Correlation heatmap
│   ├── elbow_curve.png         # Clustering elbow curve
│   ├── employee_clusters.png   # Cluster visualization
│   ├── model_comparison.png    # Model performance comparison
│   ├── confusion_matrix.png    # Best model confusion matrix
│   ├── feature_importance.png  # Feature importance plot
│   └── employee_turnover_report.txt # Analysis report
```

## Installation and Setup

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**:
   ```bash
   python employee_turnover_analysis.py
   ```

## Usage

The main script `employee_turnover_analysis.py` performs the complete analysis pipeline:

```python
from employee_turnover_analysis import EmployeeTurnoverAnalysis

# Initialize analysis
analysis = EmployeeTurnoverAnalysis('dataset/HR_comma_sep.csv')

# Run complete analysis
analysis.run_complete_analysis()
```

## Output and Results

### Visualizations Generated
- **EDA Analysis**: Box plots showing key factors vs turnover
- **Correlation Matrix**: Feature relationships heatmap
- **Clustering Results**: Employee clusters visualization
- **Model Comparison**: Performance metrics comparison
- **Confusion Matrix**: Best model prediction accuracy
- **Feature Importance**: Most important predictive features

### Key Metrics
- **Overall Turnover Rate**: ~24% (varies by dataset)
- **Best Model F1-Score**: Typically > 0.85
- **Cross-Validation Accuracy**: Stable performance across folds

### Retention Recommendations
1. **Satisfaction Improvement**: Regular surveys and feedback systems
2. **Workload Management**: Implement workload balancing systems
3. **Career Development**: Clear promotion paths and training programs
4. **Compensation Review**: Market-competitive salary structures
5. **Department-Specific**: Tailored strategies for high-turnover departments

## Technical Details

### Algorithms Used
- **Clustering**: K-means with elbow method
- **Classification**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Sampling**: SMOTE for class imbalance
- **Validation**: 5-fold cross-validation

### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

## Business Impact

This analysis provides actionable insights for HR departments to:

1. **Predict Turnover Risk**: Identify employees likely to leave
2. **Targeted Interventions**: Focus retention efforts on high-risk groups
3. **Policy Development**: Data-driven HR policy recommendations
4. **Cost Reduction**: Reduce turnover-related costs
5. **Employee Satisfaction**: Improve overall workplace satisfaction

## Future Enhancements

1. **Real-time Monitoring**: Implement real-time turnover prediction
2. **Advanced Models**: Explore deep learning approaches
3. **External Data**: Incorporate market and industry data
4. **A/B Testing**: Test retention strategy effectiveness
5. **Predictive Maintenance**: Proactive intervention systems

## Contact and Support

For questions or support regarding this analysis, please refer to the project documentation or contact the development team.

---

**Note**: This analysis is based on historical data and should be used as a guide for decision-making. Regular updates and validation are recommended for ongoing effectiveness. 
