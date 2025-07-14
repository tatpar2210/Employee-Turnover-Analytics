#!/usr/bin/env python3
"""
Create PDF Report for Employee Turnover Analytics
"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


def create_pdf_report():
    doc = SimpleDocTemplate("Employee_Turnover_Analytics_Report.pdf", pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=30,
        alignment=TA_CENTER, textColor=colors.blue)
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'], fontSize=16, spaceAfter=12,
        spaceBefore=20, textColor=colors.blue)
    subheading_style = ParagraphStyle(
        'CustomSubHeading', parent=styles['Heading3'], fontSize=14, spaceAfter=8,
        spaceBefore=12, textColor=colors.green)
    normal_style = ParagraphStyle(
        'CustomNormal', parent=styles['Normal'], fontSize=11, spaceAfter=6, alignment=TA_JUSTIFY)

    # Title Page
    story.append(Paragraph("Employee Turnover Analytics", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Course-end Project 3", heading_style))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Comprehensive Analysis Report", heading_style))
    story.append(Spacer(1, 40))
    story.append(Paragraph("Student: Tatpar Mishra", normal_style))
    story.append(Paragraph("Course: Data Science/ML Course", normal_style))
    story.append(Paragraph("Date: July 2025", normal_style))
    story.append(Paragraph("Project: Simplilearn: Employee Turnover Analytics", normal_style))
    story.append(PageBreak())

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        """
        This project analyzes employee turnover data for Portobello Tech. The analysis identifies key factors contributing to turnover, clusters employees who left, builds predictive models, and provides actionable retention strategies for HR.
        """, normal_style))
    story.append(Paragraph("Key Objectives Achieved:", subheading_style))
    objectives = [
        "• Data quality checks and missing value analysis",
        "• EDA to identify main turnover factors",
        "• Clustering of employees who left",
        "• Class imbalance handling with SMOTE",
        "• K-fold cross-validation and model evaluation",
        "• Best model selection and metric justification",
        "• Targeted retention strategy recommendations"
    ]
    for obj in objectives:
        story.append(Paragraph(obj, normal_style))
    story.append(PageBreak())

    # Introduction
    story.append(Paragraph("1. Introduction", heading_style))
    story.append(Paragraph("1.1 Project Background", subheading_style))
    story.append(Paragraph(
        """
        Portobello Tech periodically evaluates employees' work details, including satisfaction, evaluation, projects, hours, tenure, promotions, and salary. The HR department uses this data to predict and reduce employee turnover, which is costly and disruptive.
        """, normal_style))
    story.append(Paragraph("1.2 Dataset Overview", subheading_style))
    dataset_info = [
        ["Records", "14,999 employees"],
        ["Features", "satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, left, promotion_last_5years, sales, salary"],
        ["Target", "left (1 = left, 0 = stayed)"],
        ["Turnover Rate", "23.81%"]
    ]
    dataset_table = Table(dataset_info, colWidths=[2*inch, 4*inch])
    dataset_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(dataset_table)
    story.append(Spacer(1, 20))
    story.append(PageBreak())

    # Data Quality Checks
    story.append(Paragraph("2. Data Quality Checks", heading_style))
    story.append(Paragraph("• No missing values found in the dataset", normal_style))
    story.append(Paragraph("• Data types are appropriate for analysis", normal_style))
    story.append(Paragraph("• 3,008 duplicate rows detected (retained for analysis)", normal_style))
    story.append(Paragraph("• Target variable distribution: 11,428 stayed, 3,571 left", normal_style))
    story.append(PageBreak())

    # Exploratory Data Analysis
    story.append(Paragraph("3. Exploratory Data Analysis (EDA)", heading_style))
    story.append(Paragraph("Key factors contributing to turnover:", subheading_style))
    story.append(Paragraph("- Employees who left had much lower satisfaction (avg: 0.44) than those who stayed (avg: 0.67)", normal_style))
    story.append(Paragraph("- Overworked employees (>250 hours/month) and those with more projects are more likely to leave", normal_style))
    story.append(Paragraph("- Departmental turnover rates vary: HR (29%), Accounting (27%), Technical (26%), Management (14%)", normal_style))
    story.append(Paragraph("- Low salary employees have higher turnover", normal_style))
    story.append(Paragraph("- Correlation analysis and boxplots confirm these patterns", normal_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("See 'eda_analysis.png' and 'correlation_matrix.png' for visualizations.", normal_style))
    story.append(PageBreak())

    # Clustering Analysis
    story.append(Paragraph("4. Clustering of Employees Who Left", heading_style))
    story.append(Paragraph("K-means clustering (k=3) on employees who left reveals:", subheading_style))
    cluster_info = [
        ["Cluster", "Satisfaction", "Evaluation", "Projects", "Hours", "Tenure (yrs)", "Count"],
        ["0", "0.41", "0.52", "2.1", "149", "3.0", "1,649"],
        ["1", "0.12", "0.86", "6.2", "273", "4.1", "963"],
        ["2", "0.81", "0.91", "4.5", "243", "5.2", "959"]
    ]
    cluster_table = Table(cluster_info, colWidths=[0.8*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    cluster_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue)
    ]))
    story.append(cluster_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph("See 'employee_clusters.png' for visualization.", normal_style))
    story.append(PageBreak())

    # Class Imbalance Handling
    story.append(Paragraph("5. Class Imbalance Handling (SMOTE)", heading_style))
    story.append(Paragraph("• Original training set: 9,142 stayed, 2,857 left", normal_style))
    story.append(Paragraph("• After SMOTE: 9,142 stayed, 9,142 left (fully balanced)", normal_style))
    story.append(PageBreak())

    # Model Training and Evaluation
    story.append(Paragraph("6. Model Training, Cross-Validation, and Evaluation", heading_style))
    story.append(Paragraph("Models trained: Logistic Regression, Random Forest, Gradient Boosting, SVM", normal_style))
    story.append(Paragraph("5-fold cross-validation results:", subheading_style))
    model_info = [
        ["Model", "CV Accuracy", "Test Accuracy", "Precision", "Recall", "F1-Score"],
        ["Logistic Regression", "0.770", "0.759", "0.496", "0.762", "0.601"],
        ["Random Forest", "0.983", "0.987", "0.977", "0.969", "0.973"],
        ["Gradient Boosting", "0.958", "0.967", "0.921", "0.941", "0.931"],
        ["SVM", "0.696", "0.687", "0.410", "0.712", "0.520"]
    ]
    model_table = Table(model_info, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen)
    ]))
    story.append(model_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Best model: Random Forest (F1-Score: 0.973)", normal_style))
    story.append(Paragraph("Justification: F1-Score is used due to class imbalance and the need to balance precision and recall.", normal_style))
    story.append(Paragraph("Top 5 important features: satisfaction_level, time_spend_company, average_montly_hours, number_project, last_evaluation", normal_style))
    story.append(Paragraph("See 'model_comparison.png', 'confusion_matrix.png', and 'feature_importance.png' for details.", normal_style))
    story.append(PageBreak())

    # Retention Strategies
    story.append(Paragraph("7. Retention Strategies for Targeted Employees", heading_style))
    story.append(Paragraph("Targeted strategies for high-risk groups:", subheading_style))
    strategies = [
        ("Low Satisfaction (<0.5)", "2,036", [
            "Conduct regular satisfaction surveys",
            "Implement flexible work arrangements",
            "Provide career development opportunities",
            "Improve work-life balance policies",
            "Enhance recognition and reward programs"
        ]),
        ("Overworked (>250 hours/month)", "1,960", [
            "Implement workload management systems",
            "Hire additional staff",
            "Provide overtime compensation",
            "Set realistic project deadlines",
            "Encourage time-off utilization"
        ]),
        ("Long-tenured (>5 years)", "1,073", [
            "Provide career advancement opportunities",
            "Implement mentorship programs",
            "Offer specialized training",
            "Create leadership development programs",
            "Provide competitive compensation"
        ]),
        ("High Performers (evaluation >0.8)", "3,798", [
            "Implement performance-based bonuses",
            "Provide challenging projects",
            "Offer leadership opportunities",
            "Create fast-track promotion programs",
            "Provide competitive compensation"
        ]),
        ("Low Salary", "5,144", [
            "Conduct salary benchmarking",
            "Implement performance-based raises",
            "Provide additional benefits",
            "Create profit-sharing programs",
            "Offer equity or stock options"
        ])
    ]
    for group, count, recs in strategies:
        story.append(Paragraph(f"<b>{group}</b> (Employees: {count})", normal_style))
        for rec in recs:
            story.append(Paragraph(f"- {rec}", normal_style))
        story.append(Spacer(1, 6))
    story.append(Paragraph("Department-specific strategies are also recommended for high-turnover departments (HR, Accounting, Technical).", normal_style))
    story.append(PageBreak())

    # Conclusion
    story.append(Paragraph("8. Conclusion", heading_style))
    story.append(Paragraph(
        """
        This analysis provides actionable insights for reducing employee turnover and improving retention at Portobello Tech. By focusing on satisfaction, workload, career development, and compensation, HR can target high-risk groups and departments for maximum impact.
        """, normal_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("--- End of Report ---", normal_style))

    doc.build(story)
    print("PDF report generated: Employee_Turnover_Analytics_Report.pdf")

if __name__ == "__main__":
    create_pdf_report() 