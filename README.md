# Bladder Cancer Diagnostic Framework 
> using Autophagy-Related Genes
____
### Project Overview
This project proposes a machine learning-based framework for bladder cancer diagnosis using gene expression data of autophagy-related genes (ARGs). The framework integrates data preprocessing, feature selection, and explainable AI to build predictive models and offers a user-friendly interface for real-time predictions.





### Background
Bladder cancer is a common malignancy, and traditional diagnostic methods often have limitations. Autophagy-related genes (ARGs) are involved in bladder tumorigenesis, and their expression patterns can serve as potential diagnostic biomarkers. This study leverages machine learning to analyze these gene expression profiles to create an accurate, non-invasive diagnostic tool.





### Methodology
The project followed a clear methodology:


- Data Preprocessing: Gene expression datasets from 408 tumor and 19 normal bladder samples were obtained. These were merged and labeled, with cancer samples as '1' and normal samples as '0'. Missing values were handled using K-Nearest Neighbors (KNN) imputation, and the dataset was balanced using SMOTE.






- Feature Engineering: A correlation-based filtering approach was applied to reduce multicollinearity, removing 20 features with a correlation greater than 0.8.




- Model Training: Four classification algorithms—Random Forest, Support Vector Machine (SVM), Logistic Regression, and XGBoost—were trained and evaluated.



- Cancer Predictor Tool: SHAP (SHapley Additive exPlanations) values were used to interpret the model's decisions and identify the most influential genes.






- Deployment: The best-performing model was deployed via a Streamlit-based web interface for practical usability.


### Results
- Model Performance
The models were evaluated on key metrics, including accuracy, precision, recall, and ROC-AUC.


Random Forest: Achieved a perfect recall of 1.000, an F1-score of 0.993, and an ROC-AUC of 0.981.






XGBoost: Showcased a high ROC-AUC of 0.978 and an accuracy of 0.976.




Both Random Forest and XGBoost demonstrated excellent performance, with ROC-AUC values above 0.97, confirming their suitability for this classification task.



- Key Biomarkers
Feature importance analysis using XGBoost identified the top 15 genes that were most influential in the model's predictions. The top five were:

PARK2

FOS

ATP6V0B

SYNPO2

EIF4EBP1

The prominence of these genes, known for their roles in autophagy and tumor pathways, reinforces their biological relevance as potential diagnostic biomarkers. SHAP analysis confirmed that the expression levels of these genes, particularly high levels of 


EIF4EBP1, TP53INP2, SYNPO2, and PARK2, have a strong influence on predicting a cancer diagnosis.
