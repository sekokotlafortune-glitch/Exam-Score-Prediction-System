Project Overview

Goal: predict Exam_Score (a number) using 10 features such as attendance, hours studied and parental involvement.

Because the target is a number, this is a regression problem.

Approach:

Main model: Linear Regression (simple and easy to interpret)
Comparison model: Random Forest (flexible, handles complex patterns)
Evaluation: validation set, 5-fold cross-validation and a final held-out test set
Dataset
Original data: 6,607 students, 20 columns (Student Performance Factors dataset)
After cleaning and feature selection: 6,377 students, 11 columns (10 features + 1 target)
Feature	Description	Encoding
Attendance	Percentage of classes attended	Number
Hours_Studied	Hours studied per week	Number
Previous_Scores	Score from an earlier assessment	Number
Access_to_Resources	Access to learning resources	0 = Low, 1 = Medium, 2 = High
Parental_Involvement	Level of parental involvement	0 = Low, 1 = Medium, 2 = High
Tutoring_Sessions	Tutoring sessions per month	Number (0 to 8)
Parental_Education_Level	Highest parental education	0 = High School, 1 = College, 2 = Postgraduate
Peer_Influence	Influence of peers	0 = Negative, 1 = Neutral, 2 = Positive
Family_Income	Family income level	0 = Low, 1 = Medium, 2 = High
Motivation_Level	Student motivation	0 = Low, 1 = Medium, 2 = High
Exam_Score	Target: final exam score	Number (55 to 100)
Data Preparation

The following steps were done before modelling:

Removed rows with missing values (in Teacher_Quality, Parental_Education_Level and Distance_from_Home).
Checked for duplicates (none found).
Removed impossible scores: rows where Exam_Score was above 100.
Encoded categorical variables: ordinal columns mapped to 0, 1, 2 (keeping their order) and binary columns mapped to 0/1.
Selected the 10 features with the strongest absolute correlation with Exam_Score.
Methodology
Data split
Set	Share	Students	Purpose
Training	60%	3,825	Model learns from this
Validation	20%	1,276	Check and compare models while building
Test	20%	1,276	One final, unbiased evaluation

random_state=42 is used everywhere so results are reproducible.

Models
Linear Regression: a pipeline of StandardScaler followed by LinearRegression. Scaling does not change predictions, but it makes the coefficients comparable across features.
Random Forest: RandomForestRegressor with n_estimators=300, max_depth=8 and min_samples_leaf=5. The depth and leaf limits reduce overfitting.
Validation
Validation set: used to compare the models on unseen data.
5-fold cross-validation: the 5,101 non-test students are shuffled and split into 5 folds. The model is trained on 4 folds and scored on the 5th, repeated 5 times so every fold is held out once. The average R² is reported. This avoids relying on one lucky or unlucky split.
Metrics
Metric	Meaning	Better is
MAE	Average size of the prediction error, in exam points	Lower
RMSE	Typical error, with extra penalty for large misses	Lower
R²	Share of the variation in scores the model explains	Higher
Results
Model	Split	MAE	RMSE	R²
Linear Regression	Train	0.818	2.311	0.664
Linear Regression	Validation	0.707	1.539	0.808
Linear Regression	5-Fold CV	-	-	0.702
Random Forest	Train	1.023	2.095	0.724
Random Forest	Validation	1.140	1.845	0.725
Random Forest	5-Fold CV	-	-	0.626
Final test set
Model	MAE	RMSE	R²
Linear Regression	0.788	2.148	0.705
Random Forest	1.217	2.406	0.630
Key Findings
Linear Regression outperformed Random Forest on every test metric. It is off by about 0.8 points on average and explains about 70% of the variation in exam scores.
No overfitting. Linear Regression's training, cross-validation and test R² are close (0.66, 0.70, 0.705).
The validation R² (0.81) is optimistic. That one split happened to be easier than average. The cross-validation and test scores (about 0.70) are the reliable numbers.
Attendance and Hours_Studied are the strongest predictors in both models.
Feature	Linear Regression coefficient (scaled)	Random Forest importance
Attendance	2.275	0.496
Hours_Studied	1.743	0.286
Parental_Involvement	0.730	0.037
Access_to_Resources	0.705	0.033
Previous_Scores	0.690	0.074
Tutoring_Sessions	0.600	0.027
Parental_Education_Level	0.396	0.013
Motivation_Level	0.396	0.010
Family_Income	0.395	0.015
Peer_Influence	0.357	0.009
RMSE is much larger than MAE because a small number of students (about 49 out of 6,377, under 1%) are predicted badly, by more than 10 points. The typical error is only about 0.5 points.
Confusion Matrix

A confusion matrix needs categories, but Exam_Score is a number. To produce one, scores are grouped into Low / Medium / High using cut-offs from the training data (Low below 66, Medium 66 to 68, High 69 and above).

Model	Accuracy (test set)
Linear Regression	85.9%
Random Forest	79.0%

Almost all mistakes fall into the neighbouring group, and very few students are placed two groups away from their real group. This is an alternative view of the same predictions, and the cut-offs are a choice. The regression metrics above remain the primary results.

Project Structure
.
├── model_training.py                          # Training, validation, evaluation (and graphs / confusion matrix)
├── StudentPerformanceFactors_Cleaned__1_.csv  # Cleaned dataset (10 features + target)
├── README.md
└── graph*.png                                 # Saved charts (created when the script runs)

Adjust the file names above to match your repository.

How to Run

1. Clone the repository

bash
git clone <your-repo-url>
cd <your-repo-folder>

2. Install the requirements (Python 3.12 was used)

bash
pip install pandas numpy scikit-learn matplotlib

3. Run the script (the CSV must be in the same folder)

bash
python model_training.py

The script prints the results table, the final test scores, the feature coefficients and importances. It also opens charts one at a time (close each window to continue) and saves them as PNG files in the project folder.

Limitations
About 49 students (under 1%) are predicted poorly, and these cases were not investigated.
Rows with missing values or scores above 100 were removed during cleaning (6,607 down to 6,377).
Ordinal encoding (Low, Medium, High as 0, 1, 2) assumes equal spacing between levels.
Feature selection by correlation only detects straight-line relationships and ranks features one at a time.
The model finds associations, not causes. It does not prove that changing one factor will change a student's score.
Results apply to this dataset and may differ for other schools or years.
Possible Improvements
Analyse the badly predicted students with a residual analysis.
Try all features instead of only the top 10.
Tune Random Forest settings (and try Gradient Boosting) using the validation set.
Try regularised linear models such as Ridge or Lasso.
Compare different feature selection methods.
