# HackerEarth_Hiring_Challenge

Problem: Predict Project's Success

Data Preprocessing and Feature Engineering:

1.All unix time based features are converted to Standard Datetime Object. And days difference is taken from project_status_changed_at and deadline features.

2.All Currencies are converted into same scale(US Dollar).

3.For Country column if the country is US then it is marked as 1 otherwise marked as 0.

4.disable_communication is normally label encoded.

Modelling 

Random Forest model is used.
Model Hypermeter: n_estimators=500,max_features='auto',min_samples_leaf=50


