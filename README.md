Code Description
1. Research Background and Objective
The objective of this study is to utilize the Random Forest (RF) method combined with SHAP (SHapley Additive exPlanations) to identify the key school-level and student-level variables in the PISA 2022 dataset under the TELE environment. SHAP is a tool for explaining model predictions, quantifying the contribution of each feature to the model’s output, and thus identifying the most influential factors affecting student learning motivation and achievement in educational data.

2. Data and Variables
This study uses the PISA 2022 dataset, specifically focusing on the TELE environment variables, which include both school-level and student-level variables. School-level variables such as school size, teacher quality, and school resources are considered, while student-level variables include learning motivation, social interaction, and family background. The aim is to select the most influential variables for predicting student learning outcomes using the RF-based SHAP method.

3. Python Environment and Libraries
The following Python libraries were used for data processing, modeling, and analysis in this study:
pandas: for data manipulation and processing.
numpy: for numerical calculations.
scikit-learn: for building the random forest model.
shap: for calculating and visualizing SHAP values.
matplotlib and seaborn: for plotting.
