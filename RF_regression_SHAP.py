import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import shap

# Import your data as a CSV file containing your samples.
# The first column should be the target variable. The remaining columns should be the predictor variables and their values.
data = pd.read_csv('./SCHonly1.csv')

# Split data into input features and target variable which in this case is 'ALT'.
X = data.drop(columns=['SCHSIZE', 'PROPAT7','STRATIO', 'PROPSUPP',
                       'PROADMIN', 'SCHAUTO', 'RATCMP1', 'TEAFDBK',
                       'STUBEHA', 'STDTEST', 'SCHESCS',	'TOTAT',
                       'PROATCE', 'PROPAT8', 'TOTSTAFF','PROMGMT',
                       'PROOSTAF', 'SCHSEL', 'TCHPART', 'SRESPCUR',
                       'SRESPRES', 'EDULEAD', 'INSTLEAD', 'ENCOURPG',
                       'RATCMP2', 'RATTAB', 'STAFFSHORT', 'EDUSHORT',
                       'TEACHBEHA', 'TDTEST'])  # Features (input covariates)
y = data['PV1READ']  # Target variable

# Here is a comparision list between ee.Classifier.smileRandomForest hyperparameter names and  sklearn Random Forest for reference:
# n_estimators = numberOfTrees
# max_depth = maxNodes
# min_samples_split = variablesPerSplit
# min_samples_leaf = minLeafPopulation

# Set the values for each hyperparameter you would like to optimize (tune) for your Random Forest model.
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create your Random Forest model.
rf_model = RandomForestRegressor(random_state=42)

# Use GridSearchCV for hyperparameter tuning and 5-fold cross-validation. Choose best parameters based on R².
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = make_scorer(r2_score)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring=scoring, cv=kf, n_jobs=-1)
grid_search.fit(X, y)

# Extract R² values for each hyperparameter combination.
results = pd.DataFrame(grid_search.cv_results_)

# Create R² values.
r2_values = results[['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf', 'mean_test_score']]

# Export R² values to a CSV file.
r2_values.to_csv('./RF_Tuning_Results.csv', index=False)

# Build optimized Random Forest model based on best R².
best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_r2 = grid_search.best_score_

# Print the optimized hyperparametes and R² achieved on cross-validation.
print("Best Hyperparameters:", best_params)
print("Best R-squared (R²):", best_r2)

# Use the Shapley Explainer function to analyze your optimzed Random Forest model.
explainer = (best_rf_model, X)
# best_rf_model.params[best_r2] = "regression"
# explainer = shap.TreeExplainer(best_rf_model)
# explainer = shap.TreeExplainer(best_rf_model)
# best_rf_model.params[grid_search] = "regression"

# Use this to randomly sample 2000 samples and create shap values for them.
num_samples = 1000
random_sample_indices = np.random.choice(X.index, num_samples, replace=False)
random_samples = X.loc[random_sample_indices]

# See (view) the Shap summary plot:
shap_values = explainer(random_samples)

# Export the summary plot. In this case, only show the top 30 most important variables.
plt.figure()
shap.summary_plot(shap_values, random_samples, max_display=13, show=False)
plt.savefig('./Shap_summary_plot.jpg', dpi=600)
plt.close()

# Create and export a heatmap plot.
plt.figure()
shap.plots.heatmap(shap_values,  max_display=6, show=False)
plt.savefig('./Shap_heat_plot.jpg', dpi=600, bbox_inches='tight')
plt.close()

# Creat and export a bar plot. In this case, only show the top 30 most important variables.
plt.figure()
shap.plots.bar(shap_values,  max_display=30, show=False)
plt.savefig('./Shap_bar_plot.jpg', dpi=600, bbox_inches='tight')
plt.close()