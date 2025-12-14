import pandas
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

# load the dataset and seperate the 'price' column
data = pandas.read_csv('DATASET.csv')
X = data.drop('price', axis=1)
y = data['price']

# check for categorical columns
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

# OneHotEncoder and StandardScaler for both categorical and numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns)
    ], remainder='passthrough')

'''
the Sub-models (Neural Network and RF), while using adam solver in the NN for iterations
the Meta-model (Multiple Linear Regression)
'''
submodels = [
    ('mlp', MLPRegressor(hidden_layer_sizes=(16, 8, 4), max_iter=1000, solver='adam', random_state=50)),
    ('rf', RandomForestRegressor(n_estimators=600, random_state=50))
            ]

meta_model = LinearRegression()

# create a pipleline for the stacked approach
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', StackingRegressor(estimators=submodels, final_estimator=meta_model))])

# splitting dataset into 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# train the pipeline and make predictions
pipeline.fit(X_train, y_train)
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# find the target range for the evaluation metrics (on training data)
Range = y_train.max() - y_train.min()

# Model Evaluation
mse_train, mae_train, r2_train = mean_squared_error(y_train, y_train_pred), mean_absolute_error(y_train, y_train_pred), r2_score(y_train, y_train_pred)
mse_test, mae_test, r2_test = mean_squared_error(y_test, y_test_pred), mean_absolute_error(y_test, y_test_pred), r2_score(y_test, y_test_pred)

# Normalize metrics using the target range
n_mse_train = mse_train / Range
n_mae_train = mae_train / Range
n_r2_train = r2_train
n_mse_test = mse_test / Range
n_mae_test = mae_test / Range
n_r2_test = r2_test

# display normalized results (.3f)
train_results = f"Training Data: MSE = {n_mse_train:.3f}, MAE = {n_mae_train:.3f}, R2 = {n_r2_train:.3f}"
test_results = f"Testing Data: MSE = {n_mse_test:.3f}, MAE = {n_mae_test:.3f}, R2 = {n_r2_test:.3f}"
print(train_results)
print(test_results)

# save the trained model as a joblib file
joblib.dump(pipeline, 'trained_model.joblib')

# save the evaluation results as a text file to use in UI
with open('Result_accuracy.txt', 'w') as result_file:
    result_file.write(test_results)
