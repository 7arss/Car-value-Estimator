# Car-value-Estimator
I created this model a while back mainly because I wanted to figure out a new way to predict the prices of vehicles using machine learning, and secondly to test how stacked regression performs. instead of just directly training with an algortihm (ex: random forest) you train the model on 2 levels (sub model and meta model). it's basically a pipeline where you train on the dataset, get results, and train again using the results you get. as far as I know only few projects use this method and I wanted to try it for cars.  
after testing a lot of algorithms. I decided to use NN and RF for the submodel and Linear Regression for the meta model. 

```
submodels = [
    ('mlp', MLPRegressor(hidden_layer_sizes=(16, 8, 4), max_iter=1000, solver='adam', random_state=50)),
    ('rf', RandomForestRegressor(n_estimators=600, random_state=50))
            ]

meta_model = LinearRegression()

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', StackingRegressor(estimators=submodels, final_estimator=meta_model))])
```

when training and testing I mostly relied on this dataset which includes around 50k entry points (30k~ after data processing):  
https://www.kaggle.com/datasets/piumiu/used-cars-database-50000-data-points/data   
  
after some fine tuning and testing I found the model performs better on this dataset and other sample datasets much better than the existing standard approaches. it has better accuracy and acceptable overfitting.  
the only downside is that it has longer training time. for this project it wasn't much of an issue but if you have to constantly switch data then it may be cumbersome to re-train.  
   
   
# Files
the code is in 3 files. I added a small csv file called "DATASET.CSV" to just see if everything works. feel free to use whatever or change anything depending on your needs. I didn't have a specific implementation for it. but I created a functional UI in Python. the code saves the trained model as a joblib file and also automatically seperates the dataset into 80/20 and uses the 20% for testing.  
  
the files are:  
- Model.py : includes the model, it saves the trained model as joblib file and evaluation metrics as a text file (also displayed in the UI).  
- UI.py : Includes the UI elements which are input boxes and dropdown menus with a reset and estimate buttons.  
- Main.py : used to run the UI window and Use the joblib file (trained model).  
     
   
# How to use
  
the fastest way is to simply:
- Download the python packages.
- Run the model.py
- Run Main.py which will display the UI where you can enter inputs.  
  
You can use the dataset I linked or any data you have. just make sure the names match and the price column is named "price". you can also omit the UI and all the extra functionalities by just clipping the model and using it on its own. it is a good approach if you need increased accuracy and minimal overfitting especially if the data is as complex as this one.

  

