from tkinter import ttk
import pandas
import joblib

# loading the trained model as joblib file
pipeline = joblib.load('trained_model.joblib')

# use the user-input to predict
def predict_price(entry_fields, result_label, accuracy_label):
    # Get user input  from entry fields
    user_input_df = get_user_input(entry_fields)

    # use the pre-trained model to predict the car price
    predicted_price = pipeline.predict(user_input_df)

    # Update the result label with the predicted price to display it
    result_label.config(text=f"\nEstimated Price: {predicted_price[0]:.2f}€")

    # Result_accuracy.txt is a file auto-saved in Model.py which has the evaluation metrics to display here
    with open('Result_accuracy.txt', 'r') as file:
        result_accuracy = file.read()
        accuracy_label.config(text=f"\nModel Accuracy:\n{result_accuracy}")


# Function to get user input from entry fields
def get_user_input(entry_fields):
    # fetch input from the entry fields and create a DataFrame
    user_input_dict = {feature_name: entry.get() for feature_name, entry in entry_fields.items()}
    return pandas.DataFrame([user_input_dict])

# setting for the reset button
def reset_fields(entry_fields, result_label, accuracy_label):
    # Reset entry fields
    for entry in entry_fields.values():
        entry.set('Select an option')\
            if isinstance(entry, ttk.Combobox) else entry.delete(0, 'end')
    # reset displayed texts
    result_label.config(text="")
    accuracy_label.config(text="")


# GUI setup
def gui_setup(root):
    # preset for the categorical entry fields
    entry_fields = {}
    options = {
        'Year': list(range(2024, 1949, -1)),
        'transmission': ['Select an option', 'automatic', 'manual', 'hybrid'],
        'fuel_type': ['Select an option', 'gasoline', 'diesel', 'electric', 'other'],
        'damaged': ['Select an option', 'yes', 'no']
    }

    # create the entry fields for inputs
    for i, feature_name in enumerate(
            ['Brand', 'model', 'Body', 'Year', 'transmission', 'fuel_type', 'Km', 'Horsepower', 'damaged']):
        # iterate through the above list to create and set labels for entry fields
        ttk.Label(root, text=feature_name).grid(row=i, column=0, padx=6, pady=6)
        # create the entry fields
        entry_fields[feature_name] = ttk.Combobox(root, values=options.get(feature_name, []), state='readonly')\
            if feature_name in options else ttk.Entry(root)

        # for the entries with a dropdown-menu, it displays "select an option" on top
        entry_fields[feature_name].set('Select an option') if isinstance(entry_fields[feature_name],
                                                                         ttk.Combobox) else None
        # place the entries in the GUI
        entry_fields[feature_name].grid(row=i, column=1, padx=5, pady=5, sticky='w')

    # Buttons
    predict_button = ttk.Button(root, text="Predict", command=lambda: predict_price(entry_fields, result_label, accuracy_label))
    predict_button.grid(row=len(entry_fields), column=0, columnspan=3, pady=10)

    reset_button = ttk.Button(root, text="Reset", command=lambda: reset_fields(entry_fields, result_label, accuracy_label))
    reset_button.grid(row=len(entry_fields) + 1, column=0, columnspan=3, pady=10)

    # Labels
    result_label = ttk.Label(root, text="")
    result_label.grid(row=len(entry_fields) + 2, column=0, columnspan=3)

    accuracy_label = ttk.Label(root, text="")
    accuracy_label.grid(row=len(entry_fields) + 3, column=0, columnspan=3)
