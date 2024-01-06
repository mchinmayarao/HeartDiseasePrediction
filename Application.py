import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pickle

class HeartFailurePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Failure Prediction App")

        # Create and set up the GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Labels
        ttk.Label(self.root, text="Enter Patient Information:").grid(row=0, column=0, columnspan=2, pady=10)

        # Labels and Entry/Drop-down widgets for each feature
        self.labels = ["Age (years):", "Anaemia:", "CPK (Creatinine Phosphokinase):",
                       "Diabetes:", "Ejection Fraction (%):", "High Blood Pressure:",
                       "Platelets:", "Serum Creatinine:", "Serum Sodium:", "CP (Chest Pain):",
                       "Sex:", "Smoking:", "Max HR (Max Heart Rate):", "Cholesterol:"]

        self.entries = []
        for i, label in enumerate(self.labels):
            ttk.Label(self.root, text=label).grid(row=i + 1, column=0, sticky="e", padx=10, pady=5)

            if label == "Sex:":
                # For the "Sex" feature, use a drop-down menu with "Male" and "Female" options
                value_var = tk.StringVar()
                value_var.set("Male")  # Default value
                dropdown = ttk.Combobox(self.root, textvariable=value_var, values=["Male", "Female"])
                dropdown.grid(row=i + 1, column=1, padx=10, pady=5)
                self.entries.append(dropdown)
            elif label in ["Anaemia:", "Diabetes:", "High Blood Pressure:", "Smoking:"]:
                # For Yes/No features, use a drop-down menu
                value_var = tk.StringVar()
                value_var.set("No")  # Default value
                dropdown = ttk.Combobox(self.root, textvariable=value_var, values=["No", "Yes"])
                dropdown.grid(row=i + 1, column=1, padx=10, pady=5)
                self.entries.append(dropdown)
            else:
                # For other features, use an Entry widget
                entry = ttk.Entry(self.root)
                entry.grid(row=i + 1, column=1, padx=10, pady=5)
                self.entries.append(entry)

        # Predict Button
        ttk.Button(self.root, text="Predict", command=self.predict).grid(row=len(self.labels) + 1, column=0, columnspan=2, pady=10)

        # Label to display prediction result
        self.result_label = ttk.Label(self.root, text="")
        self.result_label.grid(row=len(self.labels) + 2, column=0, columnspan=2, pady=10)

    def predict(self):
        # Get user input from Entry/Drop-down widgets
        user_input = []

        for i, entry in enumerate(self.entries):
            if isinstance(entry, ttk.Combobox):
                # For drop-down menu, convert 'Yes' to 1 and 'No' to 0
                if entry["values"] == ["No", "Yes"]:
                    value = 1 if entry.get() == "Yes" else 0
                else:
                    # For "Male" and "Female"
                    value = 1 if entry.get() == "Male" else 0
            else:
                # For other features, get the input value
                value = entry.get()

            user_input.append(value)

        # Validate and convert inputs
        try:
            user_input = [float(user_input[i]) if i not in [1, 3, 5, 9] else int(user_input[i]) for i in range(len(user_input))]
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values.")
            return

        # Load the saved model
        model = pickle.load(open('model.pkl', 'rb'))

        # Make prediction
        prediction = model.predict([user_input])[0]

        # Display the prediction result in the label
        if prediction == 1:
            result = "The patient is at risk of death."
        else:
            result = "The patient is not at risk of death."

        self.result_label.config(text=result)

if __name__ == "__main__":
    root = tk.Tk()
    app = HeartFailurePredictionApp(root)
    root.mainloop()
