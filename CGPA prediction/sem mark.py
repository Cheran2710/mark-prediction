import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import gradio as gr
import os
import matplotlib.pyplot as plt

# Ensure required packages are installed (optional if already installed)
os.system('pip install --upgrade pandas scikit-learn gradio matplotlib openpyxl')

# Load the dataset with a raw string to handle Windows paths correctly
file_path = r'D:\projects\CGPA prediction\3rdsem.xlsx'  # Use raw string (r) to avoid escape sequence issues
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the path.")
    exit(1)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Select features (internal marks) and target (GPA)
X = data.drop(columns=['S.NO', 'GPA'])
y = data['GPA']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Regression (SVR) model
model = SVR(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Function to predict the fourth-semester GPA
def predict_gpa(subject1, subject2, subject3, subject4, subject5, subject6, subject7, subject8, subject9, subject10, subject11):
    """
    Predicts the fourth-semester GPA based on user-defined internal marks for 11 subjects.
    """
    subject_marks = [subject1, subject2, subject3, subject4, subject5, subject6, subject7, subject8, subject9, subject10, subject11]
    predicted_gpa = model.predict([subject_marks])[0]
    return f"Predicted GPA: {predicted_gpa:.2f}"

# Function to create a Matplotlib plot
def plot_actual_vs_predicted():
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', edgecolors='k', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual GPA')
    plt.ylabel('Predicted GPA')
    plt.title('Actual vs Predicted GPA')
    plt.grid(True)
    return plt

# Gradio interface components
inputs = [gr.Number(value=50, label=f"Subject {i+1} Marks", minimum=0, maximum=100) for i in range(11)]
outputs = [gr.Textbox(label="Predicted 4th Semester GPA"), gr.Plot(label="Actual vs Predicted GPA Plot")]

# Combined function for prediction and plotting
def combined_interface(subject1, subject2, subject3, subject4, subject5, subject6, subject7, subject8, subject9, subject10, subject11):
    gpa_result = predict_gpa(subject1, subject2, subject3, subject4, subject5, subject6, subject7, subject8, subject9, subject10, subject11)
    plot = plot_actual_vs_predicted()
    return gpa_result, plot

# Launch Gradio interface
interface = gr.Interface(
    fn=combined_interface,
    inputs=inputs,
    outputs=outputs,
    title="4th Semester GPA Predictor",
    description="Enter your internal marks (0-100) for 11 subjects to predict your 4th semester GPA."
).launch()