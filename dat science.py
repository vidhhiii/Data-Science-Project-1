import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import Button, Label, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize the global variable
current_plot = None

def plot_distribution():
    global current_plot
    if current_plot:
        current_plot.get_tk_widget().destroy()  # Destroy the old plot if it exists

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['MedHouseVal'], bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Median House Values')
    ax.set_xlabel('Median House Value')
    ax.set_ylabel('Frequency')

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    global current_plot
    current_plot = canvas

def plot_pairplot():
    global current_plot
    if current_plot:
        current_plot.get_tk_widget().destroy()  # Destroy the old plot if it exists

    # Create a new figure for pairplot
    fig = plt.figure(figsize=(10, 8))
    sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']])
    
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    global current_plot
    current_plot = canvas

def plot_correlation_matrix():
    global current_plot
    if current_plot:
        current_plot.get_tk_widget().destroy()  # Destroy the old plot if it exists

    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Correlation Matrix')

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    global current_plot
    current_plot = canvas

def plot_predictions():
    global current_plot
    if current_plot:
        current_plot.get_tk_widget().destroy()  # Destroy the old plot if it exists

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title('True Values vs. Predictions')

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    global current_plot
    current_plot = canvas

# Load the dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

# Combine features and target into one DataFrame for exploration
df = pd.concat([X, y], axis=1)

# Data Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create the Tkinter GUI
root = tk.Tk()
root.title("Housing Data Analysis")
root.geometry("800x600")  # Set window size

# Main Frame
main_frame = Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Title Label
title_label = Label(main_frame, text="Housing Data Analysis", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

# Button Frame
button_frame = Frame(main_frame)
button_frame.pack(pady=10)

# Buttons
Button(button_frame, text="Distribution of Median House Values", command=plot_distribution, width=30).pack(pady=5)
Button(button_frame, text="Pairplot of Features", command=plot_pairplot, width=30).pack(pady=5)
Button(button_frame, text="Correlation Matrix", command=plot_correlation_matrix, width=30).pack(pady=5)
Button(button_frame, text="True Values vs. Predictions", command=plot_predictions, width=30).pack(pady=5)

# Plot Frame
plot_frame = Frame(main_frame)
plot_frame.pack(fill=tk.BOTH, expand=True)

# Run the application
root.mainloop()
