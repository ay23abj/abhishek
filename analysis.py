import pandas as pd
import zipfile
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

# Download the dataset from Kaggle
path = kagglehub.dataset_download("valakhorasani/gym-members-exercise-dataset")
print("Path to dataset files:", path)

# List files in the downloaded directory to find the CSV file
file_list = os.listdir(path)
print("Files in dataset:", file_list)

# Define the CSV file path
csv_file_path = os.path.join(path, 'gym_members_exercise_tracking.csv')

# Load the CSV file into a DataFrame and display basic statistics
data = pd.read_csv(csv_file_path)
print("Descriptive Statistics:\n", data.describe())

# Plot Age Histogram
plt.figure(figsize=(8, 6))
plt.hist(data['Age'], bins=20, edgecolor='black')
plt.title("Age Distribution of Gym Members")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Plot Workout Type Pie Chart
workout_counts = data['Workout_Type'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(workout_counts, labels=workout_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Workout Types")
plt.show()

# Scatter Plot with Trend Line for Calories Burned vs. Session Duration
plt.figure(figsize=(8, 6))
sns.regplot(x='Session_Duration (hours)', y='Calories_Burned', data=data, scatter_kws={'alpha':0.6})
plt.title("Calories Burned vs. Session Duration with Trend Line")
plt.xlabel("Session Duration (hours)")
plt.ylabel("Calories Burned")
plt.show()

# Plot Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data[['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
                           'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 
                           'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)', 
                           'Experience_Level', 'BMI']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Variables")
plt.show()

# Box Plot for Calories Burned by Experience Level
plt.figure(figsize=(8, 6))
sns.boxplot(x='Experience_Level', y='Calories_Burned', data=data)
plt.title("Calories Burned by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Calories Burned")
plt.show()
