# Iris Flower Classification

This project focuses on classifying Iris flowers into different species based on the features of the flower's petals and sepals. The dataset used for this project is the famous **Iris Dataset**, which includes 150 samples from three different species of Iris flowers: **Setosa**, **Versicolor**, and **Virginica**.

## Project Overview

The objective of this project is to build a machine learning model that can predict the species of an Iris flower based on its physical measurements. This is a classic example of a classification problem where the goal is to predict the categorical label (species) from the input features (petal length, petal width, sepal length, and sepal width).

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Steps](#steps)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Installation

To run this project, you need to have Python installed along with a few libraries. Below are the installation steps:

-----------------------------------------

## Dataset

The **Iris Dataset** consists of 150 rows of data with 5 columns:

- `sepal_length`: The length of the sepal in cm
- `sepal_width`: The width of the sepal in cm
- `petal_length`: The length of the petal in cm
- `petal_width`: The width of the petal in cm
- `species`: The species of the flower (Setosa, Versicolor, Virginica)

The dataset is publicly available and can be found at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

## Technologies Used

- **Python 3.x**
- **Scikit-learn**: For building machine learning models and evaluating them
- **Pandas**: For data manipulation and analysis
- **Matplotlib** and **Seaborn**: For data visualization
- **Jupyter Notebook**: For interactive development and experimentation

## Steps

The process for this project can be broken down into the following steps:

1. **Data Loading**:
   - Import the Iris dataset using the `load_iris` function from Scikit-learn.
   - Load the data into a pandas DataFrame for better inspection and manipulation.

2. **Exploratory Data Analysis (EDA)**:
   - Perform data exploration to check for missing values, outliers, and basic statistics.
   - Visualize the relationships between features using pairplots and correlation heatmaps.

3. **Data Preprocessing**:
   - Split the dataset into training and testing sets (80% training, 20% testing).
   - Scale the feature values using `StandardScaler` to standardize the data.

4. **Model Building**:
   - Implement multiple classification algorithms such as:
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier
     - Support Vector Machine (SVM)
     - Logistic Regression
   - Train each model on the training data.

5. **Model Evaluation**:
   - Evaluate the models using accuracy, confusion matrix, and classification reports.
   - Use cross-validation to ensure the stability and robustness of the models.

6. **Model Tuning**:
   - Fine-tune hyperparameters for the best model using GridSearchCV or RandomizedSearchCV.

7. **Final Model**:
   - Choose the best-performing model and finalize it for prediction.

## Results

The final results showed that the **K-Nearest Neighbors (KNN)** classifier achieved an accuracy of **90%** in classifying the species of Iris flowers. Other models like Decision Trees, Support Vector Machine, and Logistic Regression were also evaluated with the following accuracy results:

- K-Nearest Neighbors (KNN): **90%**
- Decision Tree Classifier: **85%**
- Support Vector Machine (SVM): **88%**
- Logistic Regression: **84%**

The KNN model performed the best with feature scaling and hyperparameter tuning.

## Usage

Once the model is trained, you can predict the species of new Iris flowers by inputting the following four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

