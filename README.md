# Creditcard-Fraud-Detection-using-meachine-learning
  ![cre](https://github.com/user-attachments/assets/df93fed1-bc1a-4a57-81e1-c4fd340fb31d)


## Overview
This project aims to develop a machine learning model for credit card fraud detection using a dataset of credit card transactions. The dataset includes both legitimate and fraudulent transactions, and the goal is to predict fraudulent transactions based on various features. This project employs logistic regression for classification, and the dataset has been preprocessed to handle imbalances between the classes.

### Table of Contents
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Libraries Used](#libraries-used)
4. [Steps to Run](#steps-to-run)
5. [Results](#results)
6. [Conclusion](#conclusion)

## Project Description
The project applies machine learning techniques to identify fraudulent credit card transactions. The dataset used for this project is highly imbalanced, with only 492 instances of fraud compared to 284,315 legitimate transactions. To deal with this imbalance, an under-sampling technique is used, where the number of fraudulent transactions is matched with the number of legitimate transactions.

The project follows these major steps:
1. **Data Preprocessing**: Handles missing values, explores data types, and provides basic statistics.
2. **Data Resampling**: Uses under-sampling to create a balanced dataset.
3. **Model Building**: A logistic regression model is trained to classify transactions.
4. **Model Evaluation**: The accuracy of the model is evaluated using a confusion matrix and accuracy score.

method to implement
![image](https://github.com/user-attachments/assets/67bacddd-9ce7-4c5d-8c49-34bf7534a3d1)

## Dataset
The dataset used in this project is from the **UCI Machine Learning Repository** and contains information about credit card transactions.

- The dataset has 31 columns, including features such as `V1`, `V2`, ... `V28`, `Amount`, and `Class`.
- The `Class` column indicates whether the transaction is legitimate (0) or fraudulent (1).
- The `Amount` column indicates the transaction amount.

### Data Information:
- **Legitimate Transactions (Class 0)**: 284,315
- **Fraudulent Transactions (Class 1)**: 492

## Libraries Used
This project uses the following libraries:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning, including logistic regression, model evaluation, and train-test splitting.
- **Matplotlib/Seaborn**: For data visualization (optional, not included in the code above).
- **Google Colab**: For running the code in a cloud environment.

## Steps to Run

1. **Clone the Repository**:
   Clone this repository to your local machine or use Google Colab to access the code directly.

   ```bash
   git clone https://github.com/username/credit-card-fraud-detection.git
   ```

2. **Install the Required Libraries**:
   Make sure you have all required libraries installed. You can do this by running:

   ```bash
   pip install pandas numpy scikit-learn
   ```

3. **Upload the Dataset**:
   Upload the dataset (e.g., `CreditCardDefault.csv`) to Google Drive or ensure it is available on your local system. You can modify the code to load the dataset from a local file or cloud storage.

4. **Run the Code**:
   Execute the provided Jupyter Notebook or Python script to preprocess the data, build the model, and evaluate it.

   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score
   ```

   Follow the instructions in the notebook to run each cell.

## Results
The model's performance is evaluated using accuracy score, confusion matrix, and classification report. Below are some key results:

- **Accuracy Score**: Shows how well the model predicts both legitimate and fraudulent transactions.
- **Confusion Matrix**: Provides a detailed breakdown of the number of correct and incorrect classifications.
- **Precision, Recall, F1-Score**: Metrics that evaluate the model’s ability to correctly identify fraudulent transactions.

*Example Results:*

```plaintext
Accuracy: 99.9%
Precision: 0.95
Recall: 0.75
F1-Score: 0.84
```
## Conclusion
The credit card fraud detection model is able to effectively classify transactions as legitimate or fraudulent using logistic regression. Due to the class imbalance, the dataset is carefully preprocessed using under-sampling techniques to ensure balanced class distributions.

Despite the imbalance in the original dataset, the model provides a reasonable performance in identifying fraudulent transactions, which is critical in real-world applications for financial institutions.


