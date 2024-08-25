# Online Payment Fraud Detection

## Project Overview
**Online Payment Fraud Detection** is a data science project aimed at identifying fraudulent transactions in online payment systems. The project leverages machine learning techniques to analyze transaction data and distinguish between legitimate and fraudulent activities. By building and evaluating predictive models, the project aims to enhance security measures and reduce financial losses due to fraud.

## Introduction
With the rise of e-commerce and digital transactions, online payment fraud has become a significant threat. This project focuses on building robust machine learning models to detect fraudulent transactions effectively. The models are trained on historical transaction data and are designed to generalize well to unseen data, ensuring reliable fraud detection in real-world scenarios.

## Dataset
The dataset used for this project contains records of online transactions, each labeled as fraudulent or legitimate. It includes features such as transaction amount, payment method, location, and time of transaction. The dataset is cleaned, preprocessed, and split into training and testing sets to develop and evaluate the models.

- **Dataset link**: [Kaggle link](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection)

An example of the data structure:
```csv
1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,0,0
1,PAYMENT,1864.28,C1666544295,21249.0,19384.72,M2044282225,0.0,0.0,0,0
```

The data file `onlinefraud.csv` should be placed in the root directory of the project.

## Installation

To run this project locally, you'll need to have Python installed. Follow these steps to set up the environment:

1. **Clone the Repository:**
```bash
git clone https://github.com/Vaibhav-kesarwani/Online_Payment_Fraud_Detection.git
cd Online_Payment_Fraud_Detection
```

2. **Create a Virtual Environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install Required Packages:** Install the dependencies by running
```bash
pip install -r requirements.txt
```

## Usage

### Running the Project
To run the Text Emotion Classifier, follow these steps:
1. **Prepare the Dataset:** Ensure that your `onlinefraud.csv` file is in the root directory. This file should contain the text data and corresponding labels, separated by a comma `(,)`.
2. **Run the Script:** Execute the main script to load the data and perform emotion classification
```bash
python main.ipynb
```
3. **Output:** The script will print the first few rows of the dataset to the console, showing the text samples and their associated emotion labels.

## Transaction Type Distribution Visualization

To understand the distribution of different transaction types within the dataset, a pie chart was created using Plotly Express. This chart provides a clear visual representation of the proportion of each transaction type, such as 'fraudulent' and 'legitimate,' in the dataset.

The steps involved include:
- Counting the occurrences of each transaction type.
- Extracting the transaction types and their respective counts.
- Creating a donut-style pie chart to visualize the distribution, with a hole in the center to emphasize the relative sizes of each slice.

The resulting chart helps in quickly identifying which transaction type is more prevalent, offering valuable insights into the dataset's composition.

```python
type = data['type'].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px # type: ignore
figure = px.pie(data, values = quantity, names = transactions, hole = 0.5, title = "Distribution of Transaction Type")
figure.show()
```
<br />

![visualise](https://github.com/user-attachments/assets/342d97d0-d6b3-4d08-abee-993cf87f5855)

## Model Training
The model training is performed within the `main.ipynb` script, which processes the text data and trains a Sequential model using sklearn. You can modify the model architecture, training parameters, or the data processing steps within this script.

```python
# Training the machine learning model
from sklearn.tree import DecisionTreeClassifier # type: ignore
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)
```

## Prediction
After training the model, you can use it to predict emotions from new text inputs. Implement the prediction logic in a separate script or extend `main.ipynb` to include a prediction function.

```python
# Prediction
# features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))
```


















