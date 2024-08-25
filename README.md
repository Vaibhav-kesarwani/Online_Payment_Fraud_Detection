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
1. **Prepare the Dataset:** Ensure that your `onlinefraud.csv` file is in the root directory. This file should contain the text data and corresponding labels, separated by a semicolon (;).
2. **Run the Script:** Execute the main script to load the data and perform emotion classification
```bash
python main.ipynb
```
3. **Output:** The script will print the first few rows of the dataset to the console, showing the text samples and their associated emotion labels.







