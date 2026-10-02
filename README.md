# Dubai Property Cost Calculator

A machine learning web application that estimates Dubai property transaction values using historical Dubai real estate transaction data.
<img width="1188" height="888" alt="image" src="https://github.com/user-attachments/assets/549bd39d-12aa-4912-9bab-cd9f03f5555e" />
<img width="1063" height="437" alt="image" src="https://github.com/user-attachments/assets/539a9b36-0eb0-4acf-bc1e-e5f49967921a" />

## Overview

The project uses historical Dubai property transactions from 2026 to build a machine learning model for property value estimation.

Users can enter:

- Area
- Project
- Property type
- Number of rooms
- Property area
- Transaction type
- Ownership
- Nearest metro
- Nearest mall
- Nearest landmark

The application then generates an estimated property transaction value.

## Machine Learning

The final deployment model uses XGBoost with a log-transformed target variable.

The model incorporates:

- Property characteristics
- Location information
- Transaction information
- Historical price-per-square-metre statistics
- Historical transaction counts
- Market-level price information

## Dataset

The dataset contains 18,085 Dubai property transactions from 2026.

## Project Structure

```text
dubai-property-price-prediction/
├── data/
├── models/
├── reports/
├── src/
├── .streamlit/
├── .gitignore
├── README.md
└── requirements.txt
