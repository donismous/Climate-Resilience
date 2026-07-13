# 🌍 Climate Resilience Dashboard

An interactive data science application that explores how climate resilience has evolved across countries and estimates future resilience trends using historical data.

The project combines climate adaptation indicators with machine learning to help users understand how different countries have progressed over time and identify those that may become more or less resilient in the future.

---

## 📖 Project Overview

Climate change is affecting countries differently depending on their environmental, economic and social conditions. Governments, businesses and individuals increasingly need data-driven insights to understand resilience and adaptation capacity.

This dashboard allows users to:

* Explore historical climate resilience indicators
* Compare countries over time
* Visualize long-term trends
* Generate short-term forecasts using machine learning models
* Better understand which factors contribute most to climate resilience

---

## 🎯 Objectives

The project aims to:

* Build an end-to-end machine learning pipeline
* Demonstrate MLOps best practices
* Create an interactive dashboard for data exploration
* Produce explainable predictions

---

## 📊 Dataset

This project uses the **Notre Dame Global Adaptation Initiative (ND-GAIN)** dataset.

The dataset contains annual country-level indicators describing climate vulnerability and readiness across approximately 190 countries over multiple decades.

Example indicators include:

* Vulnerability
* Readiness
* Food
* Water
* Health
* Ecosystem
* Infrastructure
* Governance

## 📊 Dataset Folder Structure
```
raw_data/
├── readiness/
│   └── readiness.csv
├── sector/
│   ├── economic.csv
│   ├── governance.csv
│   └── social.csv
└── vulnerability/
    ├── capacity.csv
    ├── exposure.csv
    ├── sensitivity.csv
    ├── vulnerability.csv
    └── sectors/
        ├── ecosystems.csv
        ├── food.csv
        ├── habitat.csv
        ├── health.csv
        ├── infrastructure.csv
        └── water.csv
```
        
**Note:** The raw datasets are excluded from Git using `.gitignore` because they are large. Download the ND-GAIN data and place it in the `raw_data/` directory following the structure above.

## 🛠 Technologies

### Programming

* Python

### Data Analysis

* pandas
* NumPy

### Machine Learning

* scikit-learn

### Visualization

* Plotly
* Matplotlib

### Dashboard

* Streamlit

### MLOps

* Google Cloud Platform
* BigQuery
* Docker
* GitHub
* Cloud Run

---

## 🚀 Installation

Clone the repository

```bash
git clone git@github.com:donismous/Climate-Resilience.git
```

Move into the project directory

```bash
cd Climate-Resilience
```

---

## ▶ Running the Project


```bash
python main.py
```

or, if using Streamlit,

```bash
streamlit run app.py
```

---

## 📈 Machine Learning Workflow

1. Load climate resilience data
2. Clean and preprocess data
3. Engineer features
4. Train forecasting model
5. Evaluate performance
6. Generate predictions
7. Display results in the dashboard

---

## 🔮 Future Improvements

* Interactive scenario analysis
* Country recommendation engine

---
