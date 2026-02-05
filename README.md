# No-Code Machine Learning Comparison Web App

**Author:** Sakib Hossain Tahmid

**Live App:** [Click here](https://ml-app-sakib.streamlit.app/)

<p align="center">
  <img src="https://github.com/sakib-12345/ML-comperision-WEBapp/blob/main/web_icon.png" alt="Application Screenshot" width="250">
</p>


## Introduction

The **No-Code Machine Learning Comparison Web App** is a Streamlit-based platform that allows users to **train, evaluate, and compare multiple machine learning models** without writing any code. The app is designed to help users quickly identify the best-performing model for a given dataset.

This project focuses on **model comparison and performance clarity**, making it ideal for learning, experimentation, and rapid ML prototyping.



## What the App Does

The workflow is simple and efficient:

1. Upload your dataset
2. Apply automatic preprocessing
3. Train multiple ML models at once
4. Compare accuracy and results
5. Identify the top-performing model

All steps are handled through an interactive web interface.



## Core Features

### Data Upload

* Upload datasets in CSV format
* Automatic detection of columns and structure

### Automatic Preprocessing

* Handles missing values
* Performs basic data cleaning
* Prepares data for model training

### Multi-Model Training

* Train **five different machine learning models** in a single run
* Supports classification tasks
* No manual configuration required

### Model Comparison

* Displays accuracy scores for all models
* Highlights the best-performing model
* View results for each model individually



## Tech Stack

| Layer                | Technology   |
| -------------------- | ------------ |
| Programming Language | Python       |
| Machine Learning     | Scikit-learn |
| Data Processing      | Pandas       |
| Frontend             | Streamlit    |



## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sakib-12345/ML-comperision-WEBapp.git
```

### 2. Navigate to Project Directory

```bash
cd ML-comperision-WEBapp
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Application Locally

```bash
streamlit run app.py
```

The application will open automatically in your default browser.


## Live Demo

You can try the deployed version here:

[https://ml-app-sakib.streamlit.app/](https://ml-app-sakib.streamlit.app/)



## Use Cases

* Comparing ML models on the same dataset
* Learning model performance differences
* Classroom demonstrations
* Quick experimentation with datasets



## Future Improvements

* Support for regression model comparison
* Additional evaluation metrics (precision, recall, F1-score)
* Dataset visualization before training
* Downloadable comparison reports


## License

This project is licensed under the **MIT License**.

You are free to use, copy, modify, and distribute this software, provided that the original license and copyright notice are included.
