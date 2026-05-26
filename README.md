

## Overview

Invisible Maps is a Machine Learning project developed to analyze and predict disaster vulnerability patterns in Colombian municipalities using historical emergency data from the Colombian National Unit for Disaster Risk Management (UNGRD).

The project follows the CRISP-ML methodology and integrates multiple Machine Learning techniques, including:

- K-Means Clustering
- Logistic Regression
- Random Forest Classification

In addition, a Flask web application was developed to provide an interactive environment where users can explore concepts, visualize results, and generate predictions in real time.

---

## Project Objectives

The main objectives of this project are:

- Analyze historical emergency events reported by UNGRD.
- Identify territorial vulnerability patterns.
- Discover hidden geographical clusters using unsupervised learning.
- Classify risk levels using supervised Machine Learning models.
- Visualize disaster risk distribution through interactive maps.
- Provide an educational platform that combines theory, model implementation, and prediction tools.

---

## CRISP-ML Methodology

The project was developed following the CRISP-ML process:

### 1. Business Understanding

Understanding disaster risk management challenges in Colombia and defining the analytical objectives.

### 2. Data Understanding

Exploration and analysis of historical emergency records provided by UNGRD.

### 3. Data Engineering

Data cleaning, transformation, feature selection, encoding, and preparation for model training.

### 4. Model Engineering

Implementation and comparison of:

- K-Means Clustering
- Logistic Regression
- Random Forest

### 5. Model Evaluation

Performance assessment using:

- Accuracy
- Confusion Matrix
- ROC Curve
- Feature Importance Analysis

### 6. Deployment

Development of a Flask-based web application capable of generating real-time predictions and visualizations.

---

## Dataset

The project uses historical emergency and disaster event data from Colombia.

### Main Features

- Department
- Municipality
- Event Type
- Event Frequency
- Event Diversity
- Geographic Information

### Processed Files

- `UNGRD_Cleaned.xlsx`
- `UNGRD_Cleaned2.xlsx`
- `UNGRD_Scaled.xlsx`
- `Emergencias_UNGRD_Translated.xlsx`

---

## Machine Learning Models

### K-Means Clustering

Used to identify municipalities with similar disaster-risk characteristics and discover hidden territorial patterns.

Outputs:

- Cluster Assignments
- Elbow Method Analysis
- Cluster Visualization

### Logistic Regression

Used as a baseline supervised classification model to estimate vulnerability levels.

### Random Forest

The primary predictive model of the project.

Advantages:

- High predictive performance
- Robustness against overfitting
- Feature importance analysis
- Real-time prediction capability

Saved artifacts:

- `rf_model.pkl`
- `rf_encoder.pkl`
- `rf_le_dept.pkl`
- `rf_le_event.pkl`

---

## Geographic Risk Visualization

The project includes a geographic visualization system based on:

- Colombian municipalities GeoJSON
- Risk classification results
- Interactive maps generated with Python

Files:

- `municipalitiescolombia.geojson`
- `map.py`

Generated map:

- `ungrd_colombia_risk_map.png`

---

## Flask Web Application

The application provides educational and interactive modules.

### Main Sections

#### Home

Introduction to Machine Learning and project objectives.

#### Business Understanding

Project context and problem definition.

#### Data Understanding

Dataset exploration and analysis.

#### Data Engineering

Data preparation process.

#### Model Engineering

Explanation of implemented Machine Learning models.

#### Model Evaluation

Performance metrics and model comparison.

#### Prediction System

Interactive interface for generating risk predictions.

---

## Visual Results

### Evaluation Metrics

- Confusion Matrix
- ROC Curve
- Feature Importance

Generated files:

- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`

### Clustering Results

- `kmeans_clusters.png`
- `elbow_method.png`

---

## Technologies Used

### Backend

- Python
- Flask

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Folium

### Frontend

- HTML5
- CSS3

### Geographic Analysis

- GeoJSON

---

## Project Structure

```text
PMLN/
│
├── app.py
├── ClimateClustering.py
├── logisticRegressionModel.py
├── rforest.py
├── dataEngeneering.py
├── map.py
│
├── templates/
├── static/
│
├── municipalitiescolombia.geojson
│
├── rf_model.pkl
├── rf_encoder.pkl
├── rf_le_dept.pkl
├── rf_le_event.pkl
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jhohantellez/PML-N.git
cd PML-N
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Authors

Jhohan Téllez
Andrey Lopez
Felipe Navarro

Systems Engineering Student

Machine Learning Project – CRISP-ML Methodology

2026

---

## License

This project was developed for academic and educational purposes.
