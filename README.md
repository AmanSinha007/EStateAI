# 🏠 EStateAI – House Price Prediction System

##  Overview

EStateAI is a machine learning-based web application that predicts house prices based on user inputs such as area, number of rooms, and house condition.
It also provides **explainable AI insights** using SHAP values to show how each feature affects the prediction.

---

## Features

*  Accurate house price prediction
*  Feature contribution visualization (SHAP graph)
*  Base price comparison (average market price)
*  Explainable AI (XAI)
*  Clean and responsive UI
*  Flask-based web application

---

## Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn (Linear Regression)
* **Explainability:** SHAP
* **Frontend:** HTML, CSS, Chart.js
* **Tools:** Jupyter Notebook, Git, GitHub

---

## 📁 Project Structure

```
EStateAI/
│
├── app/                # Flask application
│   └── app.py
│
├── model/              # Saved ML model & files
│   ├── model.pkl
│   ├── scaler.pkl
│   └── columns.json
│
├── data/               # Dataset
│   └── india_house.csv
│
├── notebook/           # Model training notebook
│   └── analysis.ipynb
│
├── templates/          # HTML pages
│   ├── index.html
│   └── result.html
│
├── static/             # CSS files
│   └── style.css
│
├── requirements.txt    # Dependencies
└── README.md
```

---

## How to Run the Project

### Clone the repository

```
git clone https://github.com/yourusername/EStateAI.git
cd EStateAI
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run the Flask app

```
python app/app.py
```

### Open in browser

```
http://127.0.0.1:5000
```

---

##  Model Details

* **Algorithm:** XGBoost

* **Features Used:**

  * bhk
  * bath
  * area
  * number_of_floors
  * condition_of_the_house
  * grade_of_the_house
  * total_rooms
  * area_per_room

* **Target Variable:** Price

---

## Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to:

* Show **feature impact on prediction**
* Highlight **positive & negative contributions**
* Improve model transparency

---

##  Output

*  Predicted Price
*  Base Price (average)
*  Feature contribution graph

---

## Author

**Aman Sinha**
**Vikas Yadav**
**Akash Singh**

---

##  Future Improvements

* Add more features (location, amenities)
* Use advanced models (Random Forest)
* Deploy on cloud (Render / AWS / Heroku)

---

## Conclusion

This project demonstrates how machine learning and explainable AI can be combined to build a transparent and user-friendly prediction system.

---
