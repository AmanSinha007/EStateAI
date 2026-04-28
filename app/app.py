from flask import Flask, request, render_template
import pickle
import json
import numpy as np
import shap
import os
import pandas as pd

# 🔥 Base directory (project root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 🔥 Flask setup
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# 🔥 Load model files
model = pickle.load(open(os.path.join(BASE_DIR, "model/model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "model/scaler.pkl"), "rb"))

with open(os.path.join(BASE_DIR, "model/columns.json"), "r") as f:
    columns = json.load(f)['data_columns']

# 🔥 SHAP explainer
explainer = shap.Explainer(model)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # 🔹 User inputs
        bhk = float(data.get('bhk', 0))
        bath = float(data.get('bath', 0))
        area = float(data.get('area', 0))
        floors = float(data.get('number_of_floors', 0))
        condition = float(data.get('condition_of_the_house', 0))
        grade = float(data.get('grade_of_the_house', 0))

        # Feature engineering
        total_rooms = bhk + bath
        area_per_room = area / total_rooms if total_rooms != 0 else 0

        # Input data must be same in columns.json
        input_data = [
            bhk,
            bath,
            area,
            floors,
            condition,
            grade,
            total_rooms,
            area_per_room
        ]

        #Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=columns)

        # Scale input
        x = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(x)[0]
        prediction_inr = prediction * 94

        # SHAP values
        shap_values = explainer(x)
        shap_vals = shap_values.values[0]
        base_value = shap_values.base_values[0]

        # Convert to INR
        shap_vals = shap_vals * 94
        base_value_inr = base_value * 94

        # Safety fallback
        if shap_vals is None or len(shap_vals) == 0:
            shap_vals = np.array([0])
            feature_names = ["No Data"]
        else:
            feature_names = columns.copy()

        # Top 3 negative + top 3 positive
        sorted_idx = np.argsort(shap_vals)
        neg_idx = sorted_idx[:3]
        pos_idx = sorted_idx[-3:]
        selected_idx = np.concatenate([neg_idx, pos_idx])

        shap_vals = shap_vals[selected_idx]
        feature_names = [feature_names[i] for i in selected_idx]

        # Sort
        order = np.argsort(shap_vals)
        shap_vals = shap_vals[order]
        feature_names = [feature_names[i] for i in order]

        #names
        feature_names = [f.replace("_", " ").title() for f in feature_names]

        return render_template(
            "result.html",
            prediction_text=f"Estimated Price: ₹ {prediction_inr:,.0f}",
            shap_values=shap_vals.tolist(),
            feature_names=feature_names,
            base_value=f"₹ {base_value_inr:,.0f}"
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction_text=f"Error: {str(e)}",
            shap_values=[0],
            feature_names=["Error"],
            base_value="N/A"
        )


if __name__ == "__main__":
    app.run(debug=True)