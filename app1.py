from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import traceback
import joblib

app = Flask(__name__)

# Load saved model and preprocessing tools
model = joblib.load("model1.pkl")
scaler = joblib.load("scaler1.pkl")
encoders = joblib.load("encoders1.pkl")
feature_order = joblib.load("feature_names1.pkl")

# Load raw dataset for fallback stats
df = pd.read_csv("/Users/varungirish/docs/college/AI_ML/project/karnataka.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df.rename(columns={"yeilds": "yield"}, inplace=True)
df.drop(columns=["area"], inplace=True, errors="ignore")
df.dropna(inplace=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    locations = list(encoders["location"].classes_)
    crops = list(encoders["crops"].classes_)
    irrigations = list(encoders["irrigation"].classes_)

    if request.method == "POST":
        try:
            location = request.form["location"]
            crop = request.form["crop"]
            irrigation = request.form["irrigation"]
            yield_kg = float(request.form["yield"])

            # Encode inputs
            loc_enc = encoders["location"].transform([location])[0]
            crop_enc = encoders["crops"].transform([crop])[0]
            irr_enc = encoders["irrigation"].transform([irrigation])[0]

            # Subset for fallback values
            subset = df[
                (df["location"] == location) &
                (df["crops"] == crop) &
                (df["irrigation"] == irrigation)
            ]

            if subset.empty:
                subset = df[df["location"] == location]
            if subset.empty:
                subset = df.copy()

            avg = subset.mean(numeric_only=True).to_dict()
            season = subset["season"].mode().iloc[0] if not subset["season"].mode().empty else df["season"].mode().iloc[0]
            soil = subset["soil_type"].mode().iloc[0] if not subset["soil_type"].mode().empty else df["soil_type"].mode().iloc[0]
            year = int(subset["year"].mode().iloc[0] if not subset["year"].mode().empty else df["year"].mode().iloc[0])

            # Encode fallback values
            season_enc = encoders["season"].transform([season])[0]
            soil_enc = encoders["soil_type"].transform([soil])[0]

            input_dict = {
                "rainfall": avg.get("rainfall", 0),
                "temperature": avg.get("temperature", 0),
                "humidity": avg.get("humidity", 0),
                "price": avg.get("price", 0),
                "season": season_enc,
                "soil_type": soil_enc,
                "year": year,
                "location": loc_enc,
                "irrigation": irr_enc,
                "yield": yield_kg
            }

            input_df = pd.DataFrame([input_dict])[feature_order]
            input_scaled = scaler.transform(input_df)

            result = model.predict(input_scaled)[0]
            predicted_crop = encoders["crops"].inverse_transform([result])[0]
            prediction = f"\U0001F33E Recommended Crop: {predicted_crop}"

        except Exception:
            error = "Something went wrong:<br><pre>" + traceback.format_exc() + "</pre>"

    return render_template("index.html",
                           prediction=prediction,
                           error=error,
                           locations=locations,
                           crops=crops,
                           irrigations=irrigations)

if __name__ == "__main__":
    app.run(debug=True)