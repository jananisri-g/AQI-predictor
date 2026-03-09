import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — AQI TIME-SERIES DATA & FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

file_path = "delhi_data.xlsx"
df = pd.read_excel(file_path)

df_long = df.melt(id_vars=["Date"], var_name="Time", value_name="AQI")
df_long["Time"] = df_long["Time"].astype(str)
df_long["Datetime"] = pd.to_datetime(df_long["Date"].astype(str) + " " + df_long["Time"])
df_long = df_long.sort_values("Datetime").reset_index(drop=True)
df_long["AQI"] = pd.to_numeric(df_long["AQI"], errors="coerce")
df_long = df_long.dropna()

df_long["hour"]       = df_long["Datetime"].dt.hour
df_long["day"]        = df_long["Datetime"].dt.day
df_long["dayofweek"]  = df_long["Datetime"].dt.dayofweek

df_long["lag1"]  = df_long["AQI"].shift(1)
df_long["lag2"]  = df_long["AQI"].shift(2)
df_long["lag24"] = df_long["AQI"].shift(24)
df_long["lag48"] = df_long["AQI"].shift(48)
df_long["lag72"] = df_long["AQI"].shift(72)

df_long["rolling_mean_3"]  = df_long["AQI"].rolling(3).mean()
df_long["rolling_mean_6"]  = df_long["AQI"].rolling(6).mean()
df_long["rolling_mean_12"] = df_long["AQI"].rolling(12).mean()

df_long = df_long.dropna().reset_index(drop=True)

FEATURES = [
    "hour", "day", "dayofweek",
    "lag1", "lag2", "lag24", "lag48", "lag72",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
]

X = df_long[FEATURES]
y = df_long["AQI"]

split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# ── XGBoost AQI Regressor ─────────────────────────────────────────────────────
aqi_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
)
aqi_model.fit(X_train, y_train)

preds = aqi_model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
print(f"AQI Model — Mean Absolute Error: {mae:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MULTI-LABEL RECOMMENDATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────
#
# Design:
#   • Each population group has its own dedicated MultiOutputClassifier
#     (RandomForest) that predicts which advice items apply at a given AQI.
#   • Training data is synthetically generated from the same domain-expert
#     rules that previously lived in the if/elif block — this is standard
#     practice when labelled recommendation data is unavailable.
#   • The model learns a continuous mapping: AQI (float) → binary vector of
#     applicable advice items, generalising beyond fixed thresholds.
# ──────────────────────────────────────────────────────────────────────────────

# ── Master advice catalogue ───────────────────────────────────────────────────
# Each group maps to ALL advice items it could ever display.
# The model learns which subset applies at each AQI level.

GROUPS = {
    "Children (0–12 yrs)": [
        "Outdoor play is safe up to 2 hours.",
        "Avoid playgrounds next to highways or construction zones.",
        "Limit outdoor play to less than 1.5 hours.",
        "Avoid sports near traffic intersections.",
        "Restrict outdoor play to less than 60 minutes.",
        "Avoid football, cricket, or running activities outdoors.",
        "Prefer indoor activities.",
        "Avoid outdoor sports and playground activity.",
        "Keep children indoors during morning and evening pollution peaks.",
        "Completely avoid outdoor play.",
        "Keep school activities indoors.",
        "Do not allow outdoor exposure.",
        "Keep indoor air filtered using HEPA purifier.",
    ],
    "Teenagers / Athletes": [
        "Moderate sports training allowed but avoid high-intensity endurance activities.",
        "Reduce high-intensity training sessions outdoors.",
        "Use indoor gym facilities if possible.",
        "Cancel outdoor training sessions.",
        "Switch to indoor physical activities.",
    ],
    "Healthy Adults": [
        "Outdoor exercise such as running or cycling is safe.",
        "Hydrate normally and avoid heavy traffic areas.",
        "Outdoor jogging or cycling allowed but avoid peak traffic hours.",
        "Limit outdoor exercise to less than 30 minutes.",
        "Use NIOSH-certified N95 or KN95 respirator during commuting.",
        "Avoid jogging or cycling outdoors.",
        "Wear NIOSH-approved N95 respirator during travel.",
        "Limit outdoor exposure to less than 20 minutes.",
        "Avoid outdoor exercise completely.",
        "Use tightly fitted N95 respirator if outdoor travel is unavoidable.",
        "Avoid all outdoor physical activity.",
        "Wear NIOSH-certified N95 or N99 respirator if leaving home.",
    ],
    "Elderly (65+)": [
        "Normal walking and outdoor activities allowed.",
        "Carry regular medications if you have chronic conditions.",
        "Limit outdoor exposure to less than 1 hour.",
        "Avoid brisk walking near busy roads.",
        "Limit outdoor exposure to less than 30 minutes.",
        "Avoid morning walks during peak pollution hours.",
        "Avoid outdoor walks.",
        "Use HEPA H13 air purifier indoors.",
        "Stay indoors in filtered air environments.",
        "Use HEPA air purifier continuously.",
        "Remain indoors at all times.",
        "Avoid exposure to polluted air completely.",
    ],
    "Pregnant Women": [
        "Avoid outdoor exposure unless necessary.",
        "Use KN95/N95 respirator if travelling.",
        "Avoid outdoor exposure entirely.",
        "Keep indoor air clean using HEPA filtration.",
        "Strictly avoid outdoor travel.",
        "Stay in indoor environments with HEPA filtration.",
    ],
    "Asthma / COPD Patients": [
        "Carry short-acting bronchodilator inhaler (e.g., Salbutamol).",
        "Avoid outdoor exposure for prolonged periods.",
        "Stay indoors as much as possible.",
        "Use prescribed inhaled corticosteroids regularly.",
        "Avoid outdoor exposure completely.",
        "Monitor symptoms like wheezing or breathlessness.",
        "High risk of respiratory distress.",
        "Remain indoors and keep rescue inhaler ready.",
    ],
    "Heart Disease Patients": [
        "Avoid physical exertion outdoors.",
        "Monitor chest discomfort and seek medical advice if symptoms appear.",
        "Avoid exertion and monitor symptoms such as chest tightness.",
        "Consult physician if experiencing breathing difficulty.",
    ],
    "Outdoor Workers": [
        "Use NIOSH-approved N95/N99 respirator continuously.",
        "Take 20-minute indoor breaks every hour.",
        "Limit work exposure time.",
        "Use industrial-grade N99 respirator with proper seal.",
    ],
}

# ── AQI category helper ───────────────────────────────────────────────────────
def _aqi_category(aqi):
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    if aqi <= 400:  return "Very Poor"
    return "Severe"

# ── Ground-truth label generator (domain-expert rules → binary vectors) ───────
def _labels_for_group(group, aqi):
    """
    Returns a binary list: 1 = advice item applies at this AQI, 0 = does not.
    This encodes the same domain knowledge as the old if/elif block, but as
    structured training labels that the ML model learns from.
    """
    items = GROUPS[group]
    labels = [0] * len(items)

    def activate(*phrases):
        for phrase in phrases:
            for i, item in enumerate(items):
                if phrase in item:
                    labels[i] = 1

    if group == "Children (0–12 yrs)":
        if aqi <= 50:
            activate("safe up to 2 hours", "Avoid playgrounds")
        elif aqi <= 100:
            activate("less than 1.5 hours", "traffic intersections")
        elif aqi <= 200:
            activate("less than 60 minutes", "football", "Prefer indoor")
        elif aqi <= 300:
            activate("Avoid outdoor sports", "indoors during morning")
        elif aqi <= 400:
            activate("Completely avoid", "school activities")
        else:
            activate("Do not allow", "HEPA purifier")

    elif group == "Teenagers / Athletes":
        if aqi <= 100:
            activate("Moderate sports")
        elif aqi <= 200:
            activate("Reduce high-intensity", "indoor gym")
        elif aqi <= 500:
            activate("Cancel outdoor", "Switch to indoor")

    elif group == "Healthy Adults":
        if aqi <= 50:
            activate("running or cycling is safe", "Hydrate")
        elif aqi <= 100:
            activate("jogging or cycling allowed")
        elif aqi <= 200:
            activate("less than 30 minutes", "N95 or KN95 respirator during commuting")
        elif aqi <= 300:
            activate("Avoid jogging", "NIOSH-approved N95 respirator during travel", "less than 20 minutes")
        elif aqi <= 400:
            activate("Avoid outdoor exercise completely", "tightly fitted N95")
        else:
            activate("Avoid all outdoor physical", "N95 or N99 respirator if leaving")

    elif group == "Elderly (65+)":
        if aqi <= 50:
            activate("Normal walking", "Carry regular medications")
        elif aqi <= 100:
            activate("less than 1 hour", "brisk walking")
        elif aqi <= 200:
            activate("less than 30 minutes", "morning walks")
        elif aqi <= 300:
            activate("Avoid outdoor walks", "HEPA H13")
        elif aqi <= 400:
            activate("Stay indoors in filtered", "HEPA air purifier continuously")
        else:
            activate("Remain indoors at all times", "Avoid exposure to polluted")

    elif group == "Pregnant Women":
        if aqi <= 200:
            pass  # No specific advice below Moderate
        elif aqi <= 300:
            activate("unless necessary", "KN95/N95 respirator if travelling")
        elif aqi <= 400:
            activate("entirely", "indoor air clean")
        else:
            activate("Strictly avoid", "indoor environments with HEPA")

    elif group == "Asthma / COPD Patients":
        if aqi <= 200:
            activate("bronchodilator inhaler", "prolonged periods")
        elif aqi <= 300:
            activate("Stay indoors as much", "inhaled corticosteroids")
        elif aqi <= 400:
            activate("Avoid outdoor exposure completely", "wheezing")
        else:
            activate("High risk", "rescue inhaler")

    elif group == "Heart Disease Patients":
        if aqi <= 300:
            pass  # No specific advice below Poor
        elif aqi <= 400:
            activate("Avoid physical exertion", "chest discomfort")
        else:
            activate("chest tightness", "breathing difficulty")

    elif group == "Outdoor Workers":
        if aqi <= 300:
            pass  # No specific advice below Poor
        elif aqi <= 400:
            activate("N95/N99 respirator continuously", "20-minute indoor breaks")
        else:
            activate("Limit work exposure", "industrial-grade N99")

    return labels

# ── Generate synthetic training dataset ──────────────────────────────────────
# Sample 2000 AQI values uniformly across the full range.
# For each AQI, generate ground-truth binary labels per group.
# The RF model then learns a smooth, continuous mapping.

np.random.seed(42)
TRAIN_AQI = np.concatenate([
    np.random.uniform(0,   50,  350),
    np.random.uniform(50,  100, 350),
    np.random.uniform(100, 200, 400),
    np.random.uniform(200, 300, 350),
    np.random.uniform(300, 400, 300),
    np.random.uniform(400, 500, 250),
])

# ── Train one MultiOutputClassifier per group ─────────────────────────────────
rec_models = {}

for group in GROUPS:
    X_rec = TRAIN_AQI.reshape(-1, 1)
    Y_rec = np.array([_labels_for_group(group, aqi) for aqi in TRAIN_AQI])

    # Skip groups that are never active (all-zero labels)
    if Y_rec.sum() == 0:
        continue

    clf = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",
        ),
        n_jobs=-1,
    )
    clf.fit(X_rec, Y_rec)
    rec_models[group] = clf

print("Recommendation models trained for:", list(rec_models.keys()))


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def predict_aqi(input_datetime):
    """Predict AQI for a given datetime string using the XGBoost model."""
    input_dt  = pd.to_datetime(input_datetime)
    last_rows = df_long[df_long["Datetime"] < input_dt].tail(72)

    if len(last_rows) < 72:
        raise ValueError(
            "Not enough historical data before the given datetime. "
            "Please choose a later date."
        )

    features = pd.DataFrame({
        "hour":           [input_dt.hour],
        "day":            [input_dt.day],
        "dayofweek":      [input_dt.dayofweek],
        "lag1":           [last_rows.iloc[-1]["AQI"]],
        "lag2":           [last_rows.iloc[-2]["AQI"]],
        "lag24":          [last_rows.iloc[-24]["AQI"]],
        "lag48":          [last_rows.iloc[-48]["AQI"]],
        "lag72":          [last_rows.iloc[-72]["AQI"]],
        "rolling_mean_3": [last_rows.iloc[-3:]["AQI"].mean()],
        "rolling_mean_6": [last_rows.iloc[-6:]["AQI"].mean()],
        "rolling_mean_12":[last_rows.iloc[-12:]["AQI"].mean()],
    })

    return float(aqi_model.predict(features)[0])


def suggest_precautions(aqi):
    """
    Returns (category: str, precautions: dict[str, list[str]])

    Uses trained MultiOutputClassifier (Random Forest) per population group
    to predict which advice items apply at the given AQI level.
    Falls back gracefully to rule-based labels if a group model is missing.
    """
    category   = _aqi_category(aqi)
    precautions = {}
    aqi_input  = np.array([[aqi]])

    for group, items in GROUPS.items():

        if group in rec_models:
            # ML prediction: binary vector over all advice items for this group
            pred_labels = rec_models[group].predict(aqi_input)[0]
        else:
            # Fallback: use rule-based labels directly
            pred_labels = _labels_for_group(group, aqi)

        active_advice = [items[i] for i, flag in enumerate(pred_labels) if flag == 1]

        # Only include group if at least one advice item is active
        if active_advice:
            precautions[group] = active_advice

    return category, precautions


def get_actual_vs_predicted():
    """Return (actual, predicted) arrays for the test set."""

    return y_test.values, preds
