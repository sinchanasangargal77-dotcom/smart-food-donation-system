import streamlit as st
import pandas as pd
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# NGO DATA
# -----------------------------

ngo_data = {
    "ngo_name": [
        "Hope Orphanage",
        "Care Old Age Home",
        "Bright Future Daycare",
        "Helping Hands Shelter"
    ],
    "location": [
        "Whitefield",
        "Marathahalli",
        "Indiranagar",
        "KR Puram"
    ],
    "latitude": [
        12.9698,
        12.9591,
        12.9784,
        13.0080
    ],
    "longitude": [
        77.7500,
        77.6974,
        77.6408,
        77.6950
    ],
    "capacity": [
        50,
        30,
        40,
        60
    ],
    "type": [
        "Children",
        "Elderly",
        "Children",
        "Mixed"
    ]
}

ngo_df = pd.DataFrame(ngo_data)

# -----------------------------
# LOCATION COORDINATES
# -----------------------------

location_coordinates = {
    "Whitefield": (12.9698, 77.7500),
    "Marathahalli": (12.9591, 77.6974),
    "Indiranagar": (12.9784, 77.6408),
    "KR Puram": (13.0080, 77.6950)
}

# -----------------------------
# ML TRAINING DATA
# -----------------------------

food_training_data = {
    "quantity_kg": [20, 5, 15, 8, 25, 10, 30, 12, 18, 6],
    "hours_old": [2, 1, 5, 3, 6, 2, 7, 4, 5, 1],
    "temperature": [30, 25, 35, 28, 36, 29, 38, 32, 34, 26],
    "priority": [
        "High",
        "Low",
        "High",
        "Medium",
        "High",
        "Medium",
        "High",
        "Medium",
        "High",
        "Low"
    ]
}

food_df = pd.DataFrame(food_training_data)

X = food_df[["quantity_kg", "hours_old", "temperature"]]
y = food_df["priority"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("AI-Based Smart Food Waste Optimization System")

st.header("Restaurant Food Donation Form")

restaurant_name = st.text_input("Enter Restaurant Name")

food_item = st.text_input("Enter Food Item")

quantity = st.number_input(
    "Enter Quantity in KG",
    min_value=1
)

location = st.selectbox(
    "Select Location",
    list(location_coordinates.keys())
)

hours_old = st.number_input(
    "Hours Since Food Prepared",
    min_value=1
)

temperature = st.number_input(
    "Current Temperature"
)

# -----------------------------
# BUTTON
# -----------------------------

if st.button("Analyze Donation"):

    food_input = [[quantity, hours_old, temperature]]

    priority_prediction = model.predict(food_input)

    latitude, longitude = location_coordinates[location]

    restaurant_coords = (latitude, longitude)

    distances = []

    for _, row in ngo_df.iterrows():

        ngo_coords = (
            row["latitude"],
            row["longitude"]
        )

        distance = geodesic(
            restaurant_coords,
            ngo_coords
        ).km

        distances.append(distance)

    ngo_df["distance_km"] = distances

    nearest_ngo = ngo_df.loc[
        ngo_df["distance_km"].idxmin()
    ]

    st.success("Analysis Completed Successfully!")

    st.subheader("AI Food Analysis")

    st.write(
        "Predicted Priority:",
        priority_prediction[0]
    )

    st.subheader("Nearest NGO Recommendation")

    st.write(
        "NGO Name:",
        nearest_ngo["ngo_name"]
    )

    st.write(
        "Location:",
        nearest_ngo["location"]
    )

    st.write(
        "Distance:",
        round(nearest_ngo["distance_km"], 2),
        "KM"
    )

    st.write(
        "Type:",
        nearest_ngo["type"]
    )
