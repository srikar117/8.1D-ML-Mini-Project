"""
Sydney Housing Price Prediction and Decision Support System
8.1 Distinction Task - Part 6 deployment

"""

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏠")


@st.cache_resource
def train_model():
    df = pd.read_csv("model_ready.csv")
    X = pd.get_dummies(df[["Suburb"]], drop_first=True)
    X = pd.concat([
        df[["Bedrooms", "Bathrooms", "Parking_Spaces", "Land_Size_sqm",
            "Has_Land_Size", "Dwelling_Group"]],
        X
    ], axis=1)
    y = df["Sale_Price_AUD"]
    model = LinearRegression().fit(X, y)
    return model, list(X.columns)


model, feature_order = train_model()

st.title("Sydney Housing Price Predictor")
st.caption(
    "Decision-support tool for Mount Druitt, Campbelltown and Parramatta, "
    "trained on 102 manually collected sold properties. This gives an "
    "estimate to support a decision, it is not a formal valuation."
)

with st.form("property_form"):
    col1, col2 = st.columns(2)

    with col1:
        suburb = st.selectbox("Suburb", ["Mount Druitt", "Campbelltown", "Parramatta"])
        property_type = st.selectbox(
            "Property type",
            ["House", "Villa", "Semi-detached", "Terrace", "Apartment / Unit / Flat"],
        )
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1)

    with col2:
        parking = st.number_input("Parking spaces", min_value=0, max_value=10, value=1)
        land_size = st.number_input(
            "Land size (sqm) - enter 0 if not applicable or unknown",
            min_value=0, max_value=2000, value=500,
        )

    submitted = st.form_submit_button("Predict sale price")

if submitted:
    dwelling_group = 1 if property_type in ["House", "Villa", "Semi-detached", "Terrace"] else 0
    has_land_size = 1 if land_size > 0 else 0

    row = {
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Parking_Spaces": parking,
        "Land_Size_sqm": land_size,
        "Has_Land_Size": has_land_size,
        "Dwelling_Group": dwelling_group,
        "Suburb_Mount Druitt": 1 if suburb == "Mount Druitt" else 0,
        "Suburb_Parramatta": 1 if suburb == "Parramatta" else 0,
    }
    X_input = pd.DataFrame([row])[feature_order]
    prediction = model.predict(X_input)[0]

    st.success(f"Estimated sale price: ${prediction:,.0f}")

    if dwelling_group == 0 and land_size > 0:
        st.info(
            "Note: units typically do not have an individually owned land "
            "allocation. If this land size figure is a shared strata lot "
            "size rather than exclusive land, the estimate may be less reliable."
        )

    st.caption(
        "Model: Linear Regression trained via 5-fold cross-validation "
    )
