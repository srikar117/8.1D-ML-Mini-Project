# Sydney Housing Price Prediction and Decision Support System

SIT720 8.1 Distinction Task. Predicts Sydney house/unit sale prices for Mount Druitt, Campbelltown, and Parramatta using a Linear Regression model trained on 102 manually collected sold properties.

## Structure
- `data/` — cleaned dataset (`Sydney_Housing_Data.csv`) and modelling-ready version (`model_ready.csv`)
- `notebook/` — full analysis notebook (EDA, feature engineering, modelling, evaluation)
- `app/` — `app.py` (Streamlit)
- `report/` — final PDF report

## Setup
```bash
pip install streamlit scikit-learn pandas
```

## Run the app
```bash
cd app
streamlit run app.py
```

## Results
Linear Regression selected as final model (RMSE $160,149, R² 0.694), chosen over a marginally higher-scoring Random Forest due to greater stability across cross-validation folds.
