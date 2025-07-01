import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Caching Functions ---
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_large_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

# --- Load Data & Model ---
df = load_data()
model = load_model()

# --- Use model's exact feature order ---
model_feature_order = ['NO2', 'SO2', 'PM2.5', 'PM10', 'CO', 'O3']
missing = [f for f in model_feature_order if f not in df.columns]
if missing:
    st.error(f"Missing features in dataset: {missing}")
    st.stop()

X = df[model_feature_order]
y_true = df["AQI"]

# --- Generate Predictions ---
y_pred = model.predict(X)

# --- Sidebar for Visualization Selection ---
st.sidebar.title("Visualization Selector")
plot_option = st.sidebar.selectbox(
    "Choose Visualization",
    [
        "Feature Importances",
        "Prediction vs Actual",
        "Residuals Plot",
        "Correlation Heatmap",
        "AQI Distribution",
        "Boxplot of PM2.5 by AQI Category"
    ]
)

# --- Dashboard Title ---
st.title("Air Quality Prediction Dashboard")

# --- Visualization Logic ---
if plot_option == "Feature Importances":
    st.subheader("Feature Importances")
    if len(model.feature_importances_) != len(model_feature_order):
        st.error("Mismatch between model importances and selected features.")
        st.write(f"Model importances: {len(model.feature_importances_)}")
        st.write(f"Selected features: {len(model_feature_order)}")
    else:
        importance_df = pd.DataFrame({
            "Feature": model_feature_order,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        st.pyplot(fig)

elif plot_option == "Prediction vs Actual":
    st.subheader("Prediction vs Actual AQI")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, ax=ax)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.set_title("Prediction vs Actual AQI")
    st.pyplot(fig)

elif plot_option == "Residuals Plot":
    st.subheader("Residuals Plot (Actual - Predicted)")
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Predicted AQI")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

elif plot_option == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif plot_option == "AQI Distribution":
    st.subheader("AQI Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["AQI"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("AQI")
    st.pyplot(fig)

elif plot_option == "Boxplot of PM2.5 by AQI Category":
    st.subheader("Boxplot of PM2.5 by AQI Category")
    if "AQI_Category" not in df.columns:
        df["AQI_Category"] = pd.cut(
            df["AQI"],
            bins=[0, 50, 100, 150, 200, 300, 500],
            labels=["Good", "Moderate", "Unhealthy for Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"]
        )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="AQI_Category", y="PM2.5", data=df, ax=ax)
    st.pyplot(fig)
