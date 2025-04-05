import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Custom CSS
def add_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        h1 {
            color: #2F4F4F;
        }
        .stSidebar {
            background-color: #F0F2F6;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

# Load CSV file
def load_data(file):
    df = pd.read_csv(file)
    return df

# Clean missing values
def clean_data(df):
    missing_before = df.isnull().sum()
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna("Unknown", inplace=True)
    return df, missing_before

# Summary statistics
def show_statistics(df):
    st.markdown("## 📈 Summary Statistics")
    numerical_cols = df.select_dtypes(include=['number'])

    if not numerical_cols.empty:
        st.dataframe(numerical_cols.describe().style.background_gradient(cmap='Blues'))
        
        st.markdown("### 📊 Central Tendency")
        for col in numerical_cols.columns:
            with st.expander(f"🔍 {col}"):
                st.write(f"**Mean**: {numerical_cols[col].mean():.2f}")
                st.write(f"**Median**: {numerical_cols[col].median():.2f}")
                st.write(f"**Mode**: {numerical_cols[col].mode().values[0]}")
    else:
        st.info("No numerical columns found in the dataset.")

# EDA
def perform_eda(df, missing_before):
    st.markdown("## 🧾 Dataset Preview")
    st.dataframe(df.head())

    if missing_before.sum() > 0:
        st.markdown("### ❗ Missing Values Before Cleaning")
        st.write(missing_before[missing_before > 0])

    numerical_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    if len(numerical_columns) > 0:
        st.markdown("### 📊 Numerical Distribution")
        selected_num_col = st.selectbox("Select a numerical column", numerical_columns)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_num_col], bins=30, kde=True, ax=ax, color='skyblue')
        ax.set_title(f"Distribution of {selected_num_col}")
        st.pyplot(fig)

        if len(numerical_columns) > 1:
            st.markdown("### 🔥 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)

    if len(categorical_columns) > 0:
        st.markdown("### 🗂️ Categorical Column Distribution")
        selected_cat_col = st.selectbox("Select a categorical column", categorical_columns)
        fig, ax = plt.subplots()
        sns.countplot(y=selected_cat_col, data=df, order=df[selected_cat_col].value_counts().index[:10], palette='viridis', ax=ax)
        ax.set_title(f"Count Plot of {selected_cat_col}")
        st.pyplot(fig)

# Visual Analysis
def visual_analysis(df):
    st.markdown("## 🎨 Custom Visual Analysis")
    chart_type = st.selectbox("📌 Choose Chart Type", ["Scatter", "Line", "Bar", "Boxplot", "Pie Chart", "Pairplot"])

    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    if chart_type in ["Scatter", "Line", "Bar", "Boxplot"]:
        x_axis = st.selectbox("Select X-axis", df.columns)
        y_axis = st.selectbox("Select Y-axis", numerical_columns)

        fig, ax = plt.subplots()
        if chart_type == "Scatter":
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
        elif chart_type == "Line":
            sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
        elif chart_type == "Bar":
            sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
        elif chart_type == "Boxplot":
            sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax)

        ax.set_title(f"{chart_type} Plot")
        st.pyplot(fig)

    elif chart_type == "Pie Chart":
        if categorical_columns:
            pie_col = st.selectbox("Select a categorical column", categorical_columns)
            pie_data = df[pie_col].value_counts().head(10)
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.warning("No categorical columns available for pie chart.")

    elif chart_type == "Pairplot":
        if len(numerical_columns) > 1:
            st.info("Generating pairplot for numerical columns...")
            fig = sns.pairplot(df[numerical_columns])
            st.pyplot(fig)
        else:
            st.warning("Need at least 2 numerical columns for pairplot.")

# App Layout
def main():
    st.set_page_config(page_title="📊 CSV Analyzer Dashboard", layout="wide")
    add_custom_css()
    st.title("📁 CSV Analyzer Dashboard")

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png", width=100)
        st.markdown("## 📊 Navigation")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file:
            view = st.radio("View", ["Dataset Overview", "Summary Statistics", "Visual Analysis"])
        else:
            view = None

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df, missing_before = clean_data(df)

        if view == "Dataset Overview":
            perform_eda(df, missing_before)
        elif view == "Summary Statistics":
            show_statistics(df)
        elif view == "Visual Analysis":
            visual_analysis(df)

        st.sidebar.success("✅ Analysis complete")
    else:
        st.info("📤 Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()


#streamlit run 1.py

