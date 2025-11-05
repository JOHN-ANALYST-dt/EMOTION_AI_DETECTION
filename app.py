import streamlit as st

# Example: Get user input
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Comprehensive Model Performance Dashboard")

# ====== ORIGINAL DATAFRAME ======
df = pd.DataFrame({
    'Exact-match Accuracy': [0.5178, 0.4003, 0.3887, None],
    'Macro F1': [0.1626, 0.3634, 0.3579, 0.38407],
    'Micro F1': [0.4424, 0.5209, 0.5145, 0.59304],
    'Macro Recall': [0.1208, 0.5229, 0.5551, None],
    'Micro Recall': [0.3142, 0.6848, 0.6987, None],
    'Hamming Loss': [0.0909, 0.1446, 0.1497, 0.08337]
},
index=['Naive Bayes', 'SVM', 'Logistic Regression', 'BERT'])

st.header("📁 Model Metrics Table")
st.dataframe(df.round(4))


# ====== RANKING TABLE ======
ranking_df = pd.DataFrame(index=df.index)

for metric in df.columns:
    if metric == "Hamming Loss":
        ranking_df[metric] = df[metric].rank(ascending=True)   # lower = better
    else:
        ranking_df[metric] = df[metric].rank(ascending=False)  # higher = better

ranking_df["Total Rank"] = ranking_df.sum(axis=1)
ranking_df["Average Rank"] = ranking_df.mean(axis=1)

# Sort top → bottom
ranking_df = ranking_df.sort_values(by="Total Rank")

st.header("🏆 Model Ranking Table")
st.dataframe(ranking_df.round(2))


# ====== BAR CHART ======
st.header("📈 Model Performance Chart")

fig, ax = plt.subplots()
df.plot(kind='bar', ax=ax)
ax.set_title("Comprehensive Model Comparison")
ax.set_xlabel("Models")
ax.set_ylabel("Score")
plt.xticks(rotation=45)

st.pyplot(fig)


# ====== BEST MODEL SUMMARY ======
st.header("🥇 Best Model per Metric")

best_results = {}

for metric in df.columns:
    if metric == "Hamming Loss":
        best_model = df[metric].idxmin()
        best_value = df[metric].min()
    else:
        best_model = df[metric].idxmax()
        best_value = df[metric].max()

    best_results[metric] = (best_model, round(best_value, 4))

best_df = pd.DataFrame(best_results, index=["Best Model", "Score"]).T
st.dataframe(best_df)


