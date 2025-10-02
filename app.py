# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_curve, accuracy_score
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Explorer",
    page_icon="🩺",
    layout="wide",
)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(path: str = "diabetes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standard column names often used in Pima dataset
    expected = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df[expected].copy()

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -----------------------------
# Sidebar: patient input
# -----------------------------
st.sidebar.header("Patient Data")
def user_report() -> pd.DataFrame:
    pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 2)
    glucose = st.sidebar.slider('Glucose (mg/dL)', 0, 220, 120)
    bp = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 130, 72)
    skinthickness = st.sidebar.slider('Skin Thickness (mm)', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin (µU/mL)', 0, 900, 79)
    bmi = st.sidebar.slider('BMI (kg/m²)', 0.0, 70.0, 27.3)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.47)
    age = st.sidebar.slider('Age (years)', 18, 90, 33)
    return pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])

user_data = user_report()

# -----------------------------
# Split + features
# -----------------------------
X = df.drop(columns=["Outcome"])
y = df["Outcome"].astype(int)

num_features = X.columns.tolist()

# -----------------------------
# Model building utility
# -----------------------------
def build_model(base_model="rf", calibrate=True, random_state=42):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_features),
        ],
        remainder="drop"
    )

    if base_model == "lr":
        base = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        )
    else:
        base = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )

    pipe = Pipeline([("pre", pre), ("clf", base)])

    if calibrate:
        # Calibrate probabilities via CV for meaningful risk %
        calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
        return calibrated
    return pipe

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=7
)

# -----------------------------
# Train (two candidates + pick best by ROC-AUC)
# -----------------------------
@st.cache_resource
def train_and_select():
    candidates = {
        "RandomForest": build_model("rf", calibrate=True),
        "LogisticRegression": build_model("lr", calibrate=True)
    }
    scores = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        scores[name] = roc_auc_score(y_test, proba)
        fitted[name] = model
    best_name = max(scores, key=scores.get)
    return best_name, fitted[best_name], scores

best_name, model, model_scores = train_and_select()

# -----------------------------
# App layout
# -----------------------------
st.title("🩺 Diabetes Risk Explorer")
st.caption("Educational demo only — not a medical diagnosis.")

tab_explore, tab_model, tab_explain, tab_batch, tab_about = st.tabs(
    ["Explore", "Model", "Explain", "Batch Scoring", "About"]
)

# -----------------------------
# Explore Tab
# -----------------------------
with tab_explore:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Training Data Overview")
        st.dataframe(df.describe().T, use_container_width=True)

        st.markdown("**Class Balance**")
        pos = int(y.sum())
        neg = int((1 - y).sum())
        st.write(f"Positive (Outcome=1): **{pos}**  |  Negative (Outcome=0): **{neg}**")

        st.markdown("**Feature Distributions (by Outcome)**")
        feature = st.selectbox(
            "Choose a feature (Explore)",
            num_features,
            index=num_features.index("Glucose"),
            key="explore_feature"       
        )
        fig, ax = plt.subplots()
        ax.hist(df[df["Outcome"]==0][feature], bins=30, alpha=0.6, label="Outcome=0")
        ax.hist(df[df["Outcome"]==1][feature], bins=30, alpha=0.6, label="Outcome=1")
        ax.set_title(f"{feature} distribution")
        ax.set_xlabel(feature); ax.set_ylabel("Count"); ax.legend()
        st.pyplot(fig, use_container_width=True)

    with right:
        st.subheader("Patient Input")
        st.dataframe(user_data, use_container_width=True)

        # Quick scatter lens
        st.markdown("**You vs Population**")
        lens_feature = st.selectbox("Y-axis feature", ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction"])
        fig2, ax2 = plt.subplots()
        ax2.scatter(df["Age"], df[lens_feature], c=df["Outcome"], cmap="coolwarm", alpha=0.4, label="Dataset")
        ax2.scatter(user_data["Age"], user_data[lens_feature], s=160, edgecolor="black", label="You")
        ax2.set_xlabel("Age"); ax2.set_ylabel(lens_feature)
        ax2.set_title(f"Age vs {lens_feature}")
        ax2.legend()
        st.pyplot(fig2, use_container_width=True)

# -----------------------------
# Model Tab
# -----------------------------
with tab_model:
    st.subheader(f"Best Model Selected: **{best_name}**")
    cols = st.columns(3)
    with cols[0]:
        st.metric("ROC-AUC (holdout)", f"{roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f}")
    with cols[1]:
        st.metric("PR-AUC (holdout)", f"{average_precision_score(y_test, model.predict_proba(X_test)[:,1]):.3f}")
    with cols[2]:
        st.metric("Accuracy (holdout)", f"{accuracy_score(y_test, model.predict(X_test)):.3f}")

    st.markdown("### Threshold & Risk")
    default_thresh = 0.5
    threshold = st.slider("Decision threshold (positive if risk ≥ threshold)", 0.05, 0.95, default_thresh, 0.01)

    # Patient prediction
    patient_proba = float(model.predict_proba(user_data)[:, 1])
    patient_pred = int(patient_proba >= threshold)
    risk_pct = f"{patient_proba*100:.1f}%"
    color = "🟢" if patient_pred==0 else "🔴"
    st.write(f"**Patient risk (calibrated)**: {color} **{risk_pct}**  → Predicted class: **{patient_pred}** at threshold {threshold:.2f}")

    # Confusion matrix at chosen threshold
    test_proba = model.predict_proba(X_test)[:, 1]
    y_pred_thr = (test_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_thr)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    f1 = f1_score(y_test, y_pred_thr)

    cm_fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha='center', va='center')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
    ax.set_title("Confusion Matrix")
    st.pyplot(cm_fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("F1 (thresholded)", f"{f1:.3f}")
    m2.metric("Sensitivity (Recall+)", f"{sensitivity:.3f}")
    m3.metric("Specificity (Recall-)", f"{specificity:.3f}")

    # Curves
    with st.expander("Show ROC & Precision–Recall Curves"):
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        prec, rec, _ = precision_recall_curve(y_test, test_proba)

        fig_roc, axr = plt.subplots()
        axr.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(y_test, test_proba):.3f}")
        axr.plot([0,1],[0,1],'--', alpha=0.6)
        axr.set_xlabel("FPR"); axr.set_ylabel("TPR"); axr.legend(); axr.set_title("ROC")
        st.pyplot(fig_roc, use_container_width=True)

        fig_pr, axp = plt.subplots()
        axp.plot(rec, prec, label=f"PR-AUC={average_precision_score(y_test, test_proba):.3f}")
        axp.set_xlabel("Recall"); axp.set_ylabel("Precision"); axp.legend(); axp.set_title("Precision–Recall")
        st.pyplot(fig_pr, use_container_width=True)

# -----------------------------
# Explain Tab
# -----------------------------
with tab_explain:
    st.subheader("What drives the prediction?")
    st.caption("Permutation importance is model-agnostic and highlights features that most affect predictions.")

    # For permutation importance we need a fitted estimator with predict_proba
    # We compute importance on a sample for speed
    sample_idx = np.random.RandomState(42).choice(len(X_test), size=min(200, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]
    y_sample = y_test.iloc[sample_idx]

    try:
        result = permutation_importance(
            model, X_sample, y_sample, n_repeats=10, random_state=42, scoring="roc_auc"
        )
        imp_df = pd.DataFrame({
            "feature": X_sample.columns,
            "importance": result.importances_mean,
            "std": result.importances_std
        }).sort_values("importance", ascending=False)

        st.dataframe(imp_df, use_container_width=True)

        fig_imp, ax = plt.subplots()
        ax.barh(imp_df["feature"], imp_df["importance"])
        ax.invert_yaxis()
        ax.set_title("Permutation Importance (ROC-AUC change)")
        st.pyplot(fig_imp, use_container_width=True)
    except Exception as e:
        st.info(f"Permutation importance unavailable: {e}")

    st.markdown("### Partial Dependence (1D)")
    pdp_feature = st.selectbox(
        "Choose a feature (PDP)",
        num_features,
        index=num_features.index("Glucose"),
        key="pdp_feature"              
    )
    try:
        fig_pdp, axpdp = plt.subplots()
        # PartialDependenceDisplay needs the pipeline to implement predict_proba on raw features
        PartialDependenceDisplay.from_estimator(model, X, [pdp_feature], ax=axpdp, kind="average")
        st.pyplot(fig_pdp, use_container_width=True)
    except Exception as e:
        st.info(f"PDP not available for this feature/estimator: {e}")

# -----------------------------
# Batch Scoring Tab
# -----------------------------
with tab_batch:
    st.subheader("Score a CSV of Patients")
    st.caption("Upload a CSV with the same feature columns (no Outcome column needed).")
    up = st.file_uploader("Upload CSV", type=["csv"])
    thr = st.slider("Decision threshold for batch", 0.05, 0.95, 0.5, 0.01, key="batch_thr")
    if up is not None:
        try:
            new_df = pd.read_csv(up)
            # Keep only known features, ignore extras
            used = [c for c in num_features if c in new_df.columns]
            missing = [c for c in num_features if c not in new_df.columns]
            if missing:
                st.warning(f"Missing columns filled with NaN → imputed: {missing}")
                for c in missing:
                    new_df[c] = np.nan
            new_df = new_df[num_features]
            proba = model.predict_proba(new_df)[:, 1]
            pred = (proba >= thr).astype(int)
            out = new_df.copy()
            out["risk_proba"] = proba
            out["prediction"] = pred

            st.dataframe(out.head(20), use_container_width=True)

            # Download
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download Scored CSV", data=csv, file_name="diabetes_scored.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not score file: {e}")

# -----------------------------
# About Tab
# -----------------------------
with tab_about:
    st.markdown("""
**Diabetes Risk Explorer** demonstrates a responsible ML workflow over the Pima-style diabetes dataset:

- Robust preprocessing (median impute, scale) inside a single **scikit-learn Pipeline**  
- **Model selection** between calibrated Random Forest and Logistic Regression using ROC-AUC  
- **Probability calibration** (isotonic) so risk % is interpretable  
- **Stratified** split + class weighting for imbalance  
- **Threshold tuning** to trade off sensitivity/specificity depending on use case  
- Lightweight **explainability** via permutation importance and partial dependence  
- Individual patient inputs and **batch scoring** with downloadable results  

> **Disclaimer:** This is an educational demo and not a medical device or diagnosis.
""")


