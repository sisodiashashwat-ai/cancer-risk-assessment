import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Cancer Risk Assessment", layout="wide")
st.title("Cancer Risk Assessment System")

# 1. Model, Scaler (if exists), and CSV Load
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None
    df = pd.read_csv('cancer-risk-factors.csv')
    return model, scaler, df

model, scaler, df = load_assets()

# Text & Target columns drop -> Exact 17 Features bachenge
cols_to_drop = ['Patient_ID', 'Cancer_Type', 'Overall_Risk_Score', 'Risk_Level', 'Risk', 'Risk_Category']
X = df.drop(columns=cols_to_drop, errors='ignore')

st.write("Enter Patient Details")

# 2. Dynamic Input Generation for 17 Numeric Features
input_data = {}
cols = st.columns(2)

for i, col in enumerate(X.columns):
    mean_val = float(X[col].mean())
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    
    with cols[i % 2]:
        if pd.api.types.is_integer_dtype(X[col]):
            input_data[col] = st.number_input(
                f"{col}", 
                min_value=int(min_val), 
                max_value=int(max_val), 
                value=int(mean_val), 
                step=1
            )
        else:
            input_data[col] = st.number_input(
                f"{col}", 
                min_value=min_val, 
                max_value=max_val, 
                value=mean_val
            )

st.markdown("---")

# Label Mapping Fix (0 = High, 1 = Low, 2 = Medium)
label_map = {0: 'High', 1: 'Low', 2: 'Medium'}

# 3. Assessment & Prediction Logic
if st.button("Assess Cancer Risk Level", type="primary"):
    # Exact column order DataFrame
    input_df = pd.DataFrame([input_data])[X.columns]
    
    # Scale if scaler exists
    final_input = scaler.transform(input_df) if scaler is not None else input_df
        
    raw_pred = model.predict(final_input)[0]
    final_result = label_map.get(raw_pred, str(raw_pred))
    
    st.markdown("---")
    st.subheader("Assessment Result")
    
    if final_result == 'High':
        st.error(f"Predicted Risk Level: **{final_result}**")
    elif final_result == 'Medium':
        st.warning(f"Predicted Risk Level: **{final_result}**")
    else:
        st.success(f"Predicted Risk Level: **{final_result}**")

    # 4. Prediction Probabilities (Percentage Breakdown Feature)
    if hasattr(model, "predict_proba"):
        st.subheader("Prediction Probabilities")
        probs = model.predict_proba(final_input)[0]
        classes = getattr(model, "classes_", range(len(probs)))
        
        prob_list = []
        for c, p in zip(classes, probs):
            c_label = label_map.get(c, str(c))
            prob_list.append({'Risk Level': c_label, 'Probability': f"{p * 100:.2f}%"})
            
        prob_df = pd.DataFrame(prob_list)
        st.table(prob_df)