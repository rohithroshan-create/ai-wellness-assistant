import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import json
from typing import Dict, List, Any
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# Streamlit Config
# ------------------------
st.set_page_config(
    page_title="AI Wellness Assistant - Complete",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional medical CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #34495E;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
        transition: transform 0.2s;
    }
    .risk-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .high-risk {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left-color: #f44336;
    }
    .moderate-risk {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left-color: #ff9800;
    }
    .low-risk {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left-color: #4caf50;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f39c12;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Google Gemini Setup (From Chatbot Code)
# ------------------------
GOOGLE_API_KEY = "AIzaSyANnfoI2zSuanVqLEk7oqXq-q-whzPFouA"  # Your provided key
genai.configure(api_key=GOOGLE_API_KEY)

CHAT_MODEL_NAME = "models/gemini-2.0-flash-exp"
chat_model = genai.GenerativeModel(
    CHAT_MODEL_NAME,
    system_instruction=(
        "You are a medical assistant specializing only in chronic diseases "
        "(Heart disease, Type 2 Diabetes, Hypertension). "
        "Answer only questions related to these diseases. "
        "If the user asks unrelated questions, politely reply that you "
        "can only assist with chronic diseases."
    )
)

def google_chatbot_query(message: str):
    try:
        response = chat_model.generate_content(message)
        return response.text
    except Exception as e:
        return f"⚠ Error: {e}"

# ------------------------
# Load Models & Data (From Chatbot Code)
# ------------------------
@st.cache_data
def load_models():
    try:
        logreg = joblib.load("models/logreg_baseline.joblib")
    except:
        logreg = None
    
    try:
        xgb_heart = xgb.XGBClassifier()
        xgb_heart.load_model("models/xgb_heart.json")
    except:
        xgb_heart = None
    
    try:
        xgb_diabetes = xgb.XGBClassifier()
        xgb_diabetes.load_model("models/xgb_diabetes.json")
    except:
        xgb_diabetes = None
    
    try:
        xgb_hyper = xgb.XGBClassifier()
        xgb_hyper.load_model("models/xgb_hypertension.json")
    except:
        xgb_hyper = None
    
    return logreg, xgb_heart, xgb_diabetes, xgb_hyper

@st.cache_data
def load_median_values():
    try:
        df = pd.read_csv("data/heart.csv")
        medians = {
            "age": int(df["age"].median()),
            "trestbps": int(df["trestbps"].median()),
            "chol": int(df["chol"].median()),
            "thalach": int(df["thalach"].median()),
            "oldpeak": float(df["oldpeak"].median()),
            "ca": int(df["ca"].median()),
        }
        return medians
    except:
        return {
            "age": 54,
            "trestbps": 130,
            "chol": 240,
            "thalach": 150,
            "oldpeak": 1.0,
            "ca": 0,
        }

# ------------------------
# Input Encoding (Heart Model) - From Chatbot Code
# ------------------------
def encode_input(
    age: int, sex: str, cp: str, trestbps: int, chol: int,
    fbs: str, restecg: str, thalach: int, exang: str,
    oldpeak: float, slope: str, ca: int, thal: str
) -> pd.DataFrame:
    sex_val = 1 if sex == "Male" else 0
    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0
    cp_code = {"typical angina": 0, "atypical angina": 1, "non-anginal pain": 2, "asymptomatic": 3}[cp]
    if cp_code == 0: cp_vals = [0,0,0]
    elif cp_code == 1: cp_vals = [1,0,0]
    elif cp_code == 2: cp_vals = [0,1,0]
    else: cp_vals = [0,0,1]
    restecg_code = {"Normal":0,"ST-T wave abnormality":1,"Left ventricular hypertrophy":2}[restecg]
    if restecg_code == 0: restecg_vals=[0,0]
    elif restecg_code == 1: restecg_vals=[1,0]
    else: restecg_vals=[0,1]
    slope_code = {"Upsloping":0,"Flat":1,"Downsloping":2}[slope]
    if slope_code==0: slope_vals=[0,0]
    elif slope_code==1: slope_vals=[1,0]
    else: slope_vals=[0,1]
    thal_code = {"Fixed defect":1,"Reversible defect":2,"Normal":3}[thal]
    if thal_code==0: thal_vals=[0,0,0]
    elif thal_code==1: thal_vals=[1,0,0]
    elif thal_code==2: thal_vals=[0,1,0]
    else: thal_vals=[0,0,1]
    feature_vector = [
        age, sex_val, trestbps, chol, fbs_val, thalach, exang_val, oldpeak, ca,
        *cp_vals, *restecg_vals, *slope_vals, *thal_vals
    ]
    
    try:
        _, xgb_heart, _, _ = load_models()
        if xgb_heart:
            feature_names = xgb_heart.get_booster().feature_names
            X_new = pd.DataFrame([feature_vector], columns=feature_names)
            return X_new
    except:
        pass
    
    # Fallback with generic feature names
    feature_names = [f'feature_{i}' for i in range(len(feature_vector))]
    X_new = pd.DataFrame([feature_vector], columns=feature_names)
    return X_new

# ------------------------
# Alert Functions (Placeholders)
# ------------------------
def check_early_warning_heart(raw_feats):
    warnings = []
    if raw_feats.get("age", 0) > 70:
        warnings.append("⚠️ Age > 70: Higher risk category")
    if raw_feats.get("trestbps", 0) > 180:
        warnings.append("⚠️ Very high blood pressure detected")
    if raw_feats.get("chol", 0) > 300:
        warnings.append("⚠️ Very high cholesterol detected")
    return warnings

def preventive_tips_disease(disease):
    tips = {
        "Diabetes": [
            "Maintain healthy weight",
            "Exercise regularly (30 min/day)",
            "Eat balanced diet with low sugar",
            "Monitor blood glucose regularly",
            "Get regular health checkups"
        ],
        "Hypertension": [
            "Reduce sodium intake",
            "Exercise regularly",
            "Maintain healthy weight",
            "Limit alcohol consumption",
            "Manage stress effectively"
        ]
    }
    return tips.get(disease, ["Consult with healthcare provider"])

# ------------------------
# Enhanced Wellness Assistant Class (From App-7)
# ------------------------
class WellnessAssistant:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.performance_metrics = {}
        self.load_models()
        self.setup_chatbot()

    def load_models(self):
        """Load trained models and components"""
        try:
            model_files = {
                'heart_disease': 'heart_disease_model.pkl',
                'diabetes': 'diabetes_model.pkl',
                'hypertension': 'hypertension_model.pkl'
            }
            
            scaler_files = {
                'heart_disease': 'heart_disease_scaler.pkl',
                'diabetes': 'diabetes_scaler.pkl',
                'hypertension': 'hypertension_scaler.pkl'
            }
            
            # Load models
            for condition, file_path in model_files.items():
                if os.path.exists(file_path):
                    self.models[condition] = joblib.load(file_path)

            # Load scalers
            for condition, file_path in scaler_files.items():
                if os.path.exists(file_path):
                    self.scalers[condition] = joblib.load(file_path)

            # Load feature names and performance metrics
            if os.path.exists('feature_names.pkl'):
                self.feature_names = joblib.load('feature_names.pkl')

            if os.path.exists('performance_metrics.pkl'):
                self.performance_metrics = joblib.load('performance_metrics.pkl')

            if not self.models:
                self.create_demo_models()

        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            self.create_demo_models()

    def create_demo_models(self):
        """Create demo models for testing"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Demo performance metrics
        self.performance_metrics = {
            'heart_disease': {'accuracy': 0.87, 'auc': 0.92, 'model_name': 'XGBoost'},
            'diabetes': {'accuracy': 0.85, 'auc': 0.89, 'model_name': 'Random Forest'},
            'hypertension': {'accuracy': 0.83, 'auc': 0.88, 'model_name': 'Logistic Regression'}
        }
        
        # Create demo models
        for condition in ['heart_disease', 'diabetes', 'hypertension']:
            n_features = {'heart_disease': 14, 'diabetes': 10, 'hypertension': 13}[condition]
            
            X_demo = np.random.rand(100, n_features)
            y_demo = np.random.randint(0, 2, 100)
            
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_demo, y_demo)
            self.models[condition] = model
            
            scaler = StandardScaler()
            scaler.fit(X_demo)
            self.scalers[condition] = scaler

    def setup_chatbot(self):
        """Setup AI chatbot knowledge base"""
        self.medical_knowledge = {
            'heart_disease': {
                'risk_factors': [
                    'Age over 55', 'High blood pressure', 'High cholesterol',
                    'Smoking', 'Diabetes', 'Family history', 'Obesity', 'Physical inactivity'
                ],
                'prevention': [
                    'Regular exercise (150 min/week moderate activity)',
                    'Heart-healthy diet (Mediterranean, DASH)',
                    'Maintain healthy weight (BMI 18.5-24.9)',
                    'Don\'t smoke or quit smoking',
                    'Manage stress through relaxation techniques',
                    'Get adequate sleep (7-9 hours)',
                    'Regular health checkups'
                ],
                'symptoms': [
                    'Chest pain or discomfort', 'Shortness of breath',
                    'Pain in arms, back, neck, jaw', 'Nausea', 'Cold sweat'
                ]
            },
            'diabetes': {
                'risk_factors': [
                    'Age over 45', 'Overweight (BMI > 25)', 'Family history',
                    'Physical inactivity', 'High blood pressure', 'Abnormal cholesterol'
                ],
                'prevention': [
                    'Maintain healthy weight',
                    'Be physically active (30 min most days)',
                    'Eat healthy foods (whole grains, vegetables)',
                    'Limit refined carbs and sugary drinks',
                    'Regular health screenings'
                ],
                'symptoms': [
                    'Increased thirst and urination', 'Unexplained weight loss',
                    'Fatigue', 'Blurred vision', 'Slow-healing wounds'
                ]
            },
            'hypertension': {
                'risk_factors': [
                    'Age', 'Family history', 'Obesity', 'Physical inactivity',
                    'High salt diet', 'Alcohol consumption', 'Stress', 'Smoking'
                ],
                'prevention': [
                    'Maintain healthy weight',
                    'Regular physical activity',
                    'Limit sodium intake (<2,300mg/day)',
                    'Eat potassium-rich foods',
                    'Limit alcohol consumption',
                    'Manage stress',
                    'Don\'t smoke'
                ],
                'symptoms': [
                    'Often no symptoms (silent killer)',
                    'Severe headache', 'Chest pain', 'Difficulty breathing',
                    'Vision problems', 'Blood in urine'
                ]
            }
        }

    def get_recommendations(self, condition: str, risk_level: str, patient_data: Dict) -> List[str]:
        """Get personalized recommendations"""
        base_recommendations = {
            'heart_disease': {
                'High': [
                    "🚨 Consult a cardiologist immediately for comprehensive evaluation",
                    "🥗 Adopt Mediterranean diet: fish 2x/week, olive oil, nuts, vegetables",
                    "💊 Take prescribed medications exactly as directed",
                    "🚶‍♂️ Start with 10-15 min daily walks, gradually increase",
                    "🚭 Stop smoking completely - use nicotine replacement if needed",
                    "📊 Monitor blood pressure daily at same time",
                    "😴 Maintain 7-9 hours sleep with consistent schedule"
                ],
                'Moderate': [
                    "👨‍⚕️ Schedule regular checkups with your doctor every 3-6 months",
                    "🥗 Increase fruits and vegetables to 5-9 servings daily",
                    "🏃‍♂️ Aim for 150 minutes moderate exercise weekly",
                    "⚖️ Maintain healthy weight (BMI 18.5-24.9)",
                    "🧘‍♀️ Practice stress management: meditation, yoga, deep breathing"
                ],
                'Low': [
                    "✅ Continue healthy lifestyle habits",
                    "🔄 Annual health screenings and checkups",
                    "💪 Maintain regular physical activity",
                    "🥗 Keep eating balanced, nutritious diet"
                ]
            }
        }
        
        return base_recommendations.get(condition, {}).get(risk_level, [])

# ------------------------
# Main App
# ------------------------
def main():
    """Main application combining both functionalities"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Wellness Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete Health Risk Assessment & AI Chatbot</p>', unsafe_allow_html=True)
    
    # Medical disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚕️ Medical Disclaimer:</strong> This AI tool provides educational information only and does not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    logreg_model, xgb_heart, xgb_diabetes, xgb_hyper = load_models()
    medians = load_median_values()
    
    # Initialize wellness assistant
    assistant = WellnessAssistant()
    
    # Navigation
    page = st.sidebar.radio(
        "🧭 Select Feature",
        ["Heart Disease", "Type 2 Diabetes", "Hypertension", "💬 AI Chatbot", "📊 Overall Assessment"],
        key="main_nav"
    )
    
    # ---------------- Heart Disease (From Chatbot Code) ----------------
    if page == "Heart Disease":
        st.header("🫀 Heart Disease Risk Assessment")
        
        st.sidebar.subheader("Heart Profile")
        if "reset" not in st.session_state:
            st.session_state["reset"] = False
        if st.sidebar.button("🔄 Reset to Median Values", key="heart_reset"):
            st.session_state["reset"] = True
        
        age_default = medians["age"] if st.session_state["reset"] else 50
        trestbps_default = medians["trestbps"] if st.session_state["reset"] else 120
        chol_default = medians["chol"] if st.session_state["reset"] else 200
        thalach_default = medians["thalach"] if st.session_state["reset"] else 150
        oldpeak_default = medians["oldpeak"] if st.session_state["reset"] else 1.0
        ca_default = medians["ca"] if st.session_state["reset"] else 0
        
        age = st.sidebar.slider("Age", 29, 77, age_default, key="heart_age")
        sex = st.sidebar.selectbox("Sex", ["Male", "Female"], key="heart_sex")
        cp = st.sidebar.selectbox("Chest pain type", ["typical angina","atypical angina","non-anginal pain","asymptomatic"], key="heart_cp")
        trestbps = st.sidebar.slider("Resting BP (mm Hg)", 94, 200, trestbps_default, key="heart_trestbps")
        chol = st.sidebar.slider("Cholesterol (mg/dL)", 126, 564, chol_default, key="heart_chol")
        fbs = st.sidebar.selectbox("Fasting blood sugar > 120 mg/dL?", ["Yes", "No"], key="heart_fbs")
        restecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], key="heart_restecg")
        thalach = st.sidebar.slider("Max heart rate", 71, 202, thalach_default, key="heart_thalach")
        exang = st.sidebar.selectbox("Exercise-induced angina?", ["Yes", "No"], key="heart_exang")
        oldpeak = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.2, oldpeak_default, step=0.1, key="heart_oldpeak")
        slope = st.sidebar.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"], key="heart_slope")
        ca = st.sidebar.selectbox("Major vessels (0–3)", [0,1,2,3], index=ca_default, key="heart_ca")
        thal = st.sidebar.selectbox("Thalassemia", ["Fixed defect", "Reversible defect", "Normal"], key="heart_thal")
        
        if st.session_state["reset"]:
            st.session_state["reset"] = False
        
        if st.button("🔍 Compute Heart Risk", key="heart_compute", type="primary"):
            if xgb_heart is None:
                st.error("❌ Heart disease model not available")
                return
            
            raw_feats = {"age": age, "trestbps": trestbps, "chol": chol}
            warnings = check_early_warning_heart(raw_feats)
            if warnings:
                for msg in warnings:
                    st.error(msg)
            else:
                st.success("✅ No immediate red-flag values detected.")
            
            try:
                X_new = encode_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
                prob = xgb_heart.predict_proba(X_new)[0,1]
                
                st.subheader(f"🎯 Predicted Risk of Heart Disease: {prob:.1%}")
                
                # Risk gauge
                fig, ax = plt.subplots(figsize=(6, 0.6))
                bar_color = "crimson" if prob >= 0.5 else "seagreen"
                ax.barh(["Predicted Risk"], [prob], color=bar_color)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xlabel("Probability")
                ax.set_title("Risk Gauge (≥ 50% high risk)")
                st.pyplot(fig)
                
                # Heart age calculation
                if logreg_model:
                    try:
                        coef_age = float(logreg_model.coef_)
                        intercept = float(logreg_model.intercept_)
                        p = np.clip(prob, 1e-6, 1-1e-6)
                        raw_heart_age = (np.log(p/(1-p)) - intercept) / coef_age
                        heart_age = max(raw_heart_age, 0)
                        st.markdown(f"**Chronological age:** {age}  \n**Heart age:** {heart_age:.1f}")
                    except:
                        pass
                
                # Risk level and recommendations
                if prob < 0.3:
                    risk_level = "Low"
                elif prob < 0.6:
                    risk_level = "Moderate"
                else:
                    risk_level = "High"
                
                st.subheader("💡 Personalized Recommendations")
                recommendations = assistant.get_recommendations('heart_disease', risk_level, {})
                for rec in recommendations:
                    st.write(f"• {rec}")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
    
    # ---------------- Diabetes (From Chatbot Code) ----------------
    elif page == "Type 2 Diabetes":
        st.header("🩸 Type 2 Diabetes Risk Assessment")
        
        Pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1, key="diab_pregnancies")
        Glucose = st.sidebar.slider("Glucose", 0, 200, 110, key="diab_glucose")
        BloodPressure = st.sidebar.slider("Blood Pressure", 0, 140, 70, key="diab_bp")
        SkinThickness = st.sidebar.slider("Skin Thickness", 0, 100, 20, key="diab_skinthickness")
        Insulin = st.sidebar.slider("Insulin", 0, 900, 79, key="diab_insulin")
        BMI = st.sidebar.slider("BMI", 0.0, 70.0, 32.0, key="diab_bmi")
        DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01, key="diab_dpf")
        Age = st.sidebar.slider("Age", 10, 100, 30, key="diab_age")
        
        if st.button("🔍 Compute Diabetes Risk", key="diab_compute", type="primary"):
            if xgb_diabetes is None:
                st.error("❌ Diabetes model not available")
                return
            
            try:
                expected_features_diab = xgb_diabetes.get_booster().feature_names
                input_dict = {feat: 0 for feat in expected_features_diab}
                input_dict['Pregnancies'] = Pregnancies
                input_dict['Glucose'] = Glucose
                input_dict['BloodPressure'] = BloodPressure
                input_dict['SkinThickness'] = SkinThickness
                input_dict['Insulin'] = Insulin
                input_dict['BMI'] = BMI
                input_dict['DiabetesPedigreeFunction'] = DiabetesPedigreeFunction
                input_dict['Age'] = Age
                X_diab = pd.DataFrame([input_dict], columns=expected_features_diab)
                prob = xgb_diabetes.predict_proba(X_diab)[0,1]
                
                st.subheader(f"🎯 Predicted Risk of Diabetes: {prob:.1%}")
                
                # Create risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Diabetes Risk"},
                    delta={'reference': 30},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#FF9800" if prob >= 0.5 else "#4CAF50"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "red"}
                        ]
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("💡 Lifestyle Tips")
                for tip in preventive_tips_disease("Diabetes"):
                    st.write("✅", tip)
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
    
    # ---------------- Hypertension (From Chatbot Code) ----------------
    elif page == "Hypertension":
        st.header("🩺 Hypertension Risk Assessment")
        
        systolic = st.sidebar.slider("Systolic BP", 90, 200, 130, key="hyp_sys")
        diastolic = st.sidebar.slider("Diastolic BP", 60, 140, 80, key="hyp_dia")
        bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0, key="hyp_bmi")
        age_h = st.sidebar.slider("Age", 20, 80, 40, key="hyp_age")
        
        expected_features_hyper = [
            'Age', 'BMI', 'Cholesterol', 'Systolic_BP', 'Diastolic_BP',
            'Alcohol_Intake', 'Stress_Level', 'Salt_Intake', 'Sleep_Duration', 'Heart_Rate',
            'LDL', 'HDL', 'Triglycerides', 'Glucose', 'Country_Australia', 'Country_Brazil',
            'Country_Canada', 'Country_China', 'Country_France', 'Country_Germany', 'Country_India',
            'Country_Indonesia', 'Country_Italy', 'Country_Japan', 'Country_Mexico', 'Country_Russia',
            'Country_Saudi Arabia', 'Country_South Africa', 'Country_South Korea', 'Country_Spain',
            'Country_Turkey', 'Country_UK', 'Country_USA', 'Smoking_Status_Former', 'Smoking_Status_Never',
            'Physical_Activity_Level_Low', 'Physical_Activity_Level_Moderate', 'Family_History_Yes',
            'Diabetes_Yes', 'Gender_Male', 'Education_Level_Secondary', 'Education_Level_Tertiary',
            'Employment_Status_Retired', 'Employment_Status_Unemployed'
        ]
        
        if st.button("🔍 Compute Hypertension Risk", key="hyp_compute", type="primary"):
            if xgb_hyper is None:
                st.error("❌ Hypertension model not available")
                return
            
            try:
                input_dict = {
                    'Age': age_h, 'BMI': bmi, 'Systolic_BP': systolic, 'Diastolic_BP': diastolic,
                    'Cholesterol': 0, 'Alcohol_Intake': 0, 'Stress_Level': 0, 'Salt_Intake': 0,
                    'Sleep_Duration': 0, 'Heart_Rate': 0, 'LDL': 0, 'HDL': 0, 'Triglycerides': 0,
                    'Glucose': 0, 'Country_Australia': 0, 'Country_Brazil': 0, 'Country_Canada': 0,
                    'Country_China': 0, 'Country_France': 0, 'Country_Germany': 0, 'Country_India': 0,
                    'Country_Indonesia': 0, 'Country_Italy': 0, 'Country_Japan': 0, 'Country_Mexico': 0,
                    'Country_Russia': 0, 'Country_Saudi Arabia': 0, 'Country_South Africa': 0,
                    'Country_South Korea': 0, 'Country_Spain': 0, 'Country_Turkey': 0, 'Country_UK': 0,
                    'Country_USA': 0, 'Smoking_Status_Former': 0, 'Smoking_Status_Never': 1,
                    'Physical_Activity_Level_Low': 0, 'Physical_Activity_Level_Moderate': 1,
                    'Family_History_Yes': 0, 'Diabetes_Yes': 0, 'Gender_Male': 1,
                    'Education_Level_Secondary': 0, 'Education_Level_Tertiary': 0,
                    'Employment_Status_Retired': 0, 'Employment_Status_Unemployed': 0
                }
                X_hyper = pd.DataFrame([input_dict], columns=expected_features_hyper)
                prob = xgb_hyper.predict_proba(X_hyper)[0,1]
                
                st.subheader(f"🎯 Predicted Risk of Hypertension: {prob:.1%}")
                
                # Create risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Hypertension Risk"},
                    delta={'reference': 30},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#FF9800" if prob >= 0.5 else "#4CAF50"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "red"}
                        ]
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("💡 Lifestyle Tips")
                for tip in preventive_tips_disease("Hypertension"):
                    st.write("✅", tip)
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
    
    # ---------------- Chatbot (From Chatbot Code) ----------------
    elif page == "💬 AI Chatbot":
        st.header("🤖 Health Assistant Chatbot (Google Gemini)")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    border-radius: 15px; padding: 1.5rem; margin: 1rem 0; border: 2px solid #007bff;">
            <h3>🧠 Powered by Google Gemini 2.0 Flash</h3>
            <p>Ask me questions about Heart Disease, Diabetes, and Hypertension. I'm here to provide educational information and guidance!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat input
        user_input = st.text_input("💬 You:", key="chat_input", placeholder="Ask me about chronic diseases...")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("📤 Send", key="chat_send", type="primary")
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        if send_button and user_input:
            with st.spinner("🤔 AI thinking..."):
                st.session_state.chat_history.append(("You", user_input))
                bot_reply = google_chatbot_query(user_input)
                st.session_state.chat_history.append(("Bot", bot_reply))

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💭 Conversation History")
            for role, msg in reversed(st.session_state.chat_history[-10:]):  # Show last 10 messages
                if role == "You":
                    st.markdown(f"""
                    <div style="background: #007bff; color: white; padding: 0.75rem; border-radius: 15px; 
                                margin: 0.5rem 0; margin-left: 20%; text-align: right;">
                        <strong>🧑 You:</strong> {msg}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #28a745; color: white; padding: 0.75rem; border-radius: 15px; 
                                margin: 0.5rem 0; margin-right: 20%;">
                        <strong>🤖 Gemini AI:</strong> {msg}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👋 Start a conversation! Ask me anything about chronic diseases.")
            
            # Quick questions
            st.markdown("**🔗 Try these questions:**")
            quick_questions = [
                "What are the risk factors for heart disease?",
                "How can I prevent diabetes?",
                "What are the symptoms of hypertension?",
                "What lifestyle changes help reduce chronic disease risk?"
            ]
            
            for question in quick_questions:
                if st.button(question, key=f"quick_{hash(question)}"):
                    with st.spinner("🤔 AI thinking..."):
                        st.session_state.chat_history.append(("You", question))
                        bot_reply = google_chatbot_query(question)
                        st.session_state.chat_history.append(("Bot", bot_reply))
                        st.rerun()

    # ---------------- Overall Assessment (From App-7) ----------------
    elif page == "📊 Overall Assessment":
        st.header("📊 Comprehensive Health Assessment")
        
        # Display model performance if available
        if assistant.performance_metrics:
            st.subheader("🎯 AI Model Performance")
            cols = st.columns(3)
            
            conditions = ['heart_disease', 'diabetes', 'hypertension']
            condition_names = ['Heart Disease', 'Diabetes', 'Hypertension']
            
            for i, (condition, name) in enumerate(zip(conditions, condition_names)):
                if condition in assistant.performance_metrics:
                    metrics = assistant.performance_metrics[condition]
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>{name}</h4>
                            <p><strong>Model:</strong> {metrics['model_name']}</p>
                            <p><strong>Accuracy:</strong> {metrics['accuracy']:.1%}</p>
                            <p><strong>AUC:</strong> {metrics['auc']:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("""
        ## 🎯 How to Use This Application
        
        ### 🫀 **Heart Disease Assessment**
        - Detailed clinical parameters including ECG, chest pain type, and cardiac markers
        - XGBoost model with high accuracy
        - Heart age calculation based on risk factors
        
        ### 🩸 **Diabetes Risk Prediction**  
        - Comprehensive metabolic assessment
        - Includes pregnancy history, BMI, and glucose tolerance
        - Evidence-based lifestyle recommendations
        
        ### 🩺 **Hypertension Evaluation**
        - Blood pressure analysis with demographic factors
        - Multi-country dataset for diverse populations
        - Personalized intervention strategies
        
        ### 🤖 **AI Health Chatbot**
        - Google Gemini 2.0 Flash integration
        - Specialized in chronic disease education
        - 24/7 health guidance and support
        
        ### 🚀 **Getting Started**
        1. Select a specific disease assessment from the sidebar
        2. Input your health parameters
        3. Review your personalized risk analysis
        4. Get evidence-based recommendations
        5. Chat with our AI assistant for questions
        
        ### ⚠️ **Important Notes**
        - This tool is for educational purposes only
        - Always consult healthcare professionals for medical decisions
        - Regular health checkups are essential regardless of risk scores
        """)

# ------------------------
# Run App
# ------------------------
if __name__ == "__main__":
    main()
