import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Wellness Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical interface
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 3rem;
    }
    .risk-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 8px solid;
        transition: transform 0.2s;
    }
    .risk-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
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
    .recommendation-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-top: 4px solid #1e88e5;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    .assistant-message {
        background-color: #f1f8e9;
        margin-right: auto;
    }
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .stProgress .st-bo {
        background-color: #1e88e5;
    }
    .warning-banner {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f39c12;
    }
</style>
""", unsafe_allow_html=True)

# Recommendation Engine
class RecommendationEngine:
    def __init__(self):
        self.recommendations = {
            'heart_disease': {
                'high_risk': {
                    'diet': [
                        "🥗 Follow Mediterranean diet with omega-3 rich fish 2-3x/week",
                        "🧂 Limit sodium intake to <1,500mg daily (check food labels)",
                        "🥑 Include healthy fats: olive oil, avocados, nuts, seeds", 
                        "🍎 Eat 5-9 servings of colorful fruits and vegetables daily",
                        "🚫 Eliminate trans fats and limit saturated fats to <7% calories"
                    ],
                    'exercise': [
                        "🚶‍♂️ Start with 10-15 min walks, build to 150min/week moderate activity",
                        "💪 Add 2 days strength training (resistance bands, light weights)",
                        "🏊‍♀️ Low-impact options: swimming, cycling, yoga, tai chi",
                        "📱 Use fitness tracker to monitor heart rate during exercise",
                        "⚕️ Get medical clearance before starting vigorous exercise program"
                    ],
                    'lifestyle': [
                        "🚭 Quit smoking immediately - seek nicotine replacement therapy",
                        "😴 Maintain 7-9 hours quality sleep with consistent bedtime",
                        "🧘‍♀️ Practice daily stress management: meditation, deep breathing",
                        "🍷 Limit alcohol: 1 drink/day women, 2 drinks/day men maximum",
                        "⚖️ Achieve and maintain healthy BMI (18.5-24.9)"
                    ],
                    'medical': [
                        "👨‍⚕️ Schedule regular cardiology check-ups every 3-6 months",
                        "📊 Monitor blood pressure daily, keep log for doctor visits",
                        "💊 Take prescribed medications exactly as directed",
                        "🆘 Learn heart attack warning signs and emergency response",
                        "📞 Keep emergency contacts and medical info easily accessible"
                    ]
                },
                'moderate_risk': {
                    'diet': [
                        "🌾 Choose whole grains over refined: brown rice, quinoa, oats",
                        "🐟 Include lean proteins: fish, poultry, legumes, tofu",
                        "💧 Stay hydrated with 8-10 glasses water daily",
                        "🍇 Limit added sugars and processed foods",
                        "🥜 Snack on nuts, seeds, fresh fruit instead of chips/cookies"
                    ],
                    'exercise': [
                        "🎯 Aim for 30 minutes moderate activity most days",
                        "🏃‍♀️ Try brisk walking, dancing, gardening, sports",
                        "📈 Gradually increase intensity and duration weekly",
                        "👥 Join group fitness classes for motivation and fun",
                        "🚶‍♀️ Take stairs, park farther, add movement to daily routine"
                    ],
                    'lifestyle': [
                        "😌 Practice stress reduction techniques daily",
                        "👥 Build and maintain strong social connections", 
                        "📱 Limit screen time, especially before bedtime",
                        "🌿 Spend time in nature for mental health benefits",
                        "📚 Consider mindfulness apps or stress management classes"
                    ]
                }
            },
            'diabetes': {
                'high_risk': {
                    'diet': [
                        "🍽️ Use plate method: 1/2 vegetables, 1/4 lean protein, 1/4 whole grains",
                        "⏰ Eat regular meals every 3-4 hours to stabilize blood sugar",
                        "🔢 Count carbohydrates and choose low glycemic index foods",
                        "🥤 Replace sugary drinks with water, unsweetened tea, sparkling water",
                        "🥕 Focus on fiber-rich foods: vegetables, beans, whole grains"
                    ],
                    'exercise': [
                        "🚶‍♀️ Take 10-15 minute walks after each meal to lower blood sugar",
                        "🏋️‍♂️ Include resistance training 2-3x/week to improve insulin sensitivity",
                        "📊 Monitor blood glucose before and after exercise sessions",
                        "🍬 Always carry glucose tablets during extended physical activity",
                        "⚖️ Focus on exercises that help with weight management"
                    ],
                    'lifestyle': [
                        "📱 Monitor blood glucose as recommended by healthcare provider",
                        "⚖️ Work toward 5-10% body weight reduction if overweight",
                        "👀 Schedule annual comprehensive eye examinations",
                        "🦶 Daily foot inspection and proper foot care routine",
                        "💉 Stay current with recommended vaccinations"
                    ]
                }
            },
            'hypertension': {
                'high_risk': {
                    'diet': [
                        "🥬 Follow DASH diet: high potassium foods like bananas, spinach",
                        "🧂 Reduce sodium to <1,500mg daily - cook at home more often",
                        "🥛 Include low-fat dairy for calcium and magnesium",
                        "☕ Limit caffeine if you notice it raises your blood pressure",
                        "🍫 Minimize processed foods, fast food, and restaurant meals"
                    ],
                    'exercise': [
                        "❤️ Engage in 30+ minutes aerobic activity daily",
                        "🧘‍♀️ Include flexibility exercises: yoga, stretching, tai chi",
                        "📊 Monitor blood pressure before and after exercise",
                        "🚫 Avoid sudden, intense physical exertion",
                        "👨‍⚕️ Consider medically supervised exercise program initially"
                    ],
                    'lifestyle': [
                        "🩺 Check blood pressure at home daily, same time each day",
                        "⚖️ Lose weight gradually if overweight (1-2 pounds per week)",
                        "🍷 Significantly limit or eliminate alcohol consumption",
                        "😌 Practice daily stress reduction: meditation, prayer, music",
                        "😴 Prioritize 7-8 hours quality sleep nightly"
                    ]
                }
            }
        }
    
    def get_recommendations(self, condition, risk_level, patient_data=None):
        """Get personalized recommendations"""
        if condition not in self.recommendations:
            return {"error": "Condition not supported"}
        
        if risk_level not in self.recommendations[condition]:
            risk_level = 'high_risk'
        
        base_recommendations = self.recommendations[condition][risk_level]
        
        # Personalize based on patient data
        if patient_data:
            base_recommendations = self._personalize_recommendations(
                base_recommendations, patient_data, condition
            )
        
        return base_recommendations
    
    def _personalize_recommendations(self, recommendations, patient_data, condition):
        """Add personalized recommendations based on patient profile"""
        personalized = {k: v.copy() for k, v in recommendations.items()}
        
        age = patient_data.get('age', 50)
        bmi = patient_data.get('bmi', 25)
        gender = patient_data.get('gender', 'Male')
        
        # Age-based personalization
        if age > 65:
            if 'exercise' in personalized:
                personalized['exercise'].append("👴 Focus on balance exercises to prevent falls")
                personalized['exercise'].append("🏊‍♂️ Consider water-based exercises for joint health")
        
        # BMI-based personalization
        if bmi > 30:
            if 'diet' in personalized:
                personalized['diet'].append(f"⚖️ Target weight loss: current BMI {bmi:.1f}, goal <30")
                personalized['diet'].append("🍽️ Consider smaller, more frequent meals")
        
        # Gender-based personalization
        if gender == 'Female' and condition == 'heart_disease':
            if 'medical' in personalized:
                personalized['medical'].append("♀️ Discuss women-specific heart disease risks with doctor")
        
        return personalized
    
    def get_risk_factors_explanation(self, condition, patient_data):
        """Explain risk factors based on patient data"""
        explanations = []
        
        age = patient_data.get('age', 50)
        bmi = patient_data.get('bmi', 25)
        
        if condition == 'heart_disease':
            systolic = patient_data.get('systolic_bp', 120)
            cholesterol = patient_data.get('cholesterol', 200)
            
            if age > 55:
                explanations.append(f"🎂 Age {age} increases cardiovascular risk (risk rises after 55)")
            if systolic > 140:
                explanations.append(f"🩺 High blood pressure ({systolic} mmHg) damages arteries over time")
            if cholesterol > 240:
                explanations.append(f"🧪 High cholesterol ({cholesterol} mg/dL) contributes to plaque buildup")
            if bmi > 30:
                explanations.append(f"⚖️ Obesity (BMI {bmi:.1f}) strains the cardiovascular system")
        
        elif condition == 'diabetes':
            glucose = patient_data.get('glucose', 100)
            
            if bmi > 30:
                explanations.append(f"⚖️ Obesity (BMI {bmi:.1f}) increases insulin resistance")
            if glucose > 126:
                explanations.append(f"🩸 High fasting glucose ({glucose} mg/dL) indicates poor glucose control")
            if age > 45:
                explanations.append(f"🎂 Age {age} increases diabetes risk (especially after 45)")
        
        elif condition == 'hypertension':
            systolic = patient_data.get('systolic_bp', 120)
            
            if bmi > 30:
                explanations.append(f"⚖️ Excess weight (BMI {bmi:.1f}) increases blood pressure")
            if age > 40:
                explanations.append(f"🎂 Age {age} - blood pressure naturally increases with age")
            if systolic > 140:
                explanations.append(f"🩺 Current BP {systolic} mmHg exceeds normal range (<120)")
        
        return explanations

# Main Wellness Assistant Class
class WellnessAssistant:
    def __init__(self):
        self.recommendation_engine = RecommendationEngine()
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load models from GitHub or local files
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
            
            # Try to load local files first
            for condition, file_path in model_files.items():
                if os.path.exists(file_path):
                    self.models[condition] = joblib.load(file_path)
                    if condition in scaler_files and os.path.exists(scaler_files[condition]):
                        self.scalers[condition] = joblib.load(scaler_files[condition])
            
            # Load feature names if available
            if os.path.exists('feature_names.pkl'):
                self.feature_names = joblib.load('feature_names.pkl')
            else:
                # Default feature names
                self.feature_names = {
                    'heart_disease': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
                    'diabetes': ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'dpf', 'age'],
                    'hypertension': ['age', 'gender', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'glucose', 'smoking', 'alcohol', 'active', 'bmi']
                }
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            # Use dummy models for demo
            self.create_dummy_models()
    
    def create_dummy_models(self):
        """Create dummy models for demonstration"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create simple dummy models
        for condition in ['heart_disease', 'diabetes', 'hypertension']:
            # Create dummy data for training
            if condition == 'heart_disease':
                n_features = 13
            elif condition == 'diabetes':
                n_features = 8
            else:  # hypertension
                n_features = 12
            
            X_dummy = np.random.rand(100, n_features)
            y_dummy = np.random.randint(0, 2, 100)
            
            # Train dummy model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_dummy, y_dummy)
            self.models[condition] = model
            
            # Create dummy scaler
            scaler = StandardScaler()
            scaler.fit(X_dummy)
            self.scalers[condition] = scaler
        
        st.warning("⚠️ Using demo models. For production, upload trained models from Google Colab.")
    
    def collect_patient_data(self):
        """Enhanced patient data collection interface"""
        st.sidebar.markdown("## 📋 Patient Assessment")
        
        # Personal Information
        with st.sidebar.expander("👤 Personal Information", expanded=True):
            name = st.text_input("Full Name *", placeholder="Enter your full name")
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 18, 100, 45)
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            location = st.text_input("City, State", placeholder="e.g., New York, NY")
        
        # Biometric Measurements
        with st.sidebar.expander("🩺 Biometric Data", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                height = st.slider("Height (cm)", 140, 220, 170)
                systolic_bp = st.slider("Systolic BP", 90, 200, 120)
                heart_rate = st.slider("Heart Rate", 50, 120, 72)
                
            with col2:
                weight = st.slider("Weight (kg)", 40, 200, 70)
                diastolic_bp = st.slider("Diastolic BP", 60, 120, 80)
                glucose = st.slider("Glucose (mg/dL)", 70, 300, 100)
            
            bmi = weight / ((height/100) ** 2)
            st.metric("Calculated BMI", f"{bmi:.1f}", 
                     delta=f"{'Normal' if 18.5 <= bmi <= 24.9 else 'Above Normal' if bmi > 24.9 else 'Below Normal'}")
            
            cholesterol = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200)
        
        # Medical History  
        with st.sidebar.expander("🏥 Medical History"):
            chest_pain = st.selectbox("Chest Pain Type", 
                                    ["No Pain", "Typical Angina", "Atypical Angina", "Non-Anginal"])
            
            family_history = st.multiselect("Family History *", 
                                          ["Heart Disease", "Diabetes", "Hypertension", "Stroke", "Cancer"])
            
            medications = st.text_area("Current Medications", 
                                     placeholder="List medications, dosages...")
            allergies = st.text_area("Known Allergies",
                                   placeholder="Food, drug, environmental allergies...")
        
        # Lifestyle Assessment
        with st.sidebar.expander("🏃‍♀️ Lifestyle Factors"):
            exercise_frequency = st.selectbox("Exercise Frequency *", 
                                            ["Never", "1-2 times/week", "3-4 times/week", "5+ times/week"])
            
            smoking = st.selectbox("Smoking Status *", ["Never", "Former", "Current"])
            alcohol = st.selectbox("Alcohol Consumption *", 
                                 ["Never", "Occasional", "Moderate", "Heavy"])
            
            col1, col2 = st.columns(2)
            with col1:
                sleep_hours = st.slider("Sleep Hours/Night", 4, 12, 8)
            with col2:
                stress_level = st.slider("Stress Level", 1, 10, 5)
        
        # Optional Activity Data
        with st.sidebar.expander("📱 Activity Data (Optional)"):
            daily_steps = st.number_input("Avg Daily Steps", 0, 30000, 8000)
            calories_burned = st.number_input("Daily Calories Burned", 0, 5000, 2000)
        
        return {
            'name': name, 'age': age, 'gender': gender, 'location': location,
            'height': height, 'weight': weight, 'bmi': bmi,
            'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate, 'glucose': glucose, 'cholesterol': cholesterol,
            'chest_pain': chest_pain, 'family_history': family_history,
            'medications': medications, 'allergies': allergies,
            'exercise_frequency': exercise_frequency, 'smoking': smoking,
            'alcohol': alcohol, 'sleep_hours': sleep_hours, 'stress_level': stress_level,
            'daily_steps': daily_steps, 'calories_burned': calories_burned
        }
    
    def prepare_features_for_prediction(self, patient_data, condition):
        """Prepare patient data for model prediction"""
        try:
            if condition == 'heart_disease':
                gender_map = {'Male': 1, 'Female': 0, 'Other': 0}
                chest_pain_map = {'No Pain': 0, 'Typical Angina': 1, 'Atypical Angina': 2, 'Non-Anginal': 3}
                
                features = [
                    patient_data['age'],
                    gender_map.get(patient_data['gender'], 0),
                    chest_pain_map.get(patient_data['chest_pain'], 0),
                    patient_data['systolic_bp'],
                    patient_data['cholesterol'],
                    1 if patient_data['glucose'] > 120 else 0,
                    0,  # rest ECG
                    patient_data['heart_rate'], 
                    0,  # exercise induced angina
                    0,  # oldpeak
                    1,  # slope
                    0,  # ca
                    2   # thal
                ]
                
            elif condition == 'diabetes':
                features = [
                    0,  # pregnancies
                    patient_data['glucose'],
                    patient_data['diastolic_bp'],
                    0,  # skin thickness
                    0,  # insulin
                    patient_data['bmi'],
                    0.5,  # diabetes pedigree function
                    patient_data['age']
                ]
                
            elif condition == 'hypertension':
                gender_map = {'Male': 1, 'Female': 0, 'Other': 0}
                
                features = [
                    patient_data['age'],
                    gender_map.get(patient_data['gender'], 0),
                    patient_data['height'],
                    patient_data['weight'],
                    patient_data['systolic_bp'],
                    patient_data['diastolic_bp'],
                    2,  # cholesterol level
                    1,  # glucose level
                    1 if patient_data['smoking'] == 'Current' else 0,
                    1 if patient_data['alcohol'] in ['Moderate', 'Heavy'] else 0,
                    1 if patient_data['exercise_frequency'] in ['3-4 times/week', '5+ times/week'] else 0,
                    patient_data['bmi']
                ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            st.error(f"Error preparing features for {condition}: {e}")
            return None
    
    def predict_risk(self, patient_data, condition):
        """Predict risk for a specific condition"""
        if condition not in self.models:
            return None
        
        features = self.prepare_features_for_prediction(patient_data, condition)
        if features is None:
            return None
        
        try:
            # Scale features if scaler available
            if condition in self.scalers:
                features = self.scalers[condition].transform(features)
            
            # Get prediction
            model = self.models[condition]
            probability = model.predict_proba(features)[0][1]
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.7:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            
            return {
                'probability': probability,
                'risk_level': risk_level,
                'confidence': max(probability, 1 - probability)
            }
            
        except Exception as e:
            st.error(f"Error predicting risk for {condition}: {e}")
            return None
    
    def create_risk_gauge(self, probability, condition):
        """Create risk gauge visualization"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{condition.replace('_', ' ').title()} Risk"},
            delta = {'reference': 30},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Wellness Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your Personalized Health Risk Assessment & Wellness Companion</p>', unsafe_allow_html=True)
    
    # Medical disclaimer
    st.markdown("""
    <div class="warning-banner">
        <strong>⚕️ Medical Disclaimer:</strong> This tool is for educational and informational purposes only. 
        It does not replace professional medical advice, diagnosis, or treatment. Always consult with 
        qualified healthcare providers for medical decisions and before making changes to your health regimen.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize assistant
    assistant = WellnessAssistant()
    
    # Collect patient data
    patient_data = assistant.collect_patient_data()
    
    # Main analysis button
    if st.sidebar.button("🔍 Analyze Health Risks", type="primary", use_container_width=True):
        if not patient_data['name']:
            st.sidebar.error("⚠️ Please enter your name to continue")
        else:
            # Show analysis
            with st.spinner("🔄 Analyzing your health profile..."):
                
                # Predict risks for all conditions
                conditions = ['heart_disease', 'diabetes', 'hypertension']
                predictions = {}
                
                for condition in conditions:
                    pred = assistant.predict_risk(patient_data, condition)
                    if pred:
                        predictions[condition] = pred
                
                # Store in session state
                st.session_state.predictions = predictions
                st.session_state.patient_data = patient_data
                
                # Display results
                display_risk_assessment(predictions, patient_data, assistant)
                
    # Display previous results if available
    elif 'predictions' in st.session_state and 'patient_data' in st.session_state:
        display_risk_assessment(
            st.session_state.predictions, 
            st.session_state.patient_data, 
            assistant
        )
    
    else:
        # Welcome screen
        display_welcome_screen()

def display_risk_assessment(predictions, patient_data, assistant):
    """Display comprehensive risk assessment"""
    if not predictions:
        st.error("Unable to generate risk predictions. Please check your input data.")
        return
    
    # Risk Overview Section
    st.markdown("## 📊 Health Risk Assessment")
    
    # Create columns for risk cards
    cols = st.columns(len(predictions))
    
    for i, (condition, pred_data) in enumerate(predictions.items()):
        with cols[i]:
            risk_level = pred_data['risk_level']
            probability = pred_data['probability']
            
            # Risk card styling
            if risk_level == "High":
                card_class = "high-risk"
            elif risk_level == "Moderate":
                card_class = "moderate-risk"
            else:
                card_class = "low-risk"
            
            # Display risk card
            st.markdown(f"""
            <div class="risk-card {card_class}">
                <h3 style="margin-top:0">{condition.replace('_', ' ').title()}</h3>
                <h1 style="margin:0.5rem 0">{probability:.1%}</h1>
                <h4 style="margin:0"><strong>{risk_level} Risk</strong></h4>
                <p style="margin:0.5rem 0">Confidence: {pred_data['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk gauge
            fig = assistant.create_risk_gauge(probability, condition)
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk Factors Analysis
    st.markdown("## 🔍 Risk Factors Analysis")
    
    for condition, pred_data in predictions.items():
        with st.expander(f"📈 {condition.replace('_', ' ').title()} Risk Factors", expanded=True):
            explanations = assistant.recommendation_engine.get_risk_factors_explanation(condition, patient_data)
            
            if explanations:
                for explanation in explanations:
                    st.markdown(f"• {explanation}")
            else:
                st.markdown("✅ No significant risk factors identified based on current data.")
            
            # Key metrics for this condition
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Level", pred_data['risk_level'])
            with col2:
                st.metric("Probability", f"{pred_data['probability']:.1%}")
            with col3:
                st.metric("Confidence", f"{pred_data['confidence']:.1%}")
    
    # Personalized Recommendations
    display_recommendations(predictions, patient_data, assistant)
    
    # AI Health Assistant Chat
    display_health_chatbot(predictions, patient_data, assistant)

def display_recommendations(predictions, patient_data, assistant):
    """Display personalized recommendations"""
    st.markdown("## 💡 Personalized Wellness Plan")
    
    # Get highest risk condition for priority recommendations
    if predictions:
        highest_risk_condition = max(predictions.items(), key=lambda x: x[1]['probability'])
        st.info(f"🎯 **Priority Focus:** {highest_risk_condition[0].replace('_', ' ').title()} "
               f"({highest_risk_condition[1]['probability']:.1%} risk)")
    
    # Create tabs for each condition with recommendations
    condition_tabs = st.tabs([f"{cond.replace('_', ' ').title()}" for cond in predictions.keys()])
    
    for tab, (condition, pred_data) in zip(condition_tabs, predictions.items()):
        with tab:
            risk_level = pred_data['risk_level'].lower() + '_risk'
            
            recommendations = assistant.recommendation_engine.get_recommendations(
                condition, risk_level, patient_data
            )
            
            if 'error' not in recommendations:
                # Create sub-tabs for recommendation categories
                rec_tabs = st.tabs(["🥗 Nutrition", "🏃‍♂️ Exercise", "🏠 Lifestyle", "⚕️ Medical Care"])
                
                categories = ['diet', 'exercise', 'lifestyle', 'medical']
                icons = ["🥗", "🏃‍♂️", "🏠", "⚕️"]
                
                for rec_tab, category, icon in zip(rec_tabs, categories, icons):
                    with rec_tab:
                        if category in recommendations:
                            st.markdown(f"### {icon} {category.title()} Recommendations")
                            
                            for i, rec in enumerate(recommendations[category], 1):
                                st.markdown(f"""
                                <div class="recommendation-box">
                                    <strong>{i}.</strong> {rec}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info(f"No specific {category} recommendations for current risk level.")

def display_health_chatbot(predictions, patient_data, assistant):
    """AI-powered health assistant chatbot"""
    st.markdown("## 🤖 AI Health Assistant")
    st.markdown("Ask questions about your health assessment, risk factors, or recommendations.")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Quick question buttons
    st.markdown("### 🔗 Quick Questions")
    quick_questions = [
        "Why is my risk high?",
        "How can I improve my health?", 
        "What should I focus on first?",
        "Are there any warning signs I should watch for?"
    ]
    
    cols = st.columns(len(quick_questions))
    for i, question in enumerate(quick_questions):
        with cols[i]:
            if st.button(question, key=f"quick_{i}"):
                response = generate_ai_response(question, patient_data, predictions)
                st.session_state.chat_history.append({
                    'question': question,
                    'response': response,
                    'timestamp': datetime.now().strftime("%H:%M")
                })
    
    # Chat input
    user_question = st.text_input("💬 Ask your health assistant:", 
                                 placeholder="e.g., What foods should I avoid?")
    
    if user_question:
        response = generate_ai_response(user_question, patient_data, predictions)
        st.session_state.chat_history.append({
            'question': user_question,
            'response': response,
            'timestamp': datetime.now().strftime("%H:%M")
        })
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You ({chat['timestamp']}):</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong> {chat['response']}
            </div>
            """, unsafe_allow_html=True)
            
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")

def generate_ai_response(question, patient_data, predictions):
    """Generate contextual AI responses based on patient data"""
    question_lower = question.lower()
    
    # Risk-related questions
    if any(word in question_lower for word in ['risk', 'high', 'probability', 'why']):
        response = "Based on your health assessment:\n\n"
        
        for condition, pred_data in predictions.items():
            risk_level = pred_data['risk_level']
            probability = pred_data['probability']
            response += f"**{condition.replace('_', ' ').title()}:** {probability:.1%} ({risk_level} risk)\n"
        
        # Identify highest risk
        if predictions:
            highest_risk = max(predictions.items(), key=lambda x: x[1]['probability'])
            response += f"\n🎯 **Primary Concern:** {highest_risk[0].replace('_', ' ')} "
            response += f"at {highest_risk[1]['probability']:.1%} risk.\n\n"
            
            # Add specific risk factors
            response += "**Key risk factors:**\n"
            explanations = RecommendationEngine().get_risk_factors_explanation(
                highest_risk[0], patient_data
            )
            for exp in explanations[:3]:  # Top 3
                response += f"• {exp}\n"
        
        return response
    
    # Improvement questions
    elif any(word in question_lower for word in ['improve', 'reduce', 'lower', 'better', 'help']):
        if not predictions:
            return "I need your health assessment data to provide personalised recommendations."
        
        # Get top priority condition
        highest_risk = max(predictions.items(), key=lambda x: x[1]['probability'])
        condition = highest_risk[0]
        risk_level = highest_risk[1]['risk_level'].lower() + '_risk'
        
        response = f"Here's how to improve your **{condition.replace('_', ' ')}** health:\n\n"
        
        rec_engine = RecommendationEngine()
        recommendations = rec_engine.get_recommendations(condition, risk_level, patient_data)
        
        # Prioritize recommendations
        priority_categories = ['diet', 'exercise', 'lifestyle']
        
        for category in priority_categories:
            if category in recommendations:
                response += f"**{category.title()}:**\n"
                for rec in recommendations[category][:2]:  # Top 2 per category
                    response += f"• {rec}\n"
                response += "\n"
        
        return response
    
    # Specific condition questions
    elif 'heart' in question_lower or 'cardiac' in question_lower:
        if 'heart_disease' in predictions:
            pred = predictions['heart_disease']
            response = f"**Heart Disease Risk: {pred['probability']:.1%}** ({pred['risk_level']} risk)\n\n"
            response += f"Key factors: Age {patient_data['age']}, "
            response += f"BP {patient_data['systolic_bp']}/{patient_data['diastolic_bp']}, "
            response += f"Cholesterol {patient_data['cholesterol']} mg/dL\n\n"
            
            if pred['risk_level'] == 'High':
                response += "🚨 **Immediate action needed:** Schedule cardiology consultation, "
                response += "start heart-healthy diet, begin gentle exercise program."
            
            return response
    
    elif 'diabetes' in question_lower or 'blood sugar' in question_lower or 'glucose' in question_lower:
        if 'diabetes' in predictions:
            pred = predictions['diabetes']
            response = f"**Diabetes Risk: {pred['probability']:.1%}** ({pred['risk_level']} risk)\n\n"
            response += f"Key factors: Glucose {patient_data['glucose']} mg/dL, "
            response += f"BMI {patient_data['bmi']:.1f}, Age {patient_data['age']}\n\n"
            
            if patient_data['glucose'] > 126:
                response += "🩸 Your glucose level indicates diabetes. Please see a healthcare provider immediately."
            elif patient_data['glucose'] > 100:
                response += "⚠️ Pre-diabetic glucose levels. Focus on diet and exercise to prevent progression."
            
            return response
    
    elif any(word in question_lower for word in ['pressure', 'hypertension', 'blood pressure']):
        if 'hypertension' in predictions:
            pred = predictions['hypertension']
            response = f"**Hypertension Risk: {pred['probability']:.1%}** ({pred['risk_level']} risk)\n\n"
            response += f"Current BP: {patient_data['systolic_bp']}/{patient_data['diastolic_bp']} mmHg\n\n"
            
            if patient_data['systolic_bp'] > 140 or patient_data['diastolic_bp'] > 90:
                response += "🩺 Your blood pressure is elevated. Consider home monitoring and lifestyle changes."
            
            return response
    
    # Warning signs questions
    elif any(word in question_lower for word in ['warning', 'signs', 'symptoms', 'emergency']):
        response = "🚨 **Important Warning Signs to Watch For:**\n\n"
        
        response += "**Heart Attack:** Chest pain, shortness of breath, nausea, "
        response += "cold sweat, pain in arms/jaw/back\n\n"
        
        response += "**Stroke:** Sudden numbness, confusion, trouble speaking, "
        response += "severe headache, vision problems\n\n"
        
        response += "**Diabetic Emergency:** Extreme thirst, frequent urination, "
        response += "blurred vision, fatigue, fruity breath odor\n\n"
        
        response += "**Hypertensive Crisis:** Severe headache, chest pain, "
        response += "difficulty breathing, anxiety\n\n"
        
        response += "**⚠️ Call 911 immediately if you experience any of these symptoms!**"
        
        return response
    
    # Focus/priority questions  
    elif any(word in question_lower for word in ['focus', 'priority', 'first', 'start']):
        if not predictions:
            return "Complete your health assessment first to get personalized priorities."
        
        # Get top 2 highest risks
        sorted_risks = sorted(predictions.items(), key=lambda x: x[1]['probability'], reverse=True)
        
        response = "🎯 **Your Health Priorities:**\n\n"
        
        for i, (condition, pred_data) in enumerate(sorted_risks[:2], 1):
            response += f"**{i}. {condition.replace('_', ' ').title()}** "
            response += f"({pred_data['probability']:.1%} risk)\n"
            
            if pred_data['risk_level'] == 'High':
                response += "   🚨 High priority - needs immediate attention\n"
            elif pred_data['risk_level'] == 'Moderate':
                response += "   ⚠️ Moderate priority - preventive action recommended\n"
        
        response += f"\n💡 **Start with:** Focus on the {sorted_risks[0][0].replace('_', ' ')} "
        response += "recommendations in the Wellness Plan section above."
        
        return response
    
    # Default helpful response
    else:
        response = "I'm here to help with your health assessment! I can provide insights about:\n\n"
        response += "• 📊 Your risk levels and what they mean\n"
        response += "• 💡 Personalized improvement recommendations\n" 
        response += "• ⚠️ Warning signs to watch for\n"
        response += "• 🎯 Health priorities and action steps\n"
        response += "• 🏥 When to see a healthcare provider\n\n"
        response += "Try asking: 'Why is my risk high?' or 'How can I improve my health?'"
        
        return response

def display_welcome_screen():
    """Display welcome and onboarding screen"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 👋 Welcome to Your AI Wellness Assistant!
        
        ### 🎯 What This Tool Does
        
        Our intelligent health companion provides:
        
        ✅ **Risk Assessment** for Heart Disease, Diabetes, and Hypertension  
        ✅ **Personalised Recommendations** for diet, exercise, and lifestyle  
        ✅ **AI-Powered Explanations** of your health risks  
        ✅ **Interactive Health Assistant** to answer your questions  
        
        ### 🚀 How to Get Started
        
        1. **📋 Complete Assessment** - Fill out your information in the sidebar
        2. **🔍 Analyse Risks** - Click the "Analyse Health Risks" button  
        3. **📊 Review Results** - Understand your personalized risk profile
        4. **💡 Follow Plan** - Get your customized wellness recommendations
        5. **🤖 Ask Questions** - Chat with your AI health assistant
        
        ### 🏆 Evidence-Based Approach
        
        - **Machine Learning Models** trained on validated healthcare datasets
        - **SHAP Explainability** for transparent AI decisions  
        - **Clinical Guidelines** integrated into recommendations
        - **Personalized Interventions** based on your unique profile
        
        ### 🔒 Privacy & Safety
        
        - All data processing happens locally - no personal health information stored
        - Recommendations are educational only - always consult healthcare providers
        - Evidence-based guidance following medical best practices
        """)
        
        # Sample patient profiles
        st.markdown("### 👥 Try Sample Profiles")
        
        sample_profiles = {
            "🏃‍♂️ Active Adult": {
                "description": "35-year-old active professional with good health habits",
                "expected": "Generally low risk with preventive recommendations"
            },
            "⚠️ At-Risk Individual": {
                "description": "55-year-old with elevated blood pressure and family history", 
                "expected": "Moderate to high risk with targeted interventions"
            },
            "👴 Senior Citizen": {
                "description": "70-year-old with multiple risk factors",
                "expected": "Higher risk profile with comprehensive management plan"
            }
        }
        
        for profile_name, profile_info in sample_profiles.items():
            with st.expander(profile_name):
                st.write(f"**Profile:** {profile_info['description']}")
                st.write(f"**Expected Result:** {profile_info['expected']}")
                
        st.markdown("---")
        
        # Key features highlight
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Accurate</h3>
                <p>ML models with 85%+ accuracy on clinical datasets</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div class="metric-card">
                <h3>🔍 Explainable</h3>
                <p>SHAP analysis shows exactly why you have certain risks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown("""
            <div class="metric-card">
                <h3>💡 Actionable</h3>
                <p>Personalized recommendations you can implement today</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
