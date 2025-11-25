#!/usr/bin/env python3
"""
Diabetes Risk Prediction App
A user-friendly Streamlit interface for diabetes risk prediction using the deployed ML model.
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
from datetime import datetime
import uuid

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DiabetesPredictor:
    def __init__(self, api_url: str = "http://localhost:5003"):
        self.api_url = api_url
        
        # Feature importance mapping (most important features marked as required)
        self.feature_config = {
            # REQUIRED - High importance features
            'HighBP': {'required': True, 'type': 'binary', 'label': 'High Blood Pressure', 'help': 'Do you have high blood pressure?'},
            'HighChol': {'required': True, 'type': 'binary', 'label': 'High Cholesterol', 'help': 'Do you have high cholesterol?'},
            'BMI': {'required': True, 'type': 'numeric', 'label': 'BMI (Body Mass Index)', 'help': 'Your BMI (18.5-40)', 'min': 15.0, 'max': 50.0, 'default': 25.0},
            'GenHlth': {'required': True, 'type': 'scale', 'label': 'General Health', 'help': 'How would you rate your general health?', 'options': ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']},
            'Age': {'required': True, 'type': 'scale', 'label': 'Age Group', 'help': 'Your age group', 'options': ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']},
            
            # OPTIONAL - Medium importance features
            'Smoker': {'required': False, 'type': 'binary', 'label': 'Smoker', 'help': 'Have you smoked at least 100 cigarettes in your lifetime?', 'default': 0},
            'Stroke': {'required': False, 'type': 'binary', 'label': 'Stroke History', 'help': 'Have you ever had a stroke?', 'default': 0},
            'HeartDiseaseorAttack': {'required': False, 'type': 'binary', 'label': 'Heart Disease/Attack', 'help': 'Have you ever had heart disease or heart attack?', 'default': 0},
            'PhysActivity': {'required': False, 'type': 'binary', 'label': 'Physical Activity', 'help': 'Have you done physical activity in the past 30 days?', 'default': 1},
            'DiffWalk': {'required': False, 'type': 'binary', 'label': 'Difficulty Walking', 'help': 'Do you have difficulty walking or climbing stairs?', 'default': 0},
            'Sex': {'required': False, 'type': 'binary', 'label': 'Sex', 'help': 'Biological sex', 'options': ['Female', 'Male'], 'default': 0},
            'MentHlth': {'required': False, 'type': 'numeric', 'label': 'Mental Health Days', 'help': 'Days of poor mental health in past 30 days', 'min': 0, 'max': 30, 'default': 0},
            'PhysHlth': {'required': False, 'type': 'numeric', 'label': 'Physical Health Days', 'help': 'Days of poor physical health in past 30 days', 'min': 0, 'max': 30, 'default': 0},
            
            # OPTIONAL - Lower importance features (will use defaults if not provided)
            'CholCheck': {'required': False, 'type': 'binary', 'label': 'Cholesterol Check', 'help': 'Have you had cholesterol checked in past 5 years?', 'default': 1},
            'Fruits': {'required': False, 'type': 'binary', 'label': 'Fruit Consumption', 'help': 'Do you consume fruit 1+ times per day?', 'default': 1},
            'Veggies': {'required': False, 'type': 'binary', 'label': 'Vegetable Consumption', 'help': 'Do you consume vegetables 1+ times per day?', 'default': 1},
            'HvyAlcoholConsump': {'required': False, 'type': 'binary', 'label': 'Heavy Alcohol Consumption', 'help': 'Heavy drinking (men: 14+ drinks/week, women: 7+ drinks/week)', 'default': 0},
            'AnyHealthcare': {'required': False, 'type': 'binary', 'label': 'Healthcare Access', 'help': 'Do you have any healthcare coverage?', 'default': 1},
            'NoDocbcCost': {'required': False, 'type': 'binary', 'label': 'No Doctor Due to Cost', 'help': 'Was there a time when you needed to see a doctor but could not due to cost?', 'default': 0},
            'Education': {'required': False, 'type': 'scale', 'label': 'Education Level', 'help': 'Highest education level', 'options': ['Never attended/Kindergarten', 'Elementary', 'Some high school', 'High school graduate', 'Some college/technical school', 'College graduate'], 'default': 4},
            'Income': {'required': False, 'type': 'scale', 'label': 'Income Level', 'help': 'Annual household income', 'options': ['<$10k', '$10k-$15k', '$15k-$20k', '$20k-$25k', '$25k-$35k', '$35k-$50k', '$50k-$75k', '$75k+'], 'default': 5}
        }
    
    def render_sidebar(self):
        """Render sidebar with app information"""
        st.sidebar.title("🩺 Diabetes Risk Predictor")
        st.sidebar.markdown("""
        This app predicts your risk of diabetes based on health and lifestyle factors.
        
        **How to use:**
        1. Make sure you've run `./monitor_all_services.sh` first
        2. Fill in the required fields (marked with *)
        3. Optionally fill in additional fields for better accuracy
        4. Click 'Predict Risk' to get your result
        
        **Note:** This is for educational purposes only and should not replace professional medical advice.
        """)
        
        # Connection status
        st.sidebar.subheader("🔗 Connection Status")
        if self.check_api_connection():
            st.sidebar.success("✅ Connected to ML Model")
        else:
            st.sidebar.error("❌ Cannot connect to ML Model")
            st.sidebar.info("Make sure you've run `./monitor_all_services.sh` to start the inference server on port 5003")
    
    def check_api_connection(self) -> bool:
        """Check if the API is accessible"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def render_input_form(self) -> Dict[str, Any]:
        """Render the input form and return user inputs"""
        st.title("🩺 Diabetes Risk Assessment")
        st.markdown("Please provide the following information to assess your diabetes risk:")
        
        user_inputs = {}
        
        # Required fields
        st.subheader("📋 Required Information")
        required_cols = st.columns(2)
        
        col_idx = 0
        for feature, config in self.feature_config.items():
            if not config['required']:
                continue
                
            with required_cols[col_idx % 2]:
                user_inputs[feature] = self.render_input_field(feature, config, required=True)
                col_idx += 1
        
        # Optional fields
        st.subheader("📝 Additional Information (Optional)")
        st.markdown("*Providing this information can improve prediction accuracy*")
        
        with st.expander("🏥 Health History", expanded=False):
            health_cols = st.columns(2)
            health_features = ['Smoker', 'Stroke', 'HeartDiseaseorAttack', 'MentHlth', 'PhysHlth']
            
            for i, feature in enumerate(health_features):
                if feature in self.feature_config:
                    with health_cols[i % 2]:
                        user_inputs[feature] = self.render_input_field(feature, self.feature_config[feature])
        
        with st.expander("🏃 Lifestyle Factors", expanded=False):
            lifestyle_cols = st.columns(2)
            lifestyle_features = ['PhysActivity', 'DiffWalk', 'Fruits', 'Veggies', 'HvyAlcoholConsump']
            
            for i, feature in enumerate(lifestyle_features):
                if feature in self.feature_config:
                    with lifestyle_cols[i % 2]:
                        user_inputs[feature] = self.render_input_field(feature, self.feature_config[feature])
        
        with st.expander("👤 Demographics & Access", expanded=False):
            demo_cols = st.columns(2)
            demo_features = ['Sex', 'Education', 'Income', 'AnyHealthcare', 'NoDocbcCost', 'CholCheck']
            
            for i, feature in enumerate(demo_features):
                if feature in self.feature_config:
                    with demo_cols[i % 2]:
                        user_inputs[feature] = self.render_input_field(feature, self.feature_config[feature])
        
        return user_inputs
    
    def render_input_field(self, feature: str, config: Dict, required: bool = False):
        """Render individual input field based on configuration"""
        label = config['label']
        if required:
            label += " *"
        
        help_text = config.get('help', '')
        
        if config['type'] == 'binary':
            if 'options' in config:
                # Binary with custom labels
                options = config['options']
                selected = st.selectbox(label, options, help=help_text)
                return 1 if selected == options[1] else 0
            else:
                # Standard binary (Yes/No)
                return 1 if st.checkbox(label, help=help_text) else 0
        
        elif config['type'] == 'numeric':
            min_val = config.get('min', 0)
            max_val = config.get('max', 100)
            default_val = config.get('default', min_val)
            
            if required:
                return st.number_input(label, min_value=min_val, max_value=max_val, value=default_val, help=help_text)
            else:
                use_field = st.checkbox(f"Provide {config['label']}", key=f"use_{feature}")
                if use_field:
                    return st.number_input(label, min_value=min_val, max_value=max_val, value=default_val, help=help_text, key=f"input_{feature}")
                else:
                    return default_val
        
        elif config['type'] == 'scale':
            options = config['options']
            if required:
                selected = st.selectbox(label, options, help=help_text)
                return options.index(selected)
            else:
                use_field = st.checkbox(f"Provide {config['label']}", key=f"use_{feature}")
                if use_field:
                    selected = st.selectbox(label, options, help=help_text, key=f"input_{feature}")
                    return options.index(selected)
                else:
                    return config.get('default', 0)
        
        return config.get('default', 0)
    
    def prepare_features(self, user_inputs: Dict[str, Any]) -> Dict[str, float]:
        """Prepare features for API call, filling in defaults for missing values"""
        features = {}
        
        # Fill in all features
        for feature, config in self.feature_config.items():
            if feature in user_inputs:
                features[feature] = float(user_inputs[feature])
            else:
                # Use default value
                features[feature] = float(config.get('default', 0))
        
        # Add the target variable (not used for prediction but expected by API)
        features['Diabetes_binary'] = 0.0
        
        return features
    
    def make_prediction(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction API call"""
        try:
            # Remove the Diabetes_binary field as it's not needed for prediction
            prediction_features = {k: v for k, v in features.items() if k != 'Diabetes_binary'}
            
            response = requests.post(
                f"{self.api_url}/predict",
                json=prediction_features,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                # Log the prediction for health authority monitoring
                self.log_prediction(features, result)
                return result
            else:
                return {"error": f"API returned status code {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def log_prediction(self, features: Dict[str, float], result: Dict[str, Any]):
        """Log prediction to inference_data.json for health authority monitoring"""
        try:
            # Path to the inference data file
            data_file = os.path.join("dashboards", "data", "inference_data.json")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            # Read existing data
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    data = json.load(f)
                # Get current count from existing data
                current_count = len(data.get("predictions", []))
            else:
                data = {
                    "predictions": [],
                    "metadata": {
                        "total_predictions": 0,
                        "high_risk_count": 0,
                        "low_risk_count": 0,
                        "last_updated": None,
                        "created": datetime.now().isoformat(),
                        "last_github_push": None,
                        "predictions_since_last_push": 0
                    }
                }
                current_count = 0
            
            # Create prediction record
            prediction_record = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "features": {k: v for k, v in features.items() if k != 'Diabetes_binary'},
                "prediction": {
                    "diabetes_probability": result.get('diabetes_probability', 0.0),
                    "risk_level": result.get('risk_level', 'LOW'),
                    "threshold": result.get('threshold', 0.5)
                },
                "risk_factors": {
                    "high_bp": features.get('HighBP', 0),
                    "high_chol": features.get('HighChol', 0),
                    "bmi": features.get('BMI', 0),
                    "smoker": features.get('Smoker', 0),
                    "age_group": features.get('Age', 0),
                    "general_health": features.get('GenHlth', 0)
                },
                "demographics": {
                    "sex": features.get('Sex', 0),
                    "age_group": features.get('Age', 0),
                    "education": features.get('Education', 0),
                    "income": features.get('Income', 0)
                }
            }
            
            # Add to predictions list
            data["predictions"].append(prediction_record)
            
            # Update metadata
            data["metadata"]["total_predictions"] += 1
            data["metadata"]["last_updated"] = datetime.now().isoformat()
            data["metadata"]["predictions_since_last_push"] += 1
            
            if result.get('risk_level') == 'HIGH':
                data["metadata"]["high_risk_count"] += 1
            else:
                data["metadata"]["low_risk_count"] += 1
            
            # Keep only last 1000 predictions to prevent file from growing too large
            if len(data["predictions"]) > 1000:
                data["predictions"] = data["predictions"][-1000:]
            
            # Write updated data
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Check if we need to push to GitHub (more than 2 new predictions)
            new_predictions_count = len(data["predictions"]) - current_count + 1  # +1 for current prediction
            predictions_since_push = data["metadata"]["predictions_since_last_push"]
            
            print(f"📊 Inference logged: {new_predictions_count} new predictions, {predictions_since_push} since last push")
            
            if predictions_since_push >= 2:
                print("🚀 Threshold reached! Pushing to GitHub...")
                success = self.push_to_github(data_file)
                if success:
                    # Reset counter after successful push
                    data["metadata"]["predictions_since_last_push"] = 0
                    data["metadata"]["last_github_push"] = datetime.now().isoformat()
                    
                    # Update the file with reset counter
                    with open(data_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    print("✅ Successfully pushed to GitHub and reset counter")
                else:
                    print("❌ Failed to push to GitHub, counter not reset")
                
        except Exception as e:
            # Don't fail the prediction if logging fails
            print(f"Warning: Failed to log prediction: {e}")
    
    def get_counter_status(self) -> Dict[str, Any]:
        """Get current counter status from inference data file"""
        try:
            data_file = os.path.join("dashboards", "data", "inference_data.json")
            
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                metadata = data.get("metadata", {})
                return {
                    "total_predictions": metadata.get("total_predictions", 0),
                    "predictions_since_last_push": metadata.get("predictions_since_last_push", 0),
                    "last_github_push": metadata.get("last_github_push", "Never"),
                    "last_updated": metadata.get("last_updated", "Never"),
                    "high_risk_count": metadata.get("high_risk_count", 0),
                    "low_risk_count": metadata.get("low_risk_count", 0)
                }
            else:
                return {
                    "total_predictions": 0,
                    "predictions_since_last_push": 0,
                    "last_github_push": "Never",
                    "last_updated": "Never",
                    "high_risk_count": 0,
                    "low_risk_count": 0
                }
        except Exception as e:
            print(f"Error getting counter status: {e}")
            return {
                "total_predictions": 0,
                "predictions_since_last_push": 0,
                "last_github_push": "Error",
                "last_updated": "Error",
                "high_risk_count": 0,
                "low_risk_count": 0
            }
    
    def push_to_github(self, data_file: str) -> bool:
        """Push the inference data file to GitHub repository"""
        try:
            import subprocess
            import os
            
            # Get current working directory
            current_dir = os.getcwd()
            print(f"📁 Current directory: {current_dir}")
            
            # Check if we're in a git repository
            git_check = subprocess.run(['git', 'status'], 
                                     capture_output=True, text=True, cwd=current_dir)
            
            if git_check.returncode != 0:
                print("❌ Not in a git repository")
                return False
            
            # Add the inference data file to git
            add_result = subprocess.run(['git', 'add', data_file], 
                                      capture_output=True, text=True, cwd=current_dir)
            
            if add_result.returncode != 0:
                print(f"❌ Failed to add file to git: {add_result.stderr}")
                return False
            
            # Check if there are changes to commit
            status_result = subprocess.run(['git', 'status', '--porcelain', data_file], 
                                         capture_output=True, text=True, cwd=current_dir)
            
            if not status_result.stdout.strip():
                print("ℹ️ No changes to commit")
                return True
            
            # Commit the changes
            commit_message = f"Auto-update inference data - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            commit_result = subprocess.run(['git', 'commit', '-m', commit_message], 
                                         capture_output=True, text=True, cwd=current_dir)
            
            if commit_result.returncode != 0:
                print(f"❌ Failed to commit: {commit_result.stderr}")
                return False
            
            print(f"✅ Committed changes: {commit_message}")
            
            # Push to remote repository
            push_result = subprocess.run(['git', 'push'], 
                                       capture_output=True, text=True, cwd=current_dir)
            
            if push_result.returncode != 0:
                print(f"❌ Failed to push to GitHub: {push_result.stderr}")
                # Even if push fails, we committed locally, so return True
                print("⚠️ Changes committed locally but not pushed to remote")
                return True
            
            print("✅ Successfully pushed to GitHub")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git command failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Error pushing to GitHub: {e}")
            return False
    
    def render_counter_status(self):
        """Render counter status in the sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Data Tracking Status")
        
        status = self.get_counter_status()
        
        # Counter status
        st.sidebar.metric(
            label="Total Predictions",
            value=status["total_predictions"]
        )
        
        st.sidebar.metric(
            label="New Predictions (since last push)",
            value=status["predictions_since_last_push"],
            delta=f"Push at {2}" if status["predictions_since_last_push"] < 2 else "Ready to push!"
        )
        
        # Progress bar for GitHub push threshold
        progress = min(status["predictions_since_last_push"] / 2.0, 1.0)
        st.sidebar.progress(progress)
        
        if status["predictions_since_last_push"] >= 2:
            st.sidebar.success("🚀 Ready to push to GitHub!")
        else:
            remaining = 2 - status["predictions_since_last_push"]
            st.sidebar.info(f"📈 {remaining} more prediction(s) needed to trigger GitHub push")
        
        # Risk distribution
        if status["total_predictions"] > 0:
            st.sidebar.markdown("**Risk Distribution:**")
            high_pct = (status["high_risk_count"] / status["total_predictions"]) * 100
            low_pct = (status["low_risk_count"] / status["total_predictions"]) * 100
            
            st.sidebar.write(f"🔴 High Risk: {status['high_risk_count']} ({high_pct:.1f}%)")
            st.sidebar.write(f"🟢 Low Risk: {status['low_risk_count']} ({low_pct:.1f}%)")
        
        # Last update info
        st.sidebar.markdown("**Last Updates:**")
        if status["last_updated"] != "Never":
            try:
                last_update = datetime.fromisoformat(status["last_updated"].replace('Z', '+00:00'))
                st.sidebar.write(f"📝 Last prediction: {last_update.strftime('%Y-%m-%d %H:%M')}")
            except:
                st.sidebar.write(f"📝 Last prediction: {status['last_updated']}")
        else:
            st.sidebar.write("📝 Last prediction: Never")
        
        if status["last_github_push"] != "Never":
            try:
                last_push = datetime.fromisoformat(status["last_github_push"].replace('Z', '+00:00'))
                st.sidebar.write(f"🚀 Last GitHub push: {last_push.strftime('%Y-%m-%d %H:%M')}")
            except:
                st.sidebar.write(f"🚀 Last GitHub push: {status['last_github_push']}")
        else:
            st.sidebar.write("🚀 Last GitHub push: Never")
    
    def render_prediction_result(self, result: Dict[str, Any], features: Dict[str, float]):
        """Render prediction results"""
        if "error" in result:
            st.error(f"❌ Prediction failed: {result['error']}")
            return
        
        # Parse the API response format
        probability = result.get('diabetes_probability', 0.0)
        risk_level = result.get('risk_level', 'LOW')
        prediction = 1 if risk_level == 'HIGH' else 0
        
        # Main result
        st.subheader("🎯 Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **HIGH RISK**")
                st.markdown("The model indicates elevated diabetes risk")
            else:
                st.success("✅ **LOW RISK**")
                st.markdown("The model indicates lower diabetes risk")
        
        with col2:
            st.metric("Risk Probability", f"{probability:.1%}")
        
        with col3:
            confidence = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.15 else "Low"
            st.info(f"Confidence: {confidence}")
        
        # Risk interpretation
        st.subheader("📊 Risk Interpretation")
        
        if probability < 0.3:
            st.success("**Low Risk (< 30%)**: Your current health profile suggests a lower likelihood of diabetes. Continue maintaining healthy habits!")
        elif probability < 0.7:
            st.warning("**Moderate Risk (30-70%)**: Your health profile shows some risk factors. Consider lifestyle improvements and regular health checkups.")
        else:
            st.error("**High Risk (> 70%)**: Your health profile indicates significant risk factors. Please consult with a healthcare professional for proper evaluation and guidance.")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        recommendations = []
        
        # Analyze key risk factors
        if features.get('HighBP', 0) == 1:
            recommendations.append("🩺 **Blood Pressure**: Work with your doctor to manage high blood pressure through medication and lifestyle changes")
        
        if features.get('HighChol', 0) == 1:
            recommendations.append("🧪 **Cholesterol**: Follow your doctor's advice for managing high cholesterol, including diet and possible medication")
        
        if features.get('BMI', 25) > 30:
            recommendations.append("⚖️ **Weight Management**: Consider a structured weight loss program with diet and exercise")
        elif features.get('BMI', 25) > 25:
            recommendations.append("⚖️ **Weight**: Maintain a healthy weight through balanced diet and regular exercise")
        
        if features.get('PhysActivity', 1) == 0:
            recommendations.append("🏃 **Exercise**: Incorporate at least 150 minutes of moderate exercise per week")
        
        if features.get('Smoker', 0) == 1:
            recommendations.append("🚭 **Smoking**: Consider smoking cessation programs - quitting smoking significantly reduces diabetes risk")
        
        if features.get('Fruits', 1) == 0 or features.get('Veggies', 1) == 0:
            recommendations.append("🥗 **Diet**: Increase consumption of fruits and vegetables for better nutrition")
        
        # General recommendations
        recommendations.extend([
            "🩺 **Regular Checkups**: Schedule regular health screenings and blood glucose tests",
            "😴 **Sleep**: Maintain 7-9 hours of quality sleep per night",
            "🧘 **Stress Management**: Practice stress reduction techniques like meditation or yoga",
            "💧 **Hydration**: Stay well-hydrated and limit sugary drinks"
        ])
        
        for rec in recommendations[:6]:  # Show top 6 recommendations
            st.markdown(rec)
        
        # Disclaimer
        st.subheader("⚠️ Important Disclaimer")
        st.warning("""
        **This prediction is for educational purposes only and should not replace professional medical advice.**
        
        - This model is based on population data and may not account for individual circumstances
        - Always consult with healthcare professionals for proper medical evaluation
        - Regular health screenings are important regardless of this prediction
        - If you have concerns about diabetes risk, please see your doctor
        """)
        
        # Show feature summary
        with st.expander("📋 Feature Summary", expanded=False):
            feature_df = pd.DataFrame([
                {"Feature": self.feature_config[k]['label'], "Value": v, "Feature Name": k}
                for k, v in features.items() if k != 'Diabetes_binary'
            ])
            st.dataframe(feature_df, use_container_width=True)

def main():
    predictor = DiabetesPredictor()
    
    # Render sidebar
    predictor.render_sidebar()
    
    # Display counter status in sidebar
    predictor.render_counter_status()
    
    # Main app
    user_inputs = predictor.render_input_form()
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True):
            if not predictor.check_api_connection():
                st.error("❌ Cannot connect to the ML model. Please ensure the inference server is running and accessible.")
                st.info("Make sure you've run `./monitor_all_services.sh` first to start all services including the inference API on port 5003")
                return
            
            # Prepare features
            features = predictor.prepare_features(user_inputs)
            
            # Make prediction
            with st.spinner("🔄 Analyzing your health profile..."):
                result = predictor.make_prediction(features)
            
            # Show results
            predictor.render_prediction_result(result, features)

if __name__ == "__main__":
    main()