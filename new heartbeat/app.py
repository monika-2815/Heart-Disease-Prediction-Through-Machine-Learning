from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, make_response
import csv
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statistics
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from config import config
from models import db, User, Prediction, MLModel, SystemStats

def create_app(config_name='default'):
    """Application factory pattern"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    
    return app

app = create_app()
app.secret_key = app.config['SECRET_KEY']

# Database-based user authentication

# CSV file for storing prediction data
CSV_FILE = 'heart_disease_predictions.csv'
CSV_HEADERS = ['timestamp', 'username', 'age', 'sex', 'chest_pain_type', 'resting_bp', 
               'serum_cholesterol', 'fasting_bs', 'resting_ecg', 'max_heart_rate', 
               'exercise_angina', 'st_depression', 'st_slope', 'major_vessels', 
               'thalassemia', 'diagnostic_score', 'diagnosis', 'severity', 'confidence']

# Model file for storing trained ML model
MODEL_FILE = 'heart_disease_model.pkl'
LABEL_ENCODER_FILE = 'label_encoders.pkl'

# Database helper functions
def get_user_by_username(username):
    """Get user by username from database"""
    return User.query.filter_by(username=username, is_active=True).first()

def authenticate_user(username, password):
    """Authenticate user with database"""
    user = get_user_by_username(username)
    if user and user.check_password(password):
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        return user
    return None

def require_admin(f):
    """Decorator to require admin role"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login'))
        
        user = db.session.get(User, session['user_id'])
        if not user or user.role != 'admin':
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('home'))
        
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Get current logged-in user"""
    if 'user_id' not in session:
        return None
    return db.session.get(User, session['user_id'])

def create_user(username, password, email=None, full_name=None, role='user'):
    """Create new user in database"""
    if get_user_by_username(username):
        return None  # User already exists
    
    user = User(
        username=username,
        email=email,
        full_name=full_name,
        role=role
    )
    user.set_password(password)
    
    try:
        db.session.add(user)
        db.session.commit()
        return user
    except Exception as e:
        db.session.rollback()
        return None

# Store prediction data in database
def store_prediction_data(user_id, features, result):
    """Store prediction data in both CSV and database"""
    try:
        # Store in database
        prediction = Prediction(
            user_id=user_id,
            age=int(features['age']),
            sex=features['sex'],
            chest_pain_type=int(features['chest_pain_type']),
            resting_bp=int(features['resting_bp']),
            serum_cholesterol=int(features['serum_cholesterol']),
            fasting_bs=features['fasting_bs'],
            resting_ecg=int(features['resting_ecg']),
            max_heart_rate=int(features['max_heart_rate']),
            exercise_angina=features['exercise_angina'],
            st_depression=float(features['st_depression']),
            st_slope=int(features['st_slope']),
            major_vessels=int(features['major_vessels']),
            thalassemia=int(features['thalassemia']),
            diagnostic_score=result['diagnostic_score'],
            diagnosis=result['diagnosis'],
            severity=result['severity'],
            confidence=result['confidence'],
            parameter_analysis=json.dumps(result.get('parameter_analysis')) if result.get('parameter_analysis') else None,
            risk_stratification=json.dumps(result.get('risk_stratification')) if result.get('risk_stratification') else None,
            lifestyle_recommendations=json.dumps(result.get('lifestyle_recommendations')) if result.get('lifestyle_recommendations') else None,
            follow_up_plan=json.dumps(result.get('follow_up_plan')) if result.get('follow_up_plan') else None,
            medical_insights=json.dumps(result.get('medical_insights')) if result.get('medical_insights') else None,
            patient_summary=result.get('patient_summary'),
            model_type=result.get('model_type', 'rule_based')
        )
        
        db.session.add(prediction)
        db.session.commit()
        
        prediction_id = prediction.id
        
        # Also store in CSV file (matching existing format)
        try:
            # Get username for CSV
            user = db.session.get(User, user_id)
            username = user.username if user else 'unknown'
            
            csv_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
                'username': username,
                'age': features['age'],
                'sex': features['sex'],
                'chest_pain_type': features['chest_pain_type'],
                'resting_bp': features['resting_bp'],
                'serum_cholesterol': features['serum_cholesterol'],
                'fasting_bs': features['fasting_bs'],
                'resting_ecg': features['resting_ecg'],
                'max_heart_rate': features['max_heart_rate'],
                'exercise_angina': features['exercise_angina'],
                'st_depression': features['st_depression'],
                'st_slope': features['st_slope'],
                'major_vessels': features['major_vessels'],
                'thalassemia': features['thalassemia'],
                'diagnostic_score': result['diagnostic_score'],
                'diagnosis': result['diagnosis'],
                'severity': result['severity'],
                'confidence': result['confidence']
            }
            
            # Write to CSV
            csv_file = 'heart_disease_predictions.csv'
            file_exists = os.path.exists(csv_file)
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as file:
                fieldnames = list(csv_data.keys())
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(csv_data)
                
            print(f"Data stored in both database (ID: {prediction_id}) and CSV file")
            
        except Exception as csv_error:
            print(f"Error storing in CSV: {csv_error}")
            # Continue even if CSV storage fails
        
        return prediction
        
    except Exception as e:
        db.session.rollback()
        print(f"Error storing prediction: {e}")
        return None

# Retrieve prediction data from both CSV and database
def get_prediction_data(prediction_id, prefer_csv=True):
    """Retrieve prediction data from CSV first (for PDF generation), fallback to database"""
    if prefer_csv:
        # Try CSV first (for PDF generation)
        try:
            csv_file = 'heart_disease_predictions.csv'
            if os.path.exists(csv_file):
                with open(csv_file, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    row_count = 0
                    for row in reader:
                        row_count += 1
                        # Match by row number (since CSV doesn't have ID field)
                        if row_count == prediction_id:
                            return {
                                'source': 'csv',
                                'data': row
                            }
        except Exception as e:
            print(f"Error retrieving from CSV: {e}")
        
        # Fallback to database
        try:
            prediction = db.session.get(Prediction, prediction_id)
            if prediction:
                return {
                    'source': 'database',
                    'data': prediction
                }
        except Exception as e:
            print(f"Error retrieving from database: {e}")
    else:
        # Try database first (for other operations)
        try:
            prediction = db.session.get(Prediction, prediction_id)
            if prediction:
                return {
                    'source': 'database',
                    'data': prediction
                }
        except Exception as e:
            print(f"Error retrieving from database: {e}")
        
        # Fallback to CSV
        try:
            csv_file = 'heart_disease_predictions.csv'
            if os.path.exists(csv_file):
                with open(csv_file, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    row_count = 0
                    for row in reader:
                        row_count += 1
                        # Match by row number (since CSV doesn't have ID field)
                        if row_count == prediction_id:
                            return {
                                'source': 'csv',
                                'data': row
                            }
        except Exception as e:
            print(f"Error retrieving from CSV: {e}")
    
    return None

# Load training data from database
def load_training_data():
    """Load training data from database"""
    try:
        predictions = Prediction.query.all()
        training_data = []
        
        for pred in predictions:
            training_data.append({
                'age': pred.age,
                'sex': pred.sex,
                'chest_pain_type': pred.chest_pain_type,
                'resting_bp': pred.resting_bp,
                'serum_cholesterol': pred.serum_cholesterol,
                'fasting_bs': pred.fasting_bs,
                'resting_ecg': pred.resting_ecg,
                'max_heart_rate': pred.max_heart_rate,
                'exercise_angina': pred.exercise_angina,
                'st_depression': pred.st_depression,
                'st_slope': pred.st_slope,
                'major_vessels': pred.major_vessels,
                'thalassemia': pred.thalassemia,
                'diagnostic_score': pred.diagnostic_score,
                'diagnosis': pred.diagnosis,
                'severity': pred.severity,
                'confidence': pred.confidence,
                'timestamp': pred.created_at.isoformat() if pred.created_at else '',
                'username': pred.user.username if pred.user else ''
            })
        
        return training_data
    except Exception as e:
        print(f"Error loading training data: {e}")
        return []

# Calculate adaptive thresholds based on training data
def calculate_adaptive_thresholds(training_data):
    if len(training_data) < 5:  # Need minimum data points
        return None
    
    # Extract diagnostic scores and their outcomes
    scores = []
    high_risk_scores = []
    moderate_risk_scores = []
    low_risk_scores = []
    
    for row in training_data:
        try:
            score = int(row['diagnostic_score'])
            scores.append(score)
            
            if 'HIGH' in row['diagnosis'] or 'CRITICAL' in row['severity']:
                high_risk_scores.append(score)
            elif 'MODERATE' in row['diagnosis']:
                moderate_risk_scores.append(score)
            else:
                low_risk_scores.append(score)
        except (ValueError, KeyError):
            continue
    
    if not scores:
        return None
    
    # Calculate adaptive thresholds using custom quantile function
    def calculate_quantile(data, q):
        """Calculate quantile for a given dataset"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (len(sorted_data) - 1) * q
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    adaptive_thresholds = {
        'low_threshold': calculate_quantile(scores, 0.3) if len(scores) >= 3 else 6,
        'moderate_threshold': calculate_quantile(scores, 0.6) if len(scores) >= 3 else 9,
        'high_threshold': calculate_quantile(scores, 0.8) if len(scores) >= 3 else 12,
        'critical_threshold': calculate_quantile(scores, 0.9) if len(scores) >= 3 else 15,
        'total_predictions': len(scores),
        'high_risk_count': len(high_risk_scores),
        'moderate_risk_count': len(moderate_risk_scores),
        'low_risk_count': len(low_risk_scores)
    }
    
    return adaptive_thresholds

# Load trained ML model from pickle file
def load_trained_model():
    """Load the trained machine learning model and label encoders from pickle files"""
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(LABEL_ENCODER_FILE):
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            with open(LABEL_ENCODER_FILE, 'rb') as f:
                label_encoders = pickle.load(f)
            return model, label_encoders
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Save trained ML model to pickle file
def save_trained_model(model, label_encoders):
    """Save the trained machine learning model and label encoders to pickle files"""
    try:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        with open(LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(label_encoders, f)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

# Train ML model from CSV data
def train_ml_model():
    """Train a machine learning model using the collected prediction data"""
    try:
        # Load training data
        training_data = load_training_data()
        
        if len(training_data) < 10:  # Need minimum data points for training
            return {
                'success': False,
                'message': f'Insufficient data for training. Need at least 10 records, got {len(training_data)}',
                'data_count': len(training_data)
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Prepare features (X) and target (y)
        feature_columns = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'serum_cholesterol', 
                          'fasting_bs', 'resting_ecg', 'max_heart_rate', 'exercise_angina', 
                          'st_depression', 'st_slope', 'major_vessels', 'thalassemia']
        
        # Convert features to numeric
        X = df[feature_columns].copy()
        
        # Convert categorical variables
        label_encoders = {}
        categorical_columns = ['sex', 'fasting_bs', 'exercise_angina']
        
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Convert all columns to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Create target variable based on diagnosis
        y = df['diagnosis'].copy()
        
        # Create binary classification: 1 for heart disease, 0 for no heart disease
        y_binary = y.apply(lambda x: 1 if 'HIGH' in str(x) or 'MODERATE' in str(x) else 0)
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model and encoders
        save_success = save_trained_model(model, label_encoders)
        
        if save_success:
            return {
                'success': True,
                'message': f'Model trained successfully with {len(training_data)} records',
                'accuracy': round(accuracy * 100, 2),
                'data_count': len(training_data),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_importance': dict(zip(feature_columns, model.feature_importances_))
            }
        else:
            return {
                'success': False,
                'message': 'Model training completed but failed to save model',
                'accuracy': round(accuracy * 100, 2),
                'data_count': len(training_data)
            }
            
    except Exception as e:
        return {
            'success': False,
            'message': f'Error training model: {str(e)}',
            'data_count': len(training_data) if 'training_data' in locals() else 0
        }

# Predict using trained ML model
def predict_with_ml_model(features):
    """Make prediction using the trained machine learning model"""
    try:
        model, label_encoders = load_trained_model()
        
        if model is None or label_encoders is None:
            return None
        
        # Prepare features for prediction with proper column names
        feature_data = {
            'age': int(features['age']),
            'sex': 1 if features['sex'] == 'Male' else 0,
            'chest_pain_type': int(features['chest_pain_type']),
            'resting_bp': int(features['resting_bp']),
            'serum_cholesterol': int(features['serum_cholesterol']),
            'fasting_bs': 1 if features['fasting_bs'] == '> 120 mg/dl' else 0,
            'resting_ecg': int(features['resting_ecg']),
            'max_heart_rate': int(features['max_heart_rate']),
            'exercise_angina': 1 if features['exercise_angina'] == 'Yes' else 0,
            'st_depression': float(features['st_depression']),
            'st_slope': int(features['st_slope']),
            'major_vessels': int(features['major_vessels']),
            'thalassemia': int(features['thalassemia'])
        }
        
        # Create DataFrame with proper column names
        feature_df = pd.DataFrame([feature_data])
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        probability = model.predict_proba(feature_df)[0]
        
        # Get confidence score
        confidence_score = max(probability) * 100
        
        return {
            'prediction': prediction,
            'confidence': round(confidence_score, 2),
            'probabilities': {
                'no_disease': round(probability[0] * 100, 2),
                'disease': round(probability[1] * 100, 2)
            }
        }
        
    except Exception as e:
        print(f"Error in ML prediction: {e}")
        return None

# Analyze medical parameters for detailed insights
def analyze_medical_parameters(features):
    """Provide detailed analysis of each medical parameter"""
    analysis = {}
    
    # Age analysis
    age = int(features['age'])
    if age < 40:
        analysis['age'] = {'status': 'Low Risk', 'interpretation': 'Young age group with lower cardiovascular risk'}
    elif age < 65:
        analysis['age'] = {'status': 'Moderate Risk', 'interpretation': 'Middle-aged with moderate cardiovascular risk'}
    else:
        analysis['age'] = {'status': 'High Risk', 'interpretation': 'Advanced age significantly increases cardiovascular risk'}
    
    # Blood pressure analysis
    bp = int(features['resting_bp'])
    if bp < 120:
        analysis['blood_pressure'] = {'status': 'Normal', 'interpretation': 'Optimal blood pressure range'}
    elif bp < 140:
        analysis['blood_pressure'] = {'status': 'Elevated', 'interpretation': 'Pre-hypertensive range, monitor closely'}
    elif bp < 160:
        analysis['blood_pressure'] = {'status': 'High', 'interpretation': 'Stage 1 hypertension, requires management'}
    else:
        analysis['blood_pressure'] = {'status': 'Very High', 'interpretation': 'Stage 2 hypertension, immediate attention needed'}
    
    # Cholesterol analysis
    cholesterol = int(features['serum_cholesterol'])
    if cholesterol < 200:
        analysis['cholesterol'] = {'status': 'Desirable', 'interpretation': 'Optimal cholesterol levels'}
    elif cholesterol < 240:
        analysis['cholesterol'] = {'status': 'Borderline', 'interpretation': 'Borderline high cholesterol, lifestyle modifications recommended'}
    else:
        analysis['cholesterol'] = {'status': 'High', 'interpretation': 'High cholesterol, medical intervention may be required'}
    
    # Heart rate analysis
    max_hr = int(features['max_heart_rate'])
    predicted_max = 220 - age
    if max_hr >= predicted_max * 0.85:
        analysis['heart_rate'] = {'status': 'Good', 'interpretation': 'Good exercise capacity and cardiovascular fitness'}
    elif max_hr >= predicted_max * 0.70:
        analysis['heart_rate'] = {'status': 'Moderate', 'interpretation': 'Moderate exercise capacity'}
    else:
        analysis['heart_rate'] = {'status': 'Poor', 'interpretation': 'Reduced exercise capacity, may indicate cardiovascular limitations'}
    
    return analysis

# Perform comprehensive risk stratification
def perform_risk_stratification(diagnostic_score, features):
    """Provide detailed risk stratification analysis"""
    age = int(features['age'])
    sex = features['sex']
    
    # Calculate 10-year risk estimate
    base_risk = min(diagnostic_score * 2.5, 50)  # Convert to percentage
    
    # Age adjustment
    if age > 65:
        base_risk += 15
    elif age > 55:
        base_risk += 10
    elif age > 45:
        base_risk += 5
    
    # Gender adjustment
    if sex == 'Male':
        base_risk += 5
    
    # Risk categories
    if base_risk < 10:
        risk_category = 'Low Risk'
        risk_description = 'Less than 10% risk of cardiovascular events in next 10 years'
    elif base_risk < 20:
        risk_category = 'Moderate Risk'
        risk_description = '10-20% risk of cardiovascular events in next 10 years'
    elif base_risk < 30:
        risk_category = 'High Risk'
        risk_description = '20-30% risk of cardiovascular events in next 10 years'
    else:
        risk_category = 'Very High Risk'
        risk_description = 'Greater than 30% risk of cardiovascular events in next 10 years'
    
    return {
        'risk_category': risk_category,
        'risk_percentage': round(base_risk, 1),
        'risk_description': risk_description,
        'timeframe': '10-year cardiovascular risk estimate'
    }

# Generate lifestyle recommendations
def generate_lifestyle_recommendations(features, risk_factors):
    """Generate personalized lifestyle recommendations"""
    recommendations = []
    
    # Blood pressure recommendations
    bp = int(features['resting_bp'])
    if bp >= 140:
        recommendations.append({
            'category': 'Blood Pressure Management',
            'recommendations': [
                'Reduce sodium intake to less than 2,300mg per day',
                'Engage in regular aerobic exercise (150 minutes/week)',
                'Maintain healthy weight (BMI 18.5-24.9)',
                'Limit alcohol consumption',
                'Consider DASH diet (Dietary Approaches to Stop Hypertension)'
            ]
        })
    
    # Cholesterol recommendations
    cholesterol = int(features['serum_cholesterol'])
    if cholesterol >= 200:
        recommendations.append({
            'category': 'Cholesterol Management',
            'recommendations': [
                'Adopt heart-healthy diet (Mediterranean or DASH)',
                'Increase soluble fiber intake (oats, beans, fruits)',
                'Limit saturated fats and trans fats',
                'Include omega-3 fatty acids (fish, nuts)',
                'Consider plant sterols and stanols'
            ]
        })
    
    # Exercise recommendations
    if 'Exercise-induced angina' in risk_factors:
        recommendations.append({
            'category': 'Exercise Management',
            'recommendations': [
                'Start with low-intensity activities (walking, swimming)',
                'Gradually increase intensity under medical supervision',
                'Avoid high-intensity exercises initially',
                'Monitor symptoms during exercise',
                'Consider cardiac rehabilitation program'
            ]
        })
    else:
        recommendations.append({
            'category': 'Physical Activity',
            'recommendations': [
                'Aim for 150 minutes of moderate-intensity exercise weekly',
                'Include both aerobic and strength training',
                'Start slowly if new to exercise',
                'Choose activities you enjoy for consistency',
                'Monitor heart rate during exercise'
            ]
        })
    
    # General lifestyle recommendations
    recommendations.append({
        'category': 'General Health',
        'recommendations': [
            'Quit smoking if applicable',
            'Manage stress through relaxation techniques',
            'Get 7-9 hours of quality sleep nightly',
            'Stay hydrated (8 glasses of water daily)',
            'Regular health check-ups and screenings'
        ]
    })
    
    return recommendations

# Generate follow-up plan
def generate_follow_up_plan(severity, diagnostic_score):
    """Generate personalized follow-up care plan"""
    if severity in ['Critical', 'High']:
        return {
            'immediate_actions': [
                'Schedule cardiology consultation within 1-2 weeks',
                'Consider stress testing or cardiac imaging',
                'Review current medications with healthcare provider',
                'Implement immediate lifestyle modifications'
            ],
            'short_term_goals': [
                'Complete comprehensive cardiac evaluation within 1 month',
                'Establish baseline cardiovascular risk profile',
                'Begin appropriate medical therapy if indicated',
                'Start cardiac rehabilitation if appropriate'
            ],
            'long_term_monitoring': [
                'Regular follow-up every 3-6 months',
                'Annual comprehensive cardiovascular assessment',
                'Continuous monitoring of risk factors',
                'Adjust treatment plan based on response'
            ]
        }
    elif severity == 'Moderate':
        return {
            'immediate_actions': [
                'Schedule primary care follow-up within 2-4 weeks',
                'Begin lifestyle modifications immediately',
                'Consider cardiac risk assessment',
                'Review family history of cardiovascular disease'
            ],
            'short_term_goals': [
                'Complete cardiovascular risk assessment within 2-3 months',
                'Implement comprehensive lifestyle changes',
                'Monitor response to interventions',
                'Consider preventive medications if indicated'
            ],
            'long_term_monitoring': [
                'Regular follow-up every 6 months',
                'Annual cardiovascular risk reassessment',
                'Continuous lifestyle modification support',
                'Monitor for progression of risk factors'
            ]
        }
    else:
        return {
            'immediate_actions': [
                'Continue current healthy lifestyle practices',
                'Schedule routine annual physical examination',
                'Maintain regular exercise routine',
                'Continue heart-healthy diet'
            ],
            'short_term_goals': [
                'Maintain current cardiovascular health status',
                'Continue preventive health measures',
                'Regular health screenings as recommended',
                'Stay informed about cardiovascular health'
            ],
            'long_term_monitoring': [
                'Annual comprehensive health assessment',
                'Regular monitoring of cardiovascular risk factors',
                'Continue preventive lifestyle measures',
                'Stay updated on cardiovascular health guidelines'
            ]
        }

# Generate medical insights
def generate_medical_insights(features, diagnostic_score, risk_factors):
    """Generate detailed medical insights and interpretations"""
    insights = []
    
    # Age-related insights
    age = int(features['age'])
    if age > 65:
        insights.append({
            'category': 'Age-Related Risk',
            'insight': f'At age {age}, cardiovascular risk naturally increases. Regular monitoring and preventive care are essential.',
            'clinical_significance': 'High'
        })
    
    # Gender-specific insights
    sex = features['sex']
    if sex == 'Male':
        insights.append({
            'category': 'Gender Risk',
            'insight': 'Males have higher baseline cardiovascular risk compared to premenopausal females.',
            'clinical_significance': 'Moderate'
        })
    
    # Chest pain analysis
    chest_pain = int(features['chest_pain_type'])
    if chest_pain == 0:  # Typical angina
        insights.append({
            'category': 'Symptom Analysis',
            'insight': 'Typical angina symptoms are highly suggestive of coronary artery disease and require immediate evaluation.',
            'clinical_significance': 'High'
        })
    
    # ECG insights
    ecg = int(features['resting_ecg'])
    if ecg == 1:  # ST-T abnormalities
        insights.append({
            'category': 'ECG Findings',
            'insight': 'ST-T wave abnormalities may indicate myocardial ischemia or other cardiac conditions requiring further evaluation.',
            'clinical_significance': 'High'
        })
    
    # Exercise capacity insights
    max_hr = int(features['max_heart_rate'])
    predicted_max = 220 - age
    if max_hr < predicted_max * 0.70:
        insights.append({
            'category': 'Exercise Capacity',
            'insight': f'Reduced exercise capacity (achieved {max_hr} bpm vs predicted {predicted_max} bpm) may indicate cardiovascular limitations.',
            'clinical_significance': 'Moderate'
        })
    
    # Risk factor clustering
    if len(risk_factors) >= 5:
        insights.append({
            'category': 'Risk Factor Clustering',
            'insight': f'Multiple risk factors present ({len(risk_factors)} factors) significantly increase cardiovascular risk through synergistic effects.',
            'clinical_significance': 'High'
        })
    
    return insights

# Generate patient summary
def generate_patient_summary(features, diagnostic_score, severity):
    """Generate a comprehensive patient summary"""
    age = int(features['age'])
    sex = features['sex']
    
    summary = {
        'demographics': f'{age}-year-old {sex.lower()}',
        'primary_concern': 'Cardiovascular risk assessment',
        'risk_level': severity,
        'key_findings': [],
        'next_steps': []
    }
    
    # Key findings based on severity
    if severity in ['Critical', 'High']:
        summary['key_findings'] = [
            'High cardiovascular risk profile identified',
            'Multiple risk factors present',
            'Immediate medical evaluation recommended'
        ]
        summary['next_steps'] = [
            'Schedule cardiology consultation',
            'Consider advanced cardiac testing',
            'Implement aggressive risk factor modification'
        ]
    elif severity == 'Moderate':
        summary['key_findings'] = [
            'Moderate cardiovascular risk identified',
            'Some risk factors present',
            'Lifestyle modifications recommended'
        ]
        summary['next_steps'] = [
            'Primary care follow-up',
            'Implement lifestyle changes',
            'Monitor risk factors regularly'
        ]
    else:
        summary['key_findings'] = [
            'Low cardiovascular risk profile',
            'Minimal risk factors identified',
            'Continue preventive measures'
        ]
        summary['next_steps'] = [
            'Maintain healthy lifestyle',
            'Regular health screenings',
            'Continue preventive care'
        ]
    
    return summary

# Enhanced prediction function with adaptive learning and ML model
def predict_heart_disease(features):
    """
    Predictive modeling for heart disease diagnosis using patient health records
    This function analyzes medical parameters to provide diagnostic insights
    Enhanced with adaptive learning from previous predictions and ML model when available
    """
    # Try to use trained ML model first
    ml_prediction = predict_with_ml_model(features)
    
    if ml_prediction is not None:
        # Use ML model prediction
        prediction = ml_prediction['prediction']
        ml_confidence = ml_prediction['confidence']
        
        # Convert ML prediction to diagnosis format
        if prediction == 1:  # Heart disease predicted
            if ml_confidence >= 90:
                diagnosis = "HIGH PROBABILITY OF HEART DISEASE"
                severity = "High"
                confidence = f"Very High ({ml_confidence}%)"
            elif ml_confidence >= 75:
                diagnosis = "MODERATE-HIGH PROBABILITY OF HEART DISEASE"
                severity = "Moderate-High"
                confidence = f"High ({ml_confidence}%)"
            else:
                diagnosis = "MODERATE PROBABILITY OF HEART DISEASE"
                severity = "Moderate"
                confidence = f"Moderate ({ml_confidence}%)"
        else:  # No heart disease predicted
            if ml_confidence >= 90:
                diagnosis = "LOW PROBABILITY OF HEART DISEASE"
                severity = "Low"
                confidence = f"Very High ({ml_confidence}%)"
            else:
                diagnosis = "LOW-MODERATE PROBABILITY OF HEART DISEASE"
                severity = "Low-Moderate"
                confidence = f"Moderate ({ml_confidence}%)"
        
        # Generate detailed analysis for ML model prediction
        parameter_analysis = analyze_medical_parameters(features)
        risk_stratification = perform_risk_stratification(int(ml_confidence), features)
        lifestyle_recommendations = generate_lifestyle_recommendations(features, ["ML Model Analysis"])
        follow_up_plan = generate_follow_up_plan(severity, int(ml_confidence))
        medical_insights = generate_medical_insights(features, int(ml_confidence), ["ML Model Analysis"])
        
        # Create result with ML model information and detailed analysis
        result = {
            'diagnosis': diagnosis,
            'confidence': confidence,
            'severity': severity,
            'diagnostic_score': int(ml_confidence),  # Use confidence as score
            'risk_factors': ["ML Model Analysis"],
            'recommendation': "Based on machine learning model trained on historical data",
            'total_risk_factors': 1,
            'learning_status': f"ML Model Prediction (Confidence: {ml_confidence}%)",
            'ml_probabilities': ml_prediction['probabilities'],
            'model_type': 'Machine Learning',
            'parameter_analysis': parameter_analysis,
            'risk_stratification': risk_stratification,
            'lifestyle_recommendations': lifestyle_recommendations,
            'follow_up_plan': follow_up_plan,
            'medical_insights': medical_insights,
            'patient_summary': generate_patient_summary(features, int(ml_confidence), severity)
        }
        
        return result
    
    # Fallback to original rule-based prediction if ML model is not available
    # Convert features to numerical values
    age = int(features['age'])
    sex = 1 if features['sex'] == 'Male' else 0
    chest_pain = int(features['chest_pain_type'])
    bp = int(features['resting_bp'])
    cholesterol = int(features['serum_cholesterol'])
    fasting_bs = 1 if features['fasting_bs'] == '> 120 mg/dl' else 0
    ecg = int(features['resting_ecg'])
    max_hr = int(features['max_heart_rate'])
    exercise_angina = 1 if features['exercise_angina'] == 'Yes' else 0
    st_depression = float(features['st_depression'])
    st_slope = int(features['st_slope'])
    vessels = int(features['major_vessels'])
    thalassemia = int(features['thalassemia'])
    
    # Advanced diagnostic scoring system based on medical research
    diagnostic_score = 0
    risk_factors = []
    
    # Age-related risk assessment
    if age > 65:
        diagnostic_score += 3
        risk_factors.append("Advanced age (>65)")
    elif age > 50:
        diagnostic_score += 2
        risk_factors.append("Middle age (50-65)")
    
    # Gender-specific risk
    if sex == 1:  # Male
        diagnostic_score += 1
        risk_factors.append("Male gender")
    
    # Chest pain analysis (critical diagnostic factor)
    if chest_pain == 0:  # Typical Angina
        diagnostic_score += 4
        risk_factors.append("Typical angina symptoms")
    elif chest_pain == 1:  # Atypical Angina
        diagnostic_score += 3
        risk_factors.append("Atypical angina symptoms")
    elif chest_pain == 2:  # Non-anginal Pain
        diagnostic_score += 1
        risk_factors.append("Non-anginal chest pain")
    
    # Blood pressure assessment
    if bp >= 180:
        diagnostic_score += 3
        risk_factors.append("Severe hypertension (≥180 mmHg)")
    elif bp >= 140:
        diagnostic_score += 2
        risk_factors.append("Hypertension (≥140 mmHg)")
    
    # Cholesterol levels
    if cholesterol >= 300:
        diagnostic_score += 3
        risk_factors.append("Very high cholesterol (≥300 mg/dl)")
    elif cholesterol >= 200:
        diagnostic_score += 2
        risk_factors.append("High cholesterol (≥200 mg/dl)")
    
    # Blood sugar analysis
    if fasting_bs == 1:
        diagnostic_score += 2
        risk_factors.append("Elevated fasting blood sugar (>120 mg/dl)")
    
    # ECG abnormalities
    if ecg == 1:  # ST-T Wave Abnormality
        diagnostic_score += 3
        risk_factors.append("ST-T wave abnormalities on ECG")
    elif ecg == 2:  # Left Ventricular Hypertrophy
        diagnostic_score += 2
        risk_factors.append("Left ventricular hypertrophy")
    
    # Heart rate analysis
    if max_hr < 100:
        diagnostic_score += 2
        risk_factors.append("Low maximum heart rate (<100 bpm)")
    elif max_hr > 180:
        diagnostic_score += 1
        risk_factors.append("High maximum heart rate (>180 bpm)")
    
    # Exercise-induced symptoms
    if exercise_angina == 1:
        diagnostic_score += 3
        risk_factors.append("Exercise-induced angina")
    
    # ST segment analysis
    if st_depression >= 3.0:
        diagnostic_score += 4
        risk_factors.append("Significant ST depression (≥3.0 mm)")
    elif st_depression >= 1.0:
        diagnostic_score += 2
        risk_factors.append("ST depression (≥1.0 mm)")
    
    # ST slope characteristics
    if st_slope == 2:  # Downsloping
        diagnostic_score += 3
        risk_factors.append("Downsloping ST segment")
    elif st_slope == 1:  # Flat
        diagnostic_score += 1
        risk_factors.append("Flat ST segment")
    
    # Coronary vessel assessment
    if vessels >= 3:
        diagnostic_score += 4
        risk_factors.append("Multiple major vessel involvement (≥3)")
    elif vessels >= 2:
        diagnostic_score += 3
        risk_factors.append("Major vessel disease (≥2 vessels)")
    elif vessels == 1:
        diagnostic_score += 1
        risk_factors.append("Single vessel involvement")
    
    # Thalassemia assessment
    if thalassemia == 2:  # Reversable Defect
        diagnostic_score += 2
        risk_factors.append("Reversible perfusion defect")
    elif thalassemia == 1:  # Fixed Defect
        diagnostic_score += 1
        risk_factors.append("Fixed perfusion defect")
    
    # Load training data and calculate adaptive thresholds
    training_data = load_training_data()
    adaptive_thresholds = calculate_adaptive_thresholds(training_data)
    
    # Use adaptive thresholds if available, otherwise use default
    if adaptive_thresholds:
        low_threshold = adaptive_thresholds['low_threshold']
        moderate_threshold = adaptive_thresholds['moderate_threshold']
        high_threshold = adaptive_thresholds['high_threshold']
        critical_threshold = adaptive_thresholds['critical_threshold']
        
        # Add learning indicator
        learning_status = f"Enhanced with {adaptive_thresholds['total_predictions']} previous assessments"
    else:
        # Default thresholds
        low_threshold = 6
        moderate_threshold = 9
        high_threshold = 12
        critical_threshold = 15
        learning_status = "Using baseline medical thresholds"
    
    # Diagnostic classification based on adaptive scoring
    if diagnostic_score >= critical_threshold:
        diagnosis = "HIGH PROBABILITY OF HEART DISEASE"
        confidence = "Very High (95-99%)"
        recommendation = "Immediate medical evaluation and intervention recommended. Consider cardiac catheterization."
        severity = "Critical"
    elif diagnostic_score >= high_threshold:
        diagnosis = "MODERATE-HIGH PROBABILITY OF HEART DISEASE"
        confidence = "High (80-94%)"
        recommendation = "Urgent cardiac evaluation recommended. Consider stress testing and cardiac imaging."
        severity = "High"
    elif diagnostic_score >= moderate_threshold:
        diagnosis = "MODERATE PROBABILITY OF HEART DISEASE"
        confidence = "Moderate (60-79%)"
        recommendation = "Cardiac evaluation recommended. Consider non-invasive cardiac testing."
        severity = "Moderate"
    elif diagnostic_score >= low_threshold:
        diagnosis = "LOW-MODERATE PROBABILITY OF HEART DISEASE"
        confidence = "Low-Moderate (30-59%)"
        recommendation = "Monitor symptoms. Consider cardiac evaluation if symptoms persist."
        severity = "Low-Moderate"
    else:
        diagnosis = "LOW PROBABILITY OF HEART DISEASE"
        confidence = "Low (10-29%)"
        recommendation = "Continue routine health monitoring. Maintain healthy lifestyle."
        severity = "Low"
    
    # Enhanced medical parameter analysis
    parameter_analysis = analyze_medical_parameters(features)
    risk_stratification = perform_risk_stratification(diagnostic_score, features)
    lifestyle_recommendations = generate_lifestyle_recommendations(features, risk_factors)
    follow_up_plan = generate_follow_up_plan(severity, diagnostic_score)
    medical_insights = generate_medical_insights(features, diagnostic_score, risk_factors)
    
    # Create comprehensive diagnostic report
    result = {
        'diagnosis': diagnosis,
        'confidence': confidence,
        'severity': severity,
        'diagnostic_score': diagnostic_score,
        'risk_factors': risk_factors,
        'recommendation': recommendation,
        'total_risk_factors': len(risk_factors),
        'learning_status': learning_status,
        'adaptive_thresholds': adaptive_thresholds if adaptive_thresholds else None,
        'parameter_analysis': parameter_analysis,
        'risk_stratification': risk_stratification,
        'lifestyle_recommendations': lifestyle_recommendations,
        'follow_up_plan': follow_up_plan,
        'medical_insights': medical_insights,
        'patient_summary': generate_patient_summary(features, diagnostic_score, severity)
    }
    
    return result

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('home'))

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('predict.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user.id
            session['username'] = user.username
            session['user_role'] = user.role
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'error')
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        email = request.form.get('email', '')
        full_name = request.form.get('full_name', '')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html', error='Passwords do not match')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html', error='Password must be at least 6 characters long')
        
        user = create_user(username, password, email, full_name)
        if user:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username already exists', 'error')
            return render_template('register.html', error='Username already exists')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('user_role', None)
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        
        # Extract features from the form
        features = {
            'age': data['age'],
            'sex': data['sex'],
            'chest_pain_type': data['chest_pain_type'],
            'resting_bp': data['resting_bp'],
            'serum_cholesterol': data['serum_cholesterol'],
            'fasting_bs': data['fasting_bs'],
            'resting_ecg': data['resting_ecg'],
            'max_heart_rate': data['max_heart_rate'],
            'exercise_angina': data['exercise_angina'],
            'st_depression': data['st_depression'],
            'st_slope': data['st_slope'],
            'major_vessels': data['major_vessels'],
            'thalassemia': data['thalassemia']
        }
        
        # Make comprehensive diagnostic prediction
        diagnostic_result = predict_heart_disease(features)
        
        # Store prediction data in database
        prediction = store_prediction_data(session['user_id'], features, diagnostic_result)
        
        if prediction:
            return jsonify({
                'prediction': diagnostic_result,
                'prediction_id': prediction.id,
                'success': True
            })
        else:
            return jsonify({
                'error': 'Failed to store prediction',
                'success': False
            }), 500
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

def generate_pdf_report(prediction_id, user_data, prediction_data):
    """Generate a formatted PDF report for the diagnostic assessment"""
    try:
        print(f"Starting PDF generation for prediction {prediction_id}")
        print(f"User data: {user_data}")
        print(f"Prediction data keys: {list(prediction_data.keys())}")
        
        # Validate input data
        if not user_data or not prediction_data:
            print("Error: Missing user_data or prediction_data")
            return None
        
        # Create a BytesIO buffer to store the PDF
        from io import BytesIO
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.darkgreen
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6
        )
        
        # Build the PDF content
        story = []
        
        # Title
        story.append(Paragraph("HEART DISEASE DIAGNOSTIC ASSESSMENT REPORT", title_style))
        story.append(Spacer(1, 20))
        
        # Report Information
        story.append(Paragraph("Report Information", heading_style))
        report_info_data = [
            ['Report ID:', f"HD-{prediction_id:06d}"],
            ['Generated On:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Patient Name:', user_data.get('username', 'N/A')],
            ['Report Type:', 'Comprehensive Heart Disease Assessment']
        ]
        
        report_table = Table(report_info_data, colWidths=[2*inch, 4*inch])
        report_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(report_table)
        story.append(Spacer(1, 20))
        
        # Patient Information
        story.append(Paragraph("Patient Information", heading_style))
        patient_data = [
            ['Age:', f"{prediction_data['features']['age']} years"],
            ['Gender:', prediction_data['features']['sex']],
            ['Chest Pain Type:', prediction_data['features']['chest_pain_type']],
            ['Resting Blood Pressure:', f"{prediction_data['features']['resting_bp']} mmHg"],
            ['Serum Cholesterol:', f"{prediction_data['features']['serum_cholesterol']} mg/dl"],
            ['Fasting Blood Sugar:', f"{prediction_data['features']['fasting_bs']} mg/dl"],
            ['Resting ECG:', prediction_data['features']['resting_ecg']],
            ['Max Heart Rate:', f"{prediction_data['features']['max_heart_rate']} bpm"],
            ['Exercise Angina:', prediction_data['features']['exercise_angina']],
            ['ST Depression:', f"{prediction_data['features']['st_depression']} mm"],
            ['ST Slope:', prediction_data['features']['st_slope']],
            ['Major Vessels:', prediction_data['features']['major_vessels']],
            ['Thalassemia:', prediction_data['features']['thalassemia']]
        ]
        
        patient_table = Table(patient_data, colWidths=[2.5*inch, 3.5*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # Diagnostic Results
        story.append(Paragraph("Diagnostic Results", heading_style))
        
        # Primary Diagnosis
        story.append(Paragraph("Primary Diagnosis", subheading_style))
        diagnosis_color = colors.red if prediction_data['diagnosis'] == 'Heart Disease Detected' else colors.green
        story.append(Paragraph(f"<b>Result:</b> {prediction_data['diagnosis']}", 
                              ParagraphStyle('Diagnosis', parent=normal_style, 
                                           textColor=diagnosis_color, fontSize=12)))
        story.append(Spacer(1, 10))
        
        # Risk Assessment
        story.append(Paragraph("Risk Assessment", subheading_style))
        risk_data = [
            ['Diagnostic Score:', f"{prediction_data['diagnostic_score']}/20"],
            ['Severity Level:', prediction_data['severity']],
            ['Confidence Level:', prediction_data['confidence']],
            ['Model Used:', prediction_data['model_type'].title()]
        ]
        
        risk_table = Table(risk_data, colWidths=[2*inch, 4*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 20))
        
        # Detailed Analysis
        if 'parameter_analysis' in prediction_data:
            story.append(Paragraph("Detailed Medical Analysis", heading_style))
            param_analysis = prediction_data['parameter_analysis']
            # Handle both dict and JSON string
            if isinstance(param_analysis, str):
                try:
                    import json
                    param_analysis = json.loads(param_analysis)
                except:
                    param_analysis = {}
            if isinstance(param_analysis, dict):
                for param, analysis in param_analysis.items():
                    story.append(Paragraph(f"<b>{param.replace('_', ' ').title()}:</b>", subheading_style))
                    if isinstance(analysis, dict):
                        if 'status' in analysis:
                            story.append(Paragraph(f"Status: {analysis['status']}", normal_style))
                        if 'interpretation' in analysis:
                            story.append(Paragraph(f"Interpretation: {analysis['interpretation']}", normal_style))
                    else:
                        story.append(Paragraph(str(analysis), normal_style))
                    story.append(Spacer(1, 8))
        
        # Risk Stratification
        if 'risk_stratification' in prediction_data:
            story.append(Paragraph("Risk Stratification", heading_style))
            risk_strat = prediction_data['risk_stratification']
            # Handle both dict and JSON string
            if isinstance(risk_strat, str):
                try:
                    import json
                    risk_strat = json.loads(risk_strat)
                except:
                    risk_strat = {}
            if isinstance(risk_strat, dict):
                story.append(Paragraph(f"<b>10-Year Cardiovascular Risk:</b> {risk_strat.get('risk_percentage', 'N/A')}%", normal_style))
                story.append(Paragraph(f"<b>Risk Category:</b> {risk_strat.get('risk_category', 'N/A')}", normal_style))
            story.append(Spacer(1, 10))
        
        # Lifestyle Recommendations
        if 'lifestyle_recommendations' in prediction_data:
            story.append(Paragraph("Lifestyle Recommendations", heading_style))
            recommendations = prediction_data['lifestyle_recommendations']
            # Handle both list and JSON string
            if isinstance(recommendations, str):
                try:
                    import json
                    recommendations = json.loads(recommendations)
                except:
                    recommendations = []
            if isinstance(recommendations, list):
                for i, rec in enumerate(recommendations, 1):
                    if isinstance(rec, dict) and 'category' in rec:
                        story.append(Paragraph(f"<b>{rec['category']}:</b>", subheading_style))
                        for j, item in enumerate(rec.get('recommendations', []), 1):
                            story.append(Paragraph(f"  {j}. {item}", normal_style))
                    else:
                        story.append(Paragraph(f"{i}. {rec}", normal_style))
            story.append(Spacer(1, 10))
        
        # Follow-up Plan
        if 'follow_up_plan' in prediction_data:
            story.append(Paragraph("Follow-up Plan", heading_style))
            follow_up = prediction_data['follow_up_plan']
            # Handle both dict and JSON string
            if isinstance(follow_up, str):
                try:
                    import json
                    follow_up = json.loads(follow_up)
                except:
                    follow_up = {}
            if isinstance(follow_up, dict):
                if 'immediate_actions' in follow_up:
                    story.append(Paragraph("<b>Immediate Actions:</b>", subheading_style))
                    if isinstance(follow_up['immediate_actions'], list):
                        for action in follow_up['immediate_actions']:
                            story.append(Paragraph(f"• {action}", normal_style))
                    else:
                        story.append(Paragraph(follow_up['immediate_actions'], normal_style))
                
                if 'short_term' in follow_up:
                    story.append(Paragraph("<b>Short-term (1-3 months):</b>", subheading_style))
                    if isinstance(follow_up['short_term'], list):
                        for action in follow_up['short_term']:
                            story.append(Paragraph(f"• {action}", normal_style))
                    else:
                        story.append(Paragraph(follow_up['short_term'], normal_style))
                
                if 'long_term' in follow_up:
                    story.append(Paragraph("<b>Long-term (3-12 months):</b>", subheading_style))
                    if isinstance(follow_up['long_term'], list):
                        for action in follow_up['long_term']:
                            story.append(Paragraph(f"• {action}", normal_style))
                    else:
                        story.append(Paragraph(follow_up['long_term'], normal_style))
            story.append(Spacer(1, 10))
        
        # Medical Insights
        if 'medical_insights' in prediction_data:
            story.append(Paragraph("Clinical Insights", heading_style))
            insights = prediction_data['medical_insights']
            # Handle both list and JSON string
            if isinstance(insights, str):
                try:
                    import json
                    insights = json.loads(insights)
                except:
                    insights = []
            if isinstance(insights, list):
                for insight in insights:
                    story.append(Paragraph(f"• {insight}", normal_style))
            story.append(Spacer(1, 10))
        
        # Patient Summary
        if 'patient_summary' in prediction_data:
            story.append(Paragraph("Patient Summary", heading_style))
            story.append(Paragraph(prediction_data['patient_summary'], normal_style))
            story.append(Spacer(1, 20))
        
        # Footer
        story.append(Paragraph("--- End of Report ---", 
                              ParagraphStyle('Footer', parent=normal_style, 
                                           alignment=TA_CENTER, fontSize=10, 
                                           textColor=colors.grey)))
        story.append(Spacer(1, 10))
        story.append(Paragraph("This report was generated by the Heart Disease Diagnostic System", 
                              ParagraphStyle('Footer', parent=normal_style, 
                                           alignment=TA_CENTER, fontSize=9, 
                                           textColor=colors.grey)))
        story.append(Paragraph("Designed and Developed by Monika P", 
                              ParagraphStyle('Footer', parent=normal_style, 
                                           alignment=TA_CENTER, fontSize=9, 
                                           textColor=colors.grey)))
        
        # Build PDF
        doc.build(story)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        print(f"PDF generation completed successfully, size: {len(pdf_content)} bytes")
        return pdf_content
        
    except Exception as e:
        print(f"Error generating PDF for prediction {prediction_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/download_report/<int:prediction_id>')
def download_report(prediction_id):
    """Download PDF report for a specific prediction"""
    if 'user_id' not in session:
        flash('Please login to download reports', 'error')
        return redirect(url_for('login'))
    
    try:
        # Get the prediction data from CSV first (for PDF generation)
        prediction_result = get_prediction_data(prediction_id, prefer_csv=True)
        if not prediction_result:
            flash('Prediction not found', 'error')
            return redirect(url_for('personal_history'))
        
        prediction = prediction_result['data']
        data_source = prediction_result['source']
        
        print(f"Retrieved prediction {prediction_id} from {data_source}")
        
        # Check if user owns this prediction (unless admin)
        current_user = get_current_user()
        if data_source == 'database':
            user_id = prediction.user_id
            # Get user data from database
            user = db.session.get(User, user_id)
            user_data = {
                'username': user.username,
                'email': user.email if hasattr(user, 'email') else 'N/A'
            }
        else:  # CSV source
            # For CSV, we need to find user by username
            username = prediction['username']
            user = User.query.filter_by(username=username).first()
            user_id = user.id if user else None
            user_data = {
                'username': prediction['username'],
                'email': user.email if user and hasattr(user, 'email') else 'N/A'
            }
        
        # Check access control
        if current_user and current_user.role != 'admin':
            if user_id is None or user_id != session['user_id']:
                flash('Access denied', 'error')
                return redirect(url_for('personal_history'))
        elif not current_user:
            flash('Access denied', 'error')
            return redirect(url_for('personal_history'))
        
        # Reconstruct prediction data (prioritize CSV format)
        if data_source == 'csv':
            # CSV source - primary format for PDF generation
            prediction_data = {
                'features': {
                    'age': int(prediction['age']),
                    'sex': prediction['sex'],
                    'chest_pain_type': int(prediction['chest_pain_type']),
                    'resting_bp': int(prediction['resting_bp']),
                    'serum_cholesterol': int(prediction['serum_cholesterol']),
                    'fasting_bs': prediction['fasting_bs'],
                    'resting_ecg': int(prediction['resting_ecg']),
                    'max_heart_rate': int(prediction['max_heart_rate']),
                    'exercise_angina': prediction['exercise_angina'],
                    'st_depression': float(prediction['st_depression']),
                    'st_slope': int(prediction['st_slope']),
                    'major_vessels': int(prediction['major_vessels']),
                    'thalassemia': int(prediction['thalassemia'])
                },
                'diagnosis': prediction['diagnosis'],
                'diagnostic_score': int(prediction['diagnostic_score']),
                'severity': prediction['severity'],
                'confidence': prediction['confidence'],
                'model_type': 'rule_based'  # CSV format uses rule_based
            }
        else:  # Database source - fallback
            prediction_data = {
                'features': {
                    'age': prediction.age,
                    'sex': prediction.sex,
                    'chest_pain_type': prediction.chest_pain_type,
                    'resting_bp': prediction.resting_bp,
                    'serum_cholesterol': prediction.serum_cholesterol,
                    'fasting_bs': prediction.fasting_bs,
                    'resting_ecg': prediction.resting_ecg,
                    'max_heart_rate': prediction.max_heart_rate,
                    'exercise_angina': prediction.exercise_angina,
                    'st_depression': prediction.st_depression,
                    'st_slope': prediction.st_slope,
                    'major_vessels': prediction.major_vessels,
                    'thalassemia': prediction.thalassemia
                },
                'diagnosis': prediction.diagnosis,
                'diagnostic_score': prediction.diagnostic_score,
                'severity': prediction.severity,
                'confidence': prediction.confidence,
                'model_type': prediction.model_type
            }
        
        # Add detailed analysis (CSV format doesn't store detailed analysis, so we use fallback data)
        # This ensures consistent PDF generation regardless of data source
        pass
        
        # Add fallback data for predictions without detailed analysis
        if not prediction_data.get('parameter_analysis'):
            prediction_data['parameter_analysis'] = {
                'age': {'status': 'Standard', 'interpretation': 'Age-based risk assessment'},
                'blood_pressure': {'status': 'Standard', 'interpretation': 'Blood pressure evaluation'},
                'cholesterol': {'status': 'Standard', 'interpretation': 'Cholesterol level assessment'},
                'heart_rate': {'status': 'Standard', 'interpretation': 'Heart rate analysis'}
            }
        
        if not prediction_data.get('risk_stratification'):
            diagnostic_score = prediction_data.get('diagnostic_score', 0)
            severity = prediction_data.get('severity', 'Unknown')
            prediction_data['risk_stratification'] = {
                'risk_percentage': diagnostic_score,
                'risk_category': severity,
                'risk_description': f'Based on diagnostic score of {diagnostic_score}',
                'timeframe': '10-year cardiovascular risk estimate'
            }
        
        if not prediction_data.get('lifestyle_recommendations'):
            prediction_data['lifestyle_recommendations'] = [
                {'category': 'General Health', 'recommendations': [
                    'Maintain a healthy diet',
                    'Exercise regularly',
                    'Avoid smoking',
                    'Manage stress',
                    'Regular health check-ups'
                ]}
            ]
        
        if not prediction_data.get('follow_up_plan'):
            prediction_data['follow_up_plan'] = {
                'immediate_actions': ['Consult with healthcare provider'],
                'short_term': ['Monitor health parameters'],
                'long_term': ['Maintain healthy lifestyle']
            }
        
        if not prediction_data.get('medical_insights'):
            model_type = prediction_data.get('model_type', 'rule_based')
            severity = prediction_data.get('severity', 'Unknown')
            confidence = prediction_data.get('confidence', 'Unknown')
            prediction_data['medical_insights'] = [
                f'Diagnostic assessment based on {model_type}',
                f'Risk level: {severity}',
                f'Confidence: {confidence}'
            ]
        
        if not prediction_data.get('patient_summary'):
            features = prediction_data.get('features', {})
            age = features.get('age', 'Unknown')
            sex = features.get('sex', 'Unknown')
            severity = prediction_data.get('severity', 'Unknown')
            prediction_data['patient_summary'] = f'{age}-year-old {sex.lower()} with {severity.lower()} cardiovascular risk based on diagnostic assessment.'
        
        # Generate PDF with comprehensive validation
        try:
            print(f"Generating PDF for prediction {prediction_id} from {data_source}")
            
            # Validate required fields
            required_fields = ['features', 'diagnosis', 'diagnostic_score', 'severity', 'confidence', 'model_type']
            for field in required_fields:
                if field not in prediction_data:
                    print(f"Missing required field: {field}")
                    prediction_data[field] = 'N/A'
            
            # Validate features
            if 'features' in prediction_data:
                required_features = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'serum_cholesterol', 
                                   'fasting_bs', 'resting_ecg', 'max_heart_rate', 'exercise_angina', 
                                   'st_depression', 'st_slope', 'major_vessels', 'thalassemia']
                for feature in required_features:
                    if feature not in prediction_data['features']:
                        print(f"Missing feature: {feature}")
                        prediction_data['features'][feature] = 'N/A'
            
            # Ensure all detailed analysis fields are present and valid
            if 'parameter_analysis' not in prediction_data or not prediction_data['parameter_analysis']:
                prediction_data['parameter_analysis'] = {
                    'age': {'status': 'Standard', 'interpretation': 'Age-based risk assessment'},
                    'blood_pressure': {'status': 'Standard', 'interpretation': 'Blood pressure evaluation'},
                    'cholesterol': {'status': 'Standard', 'interpretation': 'Cholesterol level assessment'},
                    'heart_rate': {'status': 'Standard', 'interpretation': 'Heart rate analysis'}
                }
            
            if 'risk_stratification' not in prediction_data or not prediction_data['risk_stratification']:
                prediction_data['risk_stratification'] = {
                    'risk_percentage': prediction_data.get('diagnostic_score', 0),
                    'risk_category': prediction_data.get('severity', 'Unknown'),
                    'risk_description': f'Based on diagnostic score of {prediction_data.get("diagnostic_score", 0)}',
                    'timeframe': '10-year cardiovascular risk estimate'
                }
            
            if 'lifestyle_recommendations' not in prediction_data or not prediction_data['lifestyle_recommendations']:
                prediction_data['lifestyle_recommendations'] = [
                    {'category': 'General Health', 'recommendations': [
                        'Maintain a healthy diet',
                        'Exercise regularly',
                        'Avoid smoking',
                        'Manage stress',
                        'Regular health check-ups'
                    ]}
                ]
            
            if 'follow_up_plan' not in prediction_data or not prediction_data['follow_up_plan']:
                prediction_data['follow_up_plan'] = {
                    'immediate_actions': ['Consult with healthcare provider'],
                    'short_term': ['Monitor health parameters'],
                    'long_term': ['Maintain healthy lifestyle']
                }
            
            if 'medical_insights' not in prediction_data or not prediction_data['medical_insights']:
                model_type = prediction_data.get('model_type', 'rule_based')
                severity = prediction_data.get('severity', 'Unknown')
                confidence = prediction_data.get('confidence', 'Unknown')
                prediction_data['medical_insights'] = [
                    f'Diagnostic assessment based on {model_type}',
                    f'Risk level: {severity}',
                    f'Confidence: {confidence}'
                ]
            
            if 'patient_summary' not in prediction_data or not prediction_data['patient_summary']:
                features = prediction_data.get('features', {})
                age = features.get('age', 'Unknown')
                sex = features.get('sex', 'Unknown')
                severity = prediction_data.get('severity', 'Unknown')
                prediction_data['patient_summary'] = f'{age}-year-old {sex.lower()} with {severity.lower()} cardiovascular risk based on diagnostic assessment.'
            
            pdf_content = generate_pdf_report(prediction_id, user_data, prediction_data)
            
            if pdf_content:
                print(f"PDF generated successfully, size: {len(pdf_content)} bytes")
                # Create response
                response = make_response(pdf_content)
                response.headers['Content-Type'] = 'application/pdf'
                response.headers['Content-Disposition'] = f'attachment; filename=heart_disease_report_{prediction_id}.pdf'
                return response
            else:
                print("PDF generation returned None")
                flash('Error generating PDF report - no content generated', 'error')
                return redirect(url_for('personal_history'))
                
        except Exception as e:
            print(f"Error generating PDF for prediction {prediction_id}: {e}")
            import traceback
            traceback.print_exc()
            flash(f'Error generating PDF report: {str(e)}', 'error')
            return redirect(url_for('personal_history'))
            
    except Exception as e:
        print(f"Error in download route: {e}")
        import traceback
        traceback.print_exc()
        flash(f'Error downloading report: {str(e)}', 'error')
        return redirect(url_for('personal_history'))

@app.route('/train_model', methods=['POST'])
def train_model():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    current_user = get_current_user()
    if not current_user or current_user.role != 'admin':
        return jsonify({'error': 'Admin privileges required'}), 403
    """Endpoint to train the machine learning model"""
    
    try:
        # Train the model
        training_result = train_ml_model()
        
        return jsonify({
            'success': training_result['success'],
            'message': training_result['message'],
            'data_count': training_result['data_count'],
            'accuracy': training_result.get('accuracy', 0),
            'training_samples': training_result.get('training_samples', 0),
            'test_samples': training_result.get('test_samples', 0),
            'feature_importance': training_result.get('feature_importance', {})
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during training: {str(e)}'
        }), 500

@app.route('/model_status')
def model_status():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    """Get the current status of the ML model"""
    
    try:
        model, label_encoders = load_trained_model()
        training_data = load_training_data()
        
        status = {
            'model_available': model is not None,
            'model_file_exists': os.path.exists(MODEL_FILE),
            'encoder_file_exists': os.path.exists(LABEL_ENCODER_FILE),
            'training_data_count': len(training_data),
            'can_train': len(training_data) >= 10,
            'last_trained': None
        }
        
        # Get model file modification time if it exists
        if os.path.exists(MODEL_FILE):
            status['last_trained'] = datetime.fromtimestamp(
                os.path.getmtime(MODEL_FILE)
            ).isoformat()
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'error': f'Error checking model status: {str(e)}'
        }), 500

@app.route('/personal_history')
def personal_history():
    """Display personal prediction history for logged-in user"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Get current user
        user = db.session.get(User, session['user_id'])
        if not user:
            flash('User not found', 'error')
            return redirect(url_for('login'))
        
        # Get page number for pagination
        page = request.args.get('page', 1, type=int)
        per_page = 10  # Show 10 predictions per page
        
        # Get user's predictions with pagination
        predictions_query = Prediction.query.filter_by(user_id=session['user_id'])\
                                          .order_by(Prediction.created_at.desc())
        
        predictions_pagination = predictions_query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        # Get summary statistics for the user
        total_predictions = Prediction.query.filter_by(user_id=session['user_id']).count()
        
        # Risk distribution for this user
        risk_distribution = {
            'low': Prediction.query.filter_by(user_id=session['user_id'])\
                                  .filter(Prediction.severity.in_(['Low', 'Low-Moderate'])).count(),
            'moderate': Prediction.query.filter_by(user_id=session['user_id'])\
                                       .filter_by(severity='Moderate').count(),
            'high': Prediction.query.filter_by(user_id=session['user_id'])\
                                   .filter(Prediction.severity.in_(['High', 'Critical'])).count()
        }
        
        # Average diagnostic score for this user
        avg_score_result = db.session.query(db.func.avg(Prediction.diagnostic_score))\
                                    .filter_by(user_id=session['user_id']).scalar()
        avg_diagnostic_score = round(avg_score_result, 2) if avg_score_result else 0
        
        # Most recent prediction
        most_recent = Prediction.query.filter_by(user_id=session['user_id'])\
                                     .order_by(Prediction.created_at.desc()).first()
        
        # Get ML model status and system information for regular users
        training_data = load_training_data()
        adaptive_thresholds = calculate_adaptive_thresholds(training_data)
        model, label_encoders = load_trained_model()
        
        # Prepare data for template
        history_data = {
            'user': user,
            'predictions': predictions_pagination,
            'total_predictions': total_predictions,
            'risk_distribution': risk_distribution,
            'avg_diagnostic_score': avg_diagnostic_score,
            'most_recent': most_recent,
            'current_page': page,
            'total_pages': predictions_pagination.pages,
            # Add statistics data for regular users
            'is_admin': False,
            'model_available': model is not None,
            'system_status': 'ML Model Active' if model else ('Learning Enabled' if adaptive_thresholds else 'Baseline Mode'),
            'adaptive_thresholds': adaptive_thresholds,
            'last_trained': None
        }
        
        # Get model file modification time if it exists
        if os.path.exists(MODEL_FILE):
            history_data['last_trained'] = datetime.fromtimestamp(
                os.path.getmtime(MODEL_FILE)
            ).isoformat()
        
        return render_template('personal_history.html', data=history_data)
        
    except Exception as e:
        flash(f'Error loading personal history: {str(e)}', 'error')
        return render_template('personal_history.html', data={
            'user': None,
            'predictions': None,
            'total_predictions': 0,
            'risk_distribution': {'low': 0, 'moderate': 0, 'high': 0},
            'avg_diagnostic_score': 0,
            'most_recent': None,
            'current_page': 1,
            'total_pages': 0
        })

@app.route('/training_stats')
def training_stats():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        current_user = get_current_user()
        is_admin = current_user and current_user.role == 'admin'
        
        # Redirect regular users to personal history (which now includes stats)
        if not is_admin:
            return redirect(url_for('personal_history'))
        
        # Get training data from database
        training_data = load_training_data()
        adaptive_thresholds = calculate_adaptive_thresholds(training_data)
        model, label_encoders = load_trained_model()
        
        if is_admin:
            # Admin view - show system-wide statistics
            recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
            recent_predictions_data = [pred.to_csv_row() for pred in recent_predictions]
            
            # Get system statistics
            total_predictions = Prediction.query.count()
            total_users = User.query.count()
            active_users = User.query.filter(User.last_login >= datetime.utcnow() - timedelta(days=7)).count()
            
            # Get risk distribution
            risk_distribution = {
                'low': Prediction.query.filter(Prediction.severity.in_(['Low', 'Low-Moderate'])).count(),
                'moderate': Prediction.query.filter_by(severity='Moderate').count(),
                'high': Prediction.query.filter(Prediction.severity.in_(['High', 'Critical'])).count()
            }
            
            stats = {
                'is_admin': True,
                'total_predictions': total_predictions,
                'total_users': total_users,
                'active_users': active_users,
                'adaptive_thresholds': adaptive_thresholds,
                'recent_predictions': recent_predictions_data,
                'risk_distribution': risk_distribution,
                'system_status': 'ML Model Active' if model else ('Learning Enabled' if adaptive_thresholds else 'Baseline Mode'),
                'model_available': model is not None,
                'can_train_model': len(training_data) >= 10,
                'last_trained': None
            }
        else:
            # User view - show user-specific statistics
            user_predictions = Prediction.query.filter_by(user_id=current_user.id).all()
            recent_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).limit(10).all()
            recent_predictions_data = [pred.to_csv_row() for pred in recent_predictions]
            
            # Get user-specific statistics
            total_predictions = len(user_predictions)
            
            # Get user's risk distribution
            risk_distribution = {
                'low': len([p for p in user_predictions if p.severity in ['Low', 'Low-Moderate']]),
                'moderate': len([p for p in user_predictions if p.severity == 'Moderate']),
                'high': len([p for p in user_predictions if p.severity in ['High', 'Critical']])
            }
            
            # Calculate user's average diagnostic score
            avg_score = sum(p.diagnostic_score for p in user_predictions) / len(user_predictions) if user_predictions else 0
            
            stats = {
                'is_admin': False,
                'user_name': current_user.full_name or current_user.username,
                'total_predictions': total_predictions,
                'avg_diagnostic_score': round(avg_score, 2),
                'adaptive_thresholds': adaptive_thresholds,
                'recent_predictions': recent_predictions_data,
                'risk_distribution': risk_distribution,
                'system_status': 'ML Model Active' if model else ('Learning Enabled' if adaptive_thresholds else 'Baseline Mode'),
                'model_available': model is not None,
                'can_train_model': len(training_data) >= 10,
                'last_trained': None
            }
        
        # Get model file modification time if it exists
        if os.path.exists(MODEL_FILE):
            stats['last_trained'] = datetime.fromtimestamp(
                os.path.getmtime(MODEL_FILE)
            ).isoformat()
        
        return render_template('training_stats.html', stats=stats)
        
    except Exception as e:
        flash(f'Error loading training statistics: {str(e)}', 'error')
        return render_template('training_stats.html', stats={
            'is_admin': False,
            'total_predictions': 0,
            'adaptive_thresholds': None,
            'recent_predictions': [],
            'risk_distribution': {'low': 0, 'moderate': 0, 'high': 0},
            'system_status': 'Error',
            'model_available': False,
            'can_train_model': False,
            'last_trained': None
        })

@app.route('/admin_dashboard')
@require_admin
def admin_dashboard():
    """Admin dashboard to view all users and their details"""
    try:
        # Get all users with their prediction counts
        users = User.query.all()
        user_data = []
        
        for user in users:
            prediction_count = Prediction.query.filter_by(user_id=user.id).count()
            last_prediction = Prediction.query.filter_by(user_id=user.id)\
                                            .order_by(Prediction.created_at.desc()).first()
            
            user_data.append({
                'user': user,
                'prediction_count': prediction_count,
                'last_prediction': last_prediction,
                'is_online': user.last_login and (datetime.utcnow() - user.last_login).seconds < 3600  # Online if logged in within last hour
            })
        
        # Get system statistics
        total_users = User.query.count()
        total_predictions = Prediction.query.count()
        active_users = User.query.filter(User.last_login >= datetime.utcnow() - timedelta(days=7)).count()
        
        # Get recent activity
        recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
        
        admin_data = {
            'users': user_data,
            'total_users': total_users,
            'total_predictions': total_predictions,
            'active_users': active_users,
            'recent_predictions': recent_predictions
        }
        
        return render_template('admin_dashboard.html', data=admin_data)
        
    except Exception as e:
        flash(f'Error loading admin dashboard: {str(e)}', 'error')
        return render_template('admin_dashboard.html', data={
            'users': [],
            'total_users': 0,
            'total_predictions': 0,
            'active_users': 0,
            'recent_predictions': []
        })

@app.route('/admin/user/<int:user_id>')
@require_admin
def admin_user_details(user_id):
    """View detailed information about a specific user"""
    try:
        user = User.query.get_or_404(user_id)
        
        # Get user's predictions with pagination
        page = request.args.get('page', 1, type=int)
        per_page = 10
        
        predictions_query = Prediction.query.filter_by(user_id=user_id)\
                                          .order_by(Prediction.created_at.desc())
        
        predictions_pagination = predictions_query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        # Get user statistics
        total_predictions = Prediction.query.filter_by(user_id=user_id).count()
        
        risk_distribution = {
            'low': Prediction.query.filter_by(user_id=user_id)\
                                  .filter(Prediction.severity.in_(['Low', 'Low-Moderate'])).count(),
            'moderate': Prediction.query.filter_by(user_id=user_id)\
                                       .filter_by(severity='Moderate').count(),
            'high': Prediction.query.filter_by(user_id=user_id)\
                                   .filter(Prediction.severity.in_(['High', 'Critical'])).count()
        }
        
        avg_score_result = db.session.query(db.func.avg(Prediction.diagnostic_score))\
                                    .filter_by(user_id=user_id).scalar()
        avg_diagnostic_score = round(avg_score_result, 2) if avg_score_result else 0
        
        user_details = {
            'user': user,
            'predictions': predictions_pagination,
            'total_predictions': total_predictions,
            'risk_distribution': risk_distribution,
            'avg_diagnostic_score': avg_diagnostic_score,
            'current_page': page,
            'total_pages': predictions_pagination.pages
        }
        
        return render_template('admin_user_details.html', data=user_details)
        
    except Exception as e:
        flash(f'Error loading user details: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
