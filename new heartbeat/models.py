from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()

class User(db.Model):
    """User model for authentication and user management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100), nullable=True)
    role = db.Column(db.String(20), default='user')  # user, admin, doctor
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationship with predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    """Prediction model for storing heart disease predictions"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Patient demographics
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    
    # Medical parameters
    chest_pain_type = db.Column(db.Integer, nullable=False)
    resting_bp = db.Column(db.Integer, nullable=False)
    serum_cholesterol = db.Column(db.Integer, nullable=False)
    fasting_bs = db.Column(db.String(20), nullable=False)
    resting_ecg = db.Column(db.Integer, nullable=False)
    max_heart_rate = db.Column(db.Integer, nullable=False)
    exercise_angina = db.Column(db.String(10), nullable=False)
    st_depression = db.Column(db.Float, nullable=False)
    st_slope = db.Column(db.Integer, nullable=False)
    major_vessels = db.Column(db.Integer, nullable=False)
    thalassemia = db.Column(db.Integer, nullable=False)
    
    # Prediction results
    diagnostic_score = db.Column(db.Integer, nullable=False)
    diagnosis = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.String(50), nullable=False)
    
    # Enhanced analysis (stored as JSON)
    parameter_analysis = db.Column(db.JSON, nullable=True)
    risk_stratification = db.Column(db.JSON, nullable=True)
    lifestyle_recommendations = db.Column(db.JSON, nullable=True)
    follow_up_plan = db.Column(db.JSON, nullable=True)
    medical_insights = db.Column(db.JSON, nullable=True)
    patient_summary = db.Column(db.JSON, nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    model_type = db.Column(db.String(20), default='rule_based')  # rule_based, ml_model
    
    def to_dict(self):
        """Convert prediction to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else None,
            'age': self.age,
            'sex': self.sex,
            'chest_pain_type': self.chest_pain_type,
            'resting_bp': self.resting_bp,
            'serum_cholesterol': self.serum_cholesterol,
            'fasting_bs': self.fasting_bs,
            'resting_ecg': self.resting_ecg,
            'max_heart_rate': self.max_heart_rate,
            'exercise_angina': self.exercise_angina,
            'st_depression': self.st_depression,
            'st_slope': self.st_slope,
            'major_vessels': self.major_vessels,
            'thalassemia': self.thalassemia,
            'diagnostic_score': self.diagnostic_score,
            'diagnosis': self.diagnosis,
            'severity': self.severity,
            'confidence': self.confidence,
            'parameter_analysis': self.parameter_analysis,
            'risk_stratification': self.risk_stratification,
            'lifestyle_recommendations': self.lifestyle_recommendations,
            'follow_up_plan': self.follow_up_plan,
            'medical_insights': self.medical_insights,
            'patient_summary': self.patient_summary,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'model_type': self.model_type
        }
    
    def to_csv_row(self):
        """Convert prediction to CSV row format"""
        return {
            'timestamp': self.created_at.isoformat() if self.created_at else '',
            'username': self.user.username if self.user else '',
            'age': self.age,
            'sex': self.sex,
            'chest_pain_type': self.chest_pain_type,
            'resting_bp': self.resting_bp,
            'serum_cholesterol': self.serum_cholesterol,
            'fasting_bs': self.fasting_bs,
            'resting_ecg': self.resting_ecg,
            'max_heart_rate': self.max_heart_rate,
            'exercise_angina': self.exercise_angina,
            'st_depression': self.st_depression,
            'st_slope': self.st_slope,
            'major_vessels': self.major_vessels,
            'thalassemia': self.thalassemia,
            'diagnostic_score': self.diagnostic_score,
            'diagnosis': self.diagnosis,
            'severity': self.severity,
            'confidence': self.confidence
        }
    
    def __repr__(self):
        return f'<Prediction {self.id} by User {self.user_id}>'

class MLModel(db.Model):
    """ML Model storage and metadata"""
    __tablename__ = 'ml_models'
    
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # RandomForest, etc.
    version = db.Column(db.String(20), default='1.0')
    
    # Model performance metrics
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column('precision', db.Float, nullable=True)
    recall = db.Column('recall', db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    
    # Training data info
    training_samples = db.Column(db.Integer, nullable=True)
    test_samples = db.Column(db.Integer, nullable=True)
    feature_importance = db.Column(db.JSON, nullable=True)
    
    # Model file paths
    model_file_path = db.Column(db.String(255), nullable=True)
    encoder_file_path = db.Column(db.String(255), nullable=True)
    
    # Status and metadata
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    trained_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    def to_dict(self):
        """Convert ML model to dictionary"""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'training_samples': self.training_samples,
            'test_samples': self.test_samples,
            'feature_importance': self.feature_importance,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'trained_by': self.trained_by
        }
    
    def __repr__(self):
        return f'<MLModel {self.model_name} v{self.version}>'

class SystemStats(db.Model):
    """System statistics and analytics"""
    __tablename__ = 'system_stats'
    
    id = db.Column(db.Integer, primary_key=True)
    stat_date = db.Column(db.Date, default=datetime.utcnow().date, index=True)
    
    # Daily statistics
    total_predictions = db.Column(db.Integer, default=0)
    total_users = db.Column(db.Integer, default=0)
    active_users = db.Column(db.Integer, default=0)
    
    # Risk distribution
    low_risk_predictions = db.Column(db.Integer, default=0)
    moderate_risk_predictions = db.Column(db.Integer, default=0)
    high_risk_predictions = db.Column(db.Integer, default=0)
    critical_risk_predictions = db.Column(db.Integer, default=0)
    
    # Model usage
    rule_based_predictions = db.Column(db.Integer, default=0)
    ml_model_predictions = db.Column(db.Integer, default=0)
    
    # Performance metrics
    avg_diagnostic_score = db.Column(db.Float, nullable=True)
    avg_confidence = db.Column(db.Float, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert system stats to dictionary"""
        return {
            'id': self.id,
            'stat_date': self.stat_date.isoformat() if self.stat_date else None,
            'total_predictions': self.total_predictions,
            'total_users': self.total_users,
            'active_users': self.active_users,
            'low_risk_predictions': self.low_risk_predictions,
            'moderate_risk_predictions': self.moderate_risk_predictions,
            'high_risk_predictions': self.high_risk_predictions,
            'critical_risk_predictions': self.critical_risk_predictions,
            'rule_based_predictions': self.rule_based_predictions,
            'ml_model_predictions': self.ml_model_predictions,
            'avg_diagnostic_score': self.avg_diagnostic_score,
            'avg_confidence': self.avg_confidence,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<SystemStats {self.stat_date}>'
