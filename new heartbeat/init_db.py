#!/usr/bin/env python3
"""
Database initialization script for Heart Disease Predictor
Run this script to create the database and tables
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from models import User, Prediction, MLModel, SystemStats
from config import config

def create_database():
    """Create database and tables"""
    app = create_app()
    
    with app.app_context():
        try:
            # Create all tables
            print("Creating database tables...")
            db.create_all()
            print("✅ Database tables created successfully!")
            
            # Create default admin user
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                admin_user = User(
                    username='admin',
                    email='admin@heartpredictor.com',
                    full_name='System Administrator',
                    role='admin'
                )
                admin_user.set_password('admin123')
                db.session.add(admin_user)
                print("✅ Default admin user created (username: admin, password: admin123)")
            
            # Create default doctor user
            doctor_user = User.query.filter_by(username='doctor').first()
            if not doctor_user:
                doctor_user = User(
                    username='doctor',
                    email='doctor@heartpredictor.com',
                    full_name='Dr. Medical Professional',
                    role='doctor'
                )
                doctor_user.set_password('doctor123')
                db.session.add(doctor_user)
                print("✅ Default doctor user created (username: doctor, password: doctor123)")
            
            # Create default test user
            test_user = User.query.filter_by(username='user1').first()
            if not test_user:
                test_user = User(
                    username='user1',
                    email='user1@heartpredictor.com',
                    full_name='Test User',
                    role='user'
                )
                test_user.set_password('password123')
                db.session.add(test_user)
                print("✅ Default test user created (username: user1, password: password123)")
            
            # Commit all changes
            db.session.commit()
            print("✅ Database initialization completed successfully!")
            
            # Display connection info
            print("\n📊 Database Connection Information:")
            print(f"   Host: {app.config['MYSQL_HOST']}")
            print(f"   Port: {app.config['MYSQL_PORT']}")
            print(f"   Database: {app.config['MYSQL_DATABASE']}")
            print(f"   Username: {app.config['MYSQL_USERNAME']}")
            
            print("\n👥 Default Users Created:")
            print("   Admin: admin / admin123")
            print("   Doctor: doctor / doctor123")
            print("   User: user1 / password123")
            
            print("\n🌐 Access phpMyAdmin at: http://localhost/phpmyadmin")
            print("   Database: heart_disease_predictor")
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            db.session.rollback()
            return False
    
    return True

def migrate_csv_data():
    """Migrate existing CSV data to database"""
    app = create_app()
    
    with app.app_context():
        try:
            import csv
            from datetime import datetime
            
            csv_file = 'heart_disease_predictions.csv'
            if not os.path.exists(csv_file):
                print("📄 No CSV file found to migrate")
                return True
            
            print("📄 Migrating CSV data to database...")
            
            # Get or create a default user for migrated data
            default_user = User.query.filter_by(username='admin').first()
            if not default_user:
                print("❌ No admin user found. Please run create_database() first.")
                return False
            
            migrated_count = 0
            
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Check if prediction already exists
                    existing = Prediction.query.filter_by(
                        user_id=default_user.id,
                        age=int(row['age']),
                        created_at=datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                    ).first()
                    
                    if existing:
                        continue  # Skip if already exists
                    
                    # Create new prediction
                    prediction = Prediction(
                        user_id=default_user.id,
                        age=int(row['age']),
                        sex=row['sex'],
                        chest_pain_type=int(row['chest_pain_type']),
                        resting_bp=int(row['resting_bp']),
                        serum_cholesterol=int(row['serum_cholesterol']),
                        fasting_bs=row['fasting_bs'],
                        resting_ecg=int(row['resting_ecg']),
                        max_heart_rate=int(row['max_heart_rate']),
                        exercise_angina=row['exercise_angina'],
                        st_depression=float(row['st_depression']),
                        st_slope=int(row['st_slope']),
                        major_vessels=int(row['major_vessels']),
                        thalassemia=int(row['thalassemia']),
                        diagnostic_score=int(row['diagnostic_score']),
                        diagnosis=row['diagnosis'],
                        severity=row['severity'],
                        confidence=row['confidence'],
                        created_at=datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')),
                        model_type='rule_based'
                    )
                    
                    db.session.add(prediction)
                    migrated_count += 1
            
            db.session.commit()
            print(f"✅ Migrated {migrated_count} predictions from CSV to database")
            
        except Exception as e:
            print(f"❌ Error migrating CSV data: {e}")
            db.session.rollback()
            return False
    
    return True

def show_database_info():
    """Show database information and statistics"""
    app = create_app()
    
    with app.app_context():
        try:
            print("\n📊 Database Statistics:")
            
            # Count records
            user_count = User.query.count()
            prediction_count = Prediction.query.count()
            model_count = MLModel.query.count()
            
            print(f"   Users: {user_count}")
            print(f"   Predictions: {prediction_count}")
            print(f"   ML Models: {model_count}")
            
            # Show recent predictions
            recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(5).all()
            if recent_predictions:
                print("\n📋 Recent Predictions:")
                for pred in recent_predictions:
                    print(f"   {pred.created_at.strftime('%Y-%m-%d %H:%M')} - {pred.user.username} - {pred.diagnosis}")
            
            # Show users
            users = User.query.all()
            if users:
                print("\n👥 Users:")
                for user in users:
                    print(f"   {user.username} ({user.role}) - {user.prediction_count if hasattr(user, 'prediction_count') else 0} predictions")
            
        except Exception as e:
            print(f"❌ Error getting database info: {e}")

if __name__ == '__main__':
    print("🚀 Heart Disease Predictor - Database Initialization")
    print("=" * 50)
    
    # Create database and tables
    if create_database():
        print("\n📄 Migrating existing CSV data...")
        migrate_csv_data()
        
        print("\n📊 Database Information:")
        show_database_info()
        
        print("\n✅ Database setup completed successfully!")
        print("\n🌐 Next steps:")
        print("   1. Start XAMPP server")
        print("   2. Access phpMyAdmin at http://localhost/phpmyadmin")
        print("   3. Run the Flask application: python app.py")
    else:
        print("\n❌ Database setup failed!")
        sys.exit(1)
