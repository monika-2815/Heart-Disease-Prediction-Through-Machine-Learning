# Database Integration Setup Guide

## 🎯 Overview

The Heart Disease Predictor application has been successfully integrated with MySQL database through XAMPP server and phpMyAdmin. This guide provides step-by-step instructions for setting up and using the database features.

## 🚀 Quick Setup (5 Minutes)

### Step 1: Install XAMPP
1. Download [XAMPP](https://www.apachefriends.org/download.html)
2. Install and start XAMPP Control Panel
3. Start **Apache** and **MySQL** services
4. Open phpMyAdmin: `http://localhost/phpmyadmin`

### Step 2: Create Database
1. In phpMyAdmin, click "New"
2. Database name: `heart_disease_predictor`
3. Collation: `utf8mb4_unicode_ci`
4. Click "Create"

### Step 3: Import Schema
1. Select `heart_disease_predictor` database
2. Click "Import" tab
3. Choose file: `setup_database.sql`
4. Click "Go"

### Step 4: Install Dependencies
```bash
# Run the automated installer
install_database_dependencies.bat

# Or manually
pip install -r requirements.txt
```

### Step 5: Initialize Database
```bash
python init_db.py
```

### Step 6: Start Application
```bash
python app.py
```

## 🔐 Default Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | admin | admin123 |
| Doctor | doctor | doctor123 |
| User | user1 | password123 |

## 📊 Database Schema

### Tables Created:
- **users**: User authentication and profiles
- **predictions**: Heart disease prediction data
- **ml_models**: Machine learning model metadata
- **system_stats**: System analytics and statistics

### Key Features:
- **Foreign Key Relationships**: Proper data integrity
- **Indexes**: Optimized for performance
- **JSON Storage**: Complex data structures
- **Timestamps**: Automatic date tracking
- **Role-based Access**: Admin, Doctor, User roles

## 🔧 Configuration

Database settings in `config.py`:
```python
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_USERNAME = 'root'
MYSQL_PASSWORD = ''  # Change if you set a password
MYSQL_DATABASE = 'heart_disease_predictor'
```

## 📈 New Features Added

### 1. User Management
- Database-based authentication
- Password hashing with Werkzeug
- Role-based permissions
- User profiles and metadata

### 2. Data Storage
- All predictions stored in database
- Enhanced data with JSON fields
- Automatic timestamps
- User association

### 3. Analytics
- Real-time statistics
- Risk distribution analysis
- User activity tracking
- System performance metrics

### 4. Machine Learning
- Database-driven training data
- Model metadata storage
- Performance tracking
- Version management

## 🛠️ Troubleshooting

### Common Issues:

#### 1. Connection Error
```
Error: Can't connect to MySQL server
```
**Fix:** Ensure XAMPP MySQL is running

#### 2. Table Missing
```
Error: Table doesn't exist
```
**Fix:** Run `python init_db.py`

#### 3. Permission Denied
```
Error: Access denied for user
```
**Fix:** Check MySQL password in `config.py`

#### 4. Import Error
```
Error: No module named 'pymysql'
```
**Fix:** Run `pip install PyMySQL Flask-SQLAlchemy cryptography`

## 📋 Migration from CSV

The system automatically migrates existing CSV data:
- Preserves all historical predictions
- Maintains data integrity
- Creates user associations
- Updates timestamps

## 🔍 Database Queries

### View All Predictions
```sql
SELECT * FROM user_predictions ORDER BY created_at DESC;
```

### Daily Statistics
```sql
SELECT * FROM daily_stats ORDER BY date DESC;
```

### User Activity
```sql
SELECT username, COUNT(*) as predictions 
FROM user_predictions 
GROUP BY username 
ORDER BY predictions DESC;
```

## 🎯 Benefits

### For Users:
- Persistent login sessions
- Personal prediction history
- Role-based access
- Enhanced security

### For Administrators:
- Complete data analytics
- User management
- System monitoring
- Backup capabilities

### For Developers:
- Scalable architecture
- Professional database design
- Easy maintenance
- Extensible framework

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Verify XAMPP services are running
3. Ensure database credentials are correct
4. Run `python init_db.py` to reset database

## 🎉 Success!

Your Heart Disease Predictor application is now fully integrated with MySQL database! Enjoy the enhanced features and professional-grade data management capabilities.
