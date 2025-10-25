# Heart Disease Predictor

A comprehensive web application for predicting heart disease risk using machine learning and adaptive learning algorithms. Built with Flask, MySQL database, and modern web technologies.

## Features

- 🔐 **User Authentication**: Secure login and registration system
- 🤖 **AI-Powered Diagnostics**: Machine learning models for heart disease prediction
- 📊 **Personal History**: Track your prediction history and health trends
- 🧠 **Adaptive Learning**: System learns from historical data to improve accuracy
- 📱 **Responsive Design**: Works on all devices
- 🗄️ **Database Integration**: Full MySQL database with phpMyAdmin support

## Installation & Setup

### Prerequisites
- Python 3.7+
- XAMPP Server (for MySQL database)

### Quick Setup

1. **Install XAMPP**
   - Download and install [XAMPP](https://www.apachefriends.org/download.html)
   - Start Apache and MySQL services
   - Open phpMyAdmin at `http://localhost/phpmyadmin`

2. **Create Database**
   - Create database: `heart_disease_predictor`
   - Import `setup_database.sql` in phpMyAdmin

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize Database**
   ```bash
   python init_db.py
   ```

5. **Run Application**
```bash
python app.py
6. **Access Application**
   - Open browser: `http://localhost:5000`

## Default Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | admin | admin123 |
| Doctor | doctor | doctor123 |
| User | user1 | password123 |

## Project Structure

```
heart-disease-predictor/
├── app.py                          # Main Flask application
├── config.py                       # Database configuration
├── models.py                       # Database models
├── init_db.py                      # Database initialization
├── setup_database.sql              # Database schema
├── requirements.txt                # Python dependencies
└── templates/                      # HTML templates
    ├── home.html                   # Home page
    ├── predict.html                # Prediction form
    ├── personal_history.html       # User history
    ├── login.html                  # Login page
    ├── register.html               # Registration page
    └── training_stats.html         # Statistics page
```

## How It Works

1. **Login/Register**: Create account or use demo credentials
2. **Make Prediction**: Fill medical parameters form
3. **Get Results**: Receive detailed diagnostic assessment
4. **View History**: Track your prediction history
5. **Monitor Stats**: View system performance and analytics

## Medical Parameters

The application analyzes 13 key medical parameters:
- Age, Gender, Chest Pain Type
- Blood Pressure, Cholesterol, Blood Sugar
- ECG Results, Heart Rate, Exercise Angina
- ST Depression, ST Slope, Major Vessels, Thalassemia

## Database Features

- **User Management**: Role-based authentication
- **Prediction Storage**: All results saved in MySQL
- **ML Model Training**: Train models from historical data
- **Analytics**: Comprehensive statistics and reporting
- **Data Migration**: Automatic CSV to database migration

## Configuration

Database settings in `config.py`:
```python
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_USERNAME = 'root'
MYSQL_PASSWORD = ''
MYSQL_DATABASE = 'heart_disease_predictor'
```

## Troubleshooting

### Common Issues

**Database Connection Error**
- Ensure XAMPP MySQL service is running
- Check database credentials in `config.py`

**Table Doesn't Exist**
- Run `setup_database.sql` in phpMyAdmin
- Or run `python init_db.py`

**Import Error**
```bash
pip install PyMySQL Flask-SQLAlchemy cryptography
```

## Contributing

Feel free to contribute by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Submitting pull requests

## License

This project is developed for educational and research purposes.

---

**Designed and Developed by Monika P** ❤️
