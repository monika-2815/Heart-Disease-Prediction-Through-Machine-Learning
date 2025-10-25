@echo off
echo Installing Database Dependencies for Heart Disease Predictor
echo ============================================================

echo.
echo Installing MySQL and Flask-SQLAlchemy dependencies...
pip install PyMySQL>=1.1.0
pip install Flask-SQLAlchemy>=3.0.0
pip install cryptography>=3.4.8

echo.
echo Installing all requirements...
pip install -r requirements.txt

echo.
echo Database dependencies installed successfully!
echo.
echo Next steps:
echo 1. Start XAMPP server
echo 2. Open phpMyAdmin at http://localhost/phpmyadmin
echo 3. Create database 'heart_disease_predictor'
echo 4. Run setup_database.sql in phpMyAdmin
echo 5. Run: python init_db.py
echo 6. Start the application: python app.py
echo.
pause
