@echo off
echo Starting Heart Disease Predictor Application...
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting the application...
echo.
echo The application will be available at: http://localhost:5000
echo Press Ctrl+C to stop the application
echo.
python app.py
pause
