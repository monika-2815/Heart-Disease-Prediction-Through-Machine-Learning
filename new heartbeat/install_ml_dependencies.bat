@echo off
echo Installing Machine Learning Dependencies for Heart Disease Predictor...
echo.

echo Installing scikit-learn, pandas, numpy, and pickle-mixin...
pip install scikit-learn>=1.3.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install pickle-mixin>=1.0.2

echo.
echo Installation complete!
echo.
echo You can now:
echo 1. Run the application with: python app.py
echo 2. Use the Training Statistics page to train ML models
echo 3. The system will automatically use trained models for predictions
echo.
pause
