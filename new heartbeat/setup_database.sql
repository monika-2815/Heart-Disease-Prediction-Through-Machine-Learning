-- Heart Disease Predictor Database Setup Script (Fixed)
-- Run this script in phpMyAdmin or MySQL command line

-- Create database
CREATE DATABASE IF NOT EXISTS heart_disease_predictor 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- Use the database
USE heart_disease_predictor;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    INDEX idx_username (username),
    INDEX idx_email (email),
    INDEX idx_role (role)
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    age INT NOT NULL,
    sex VARCHAR(10) NOT NULL,
    chest_pain_type INT NOT NULL,
    resting_bp INT NOT NULL,
    serum_cholesterol INT NOT NULL,
    fasting_bs VARCHAR(20) NOT NULL,
    resting_ecg INT NOT NULL,
    max_heart_rate INT NOT NULL,
    exercise_angina VARCHAR(10) NOT NULL,
    st_depression DECIMAL(5,2) NOT NULL,
    st_slope INT NOT NULL,
    major_vessels INT NOT NULL,
    thalassemia INT NOT NULL,
    diagnostic_score INT NOT NULL,
    diagnosis TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    confidence VARCHAR(50) NOT NULL,
    parameter_analysis JSON,
    risk_stratification JSON,
    lifestyle_recommendations JSON,
    follow_up_plan JSON,
    medical_insights JSON,
    patient_summary JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_type VARCHAR(20) DEFAULT 'rule_based',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_severity (severity),
    INDEX idx_diagnostic_score (diagnostic_score)
);

-- Create ml_models table (Fixed - using backquotes for reserved keywords)
CREATE TABLE IF NOT EXISTS ml_models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) DEFAULT '1.0',
    accuracy DECIMAL(5,4),
    `precision` DECIMAL(5,4),
    `recall` DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    training_samples INT,
    test_samples INT,
    feature_importance JSON,
    model_file_path VARCHAR(255),
    encoder_file_path VARCHAR(255),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trained_by INT,
    FOREIGN KEY (trained_by) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_model_name (model_name),
    INDEX idx_is_active (is_active)
);

-- Create system_stats table
CREATE TABLE IF NOT EXISTS system_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stat_date DATE DEFAULT (CURRENT_DATE),
    total_predictions INT DEFAULT 0,
    total_users INT DEFAULT 0,
    active_users INT DEFAULT 0,
    low_risk_predictions INT DEFAULT 0,
    moderate_risk_predictions INT DEFAULT 0,
    high_risk_predictions INT DEFAULT 0,
    critical_risk_predictions INT DEFAULT 0,
    rule_based_predictions INT DEFAULT 0,
    ml_model_predictions INT DEFAULT 0,
    avg_diagnostic_score DECIMAL(5,2),
    avg_confidence DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_stat_date (stat_date),
    INDEX idx_stat_date (stat_date)
);

-- Insert default users (with proper password hashes)
INSERT IGNORE INTO users (username, email, password_hash, full_name, role) VALUES
('admin', 'admin@heartpredictor.com', 'pbkdf2:sha256:260000$example$hash', 'System Administrator', 'admin'),
('doctor', 'doctor@heartpredictor.com', 'pbkdf2:sha256:260000$example$hash', 'Dr. Medical Professional', 'doctor'),
('user1', 'user1@heartpredictor.com', 'pbkdf2:sha256:260000$example$hash', 'Test User', 'user');

-- Note: The password hashes above are examples. The actual application will generate proper hashes.

-- Create views for easier data access
CREATE OR REPLACE VIEW user_predictions AS
SELECT 
    p.id,
    p.user_id,
    u.username,
    u.full_name,
    p.age,
    p.sex,
    p.diagnostic_score,
    p.diagnosis,
    p.severity,
    p.confidence,
    p.created_at,
    p.model_type
FROM predictions p
JOIN users u ON p.user_id = u.id
ORDER BY p.created_at DESC;

CREATE OR REPLACE VIEW daily_stats AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_predictions,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(diagnostic_score) as avg_score,
    COUNT(CASE WHEN severity IN ('Low', 'Low-Moderate') THEN 1 END) as low_risk,
    COUNT(CASE WHEN severity = 'Moderate' THEN 1 END) as moderate_risk,
    COUNT(CASE WHEN severity IN ('High', 'Critical') THEN 1 END) as high_risk
FROM predictions
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Create indexes for better performance
CREATE INDEX idx_predictions_user_date ON predictions(user_id, created_at);
CREATE INDEX idx_predictions_severity_date ON predictions(severity, created_at);
CREATE INDEX idx_users_created_at ON users(created_at);

-- Show table information
SHOW TABLES;
DESCRIBE users;
DESCRIBE predictions;
DESCRIBE ml_models;
DESCRIBE system_stats;

-- Display success message
SELECT 'Database setup completed successfully!' as status;
