# BreatheScan-L: AI-Powered Breath Analysis System

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Configuration](#configuration)
* [Running the Application](#running-the-application)
* [API Endpoints](#api-endpoints)
* [Usage Guide](#usage-guide)
* [Troubleshooting](#troubleshooting)

---

## Overview

**BreatheScan-L** is an advanced health diagnostic system that uses breath analysis combined with artificial intelligence to assess potential health risks. The system analyzes:

* Breath Color: Captures and processes images of breath indicators to detect color changes
* Gas Sensors: Integrates IoT sensors (MQ sensors) for real-time gas concentration detection
* Medical Questionnaire: Collects patient health history through structured questionnaires
* Machine Learning: Uses logistic regression models to predict health risk levels

### Key Applications

* Early detection of liver disease indicators
* Breath odor and health assessment
* Non-invasive health screening
* Personalized health risk analysis

---

## Features

### 1. User Management System

* User registration and authentication
* Secure password hashing with bcrypt
* Session management
* Role-based access (User/Admin)
* User profile management

### 2. Breath Analysis Engine (Breathescan.py)

* Image-based breath color analysis using OpenCV and HSV color space
* Multi-sensor data integration (MQ135, MQ136, MQ137, MQ138)
* Feature extraction pipeline with standardization
* Machine learning-based risk classification
* Real-time analysis with model versioning

### 3. HarmoMed Medical Module (HarmoMed.py)

* Advanced image processing for medical diagnostics
* Binary image analysis for visual health indicators
* Time-series data logging and tracking
* Historical analysis and trend detection
* Integration with medical questionnaires

### 4. AI Agent with Memory (basic_ai_agent_with_memory.py)

* Intelligent chatbot with conversation history
* Context-aware responses using LangChain
* Integration with Ollama for local LLM deployment
* Memory persistence in JSON format
* Thai language support with pythainlp

### 5. Web Dashboard

* Responsive Flask-based web interface
* User dashboard for viewing analysis results
* Admin dashboard for data management
* Real-time data visualization
* Mobile-friendly design with Tailwind CSS
* Knowledge base and research information

### 6. IoT Sensor Integration

* Arduino-compatible sensor module (sensor.ino)
* Real-time gas sensor data collection
* CSV-based data logging
* Configurable sensor calibration

---

## Technology Stack

### Backend

* Framework: Flask 3.1.2
* Machine Learning: scikit-learn, joblib
* Data Processing: pandas, numpy
* Computer Vision: OpenCV (cv2)
* AI Integration: LangChain 0.1.0, LangChain Ollama 0.0.1
* Security: werkzeug

### Frontend

* CSS: Tailwind CSS 3.0
* JavaScript: Vanilla JS with responsive design
* Template Engine: Jinja2 (Flask)

### IoT & Hardware

* Sensor SDK: ultralytics 8.0.230
* Arduino IDE: sensor.ino for microcontroller

### Database

* Format: CSV-based data storage
* Image Storage: File-based with structured directories

### AI/ML

* Model Type: Logistic Regression
* LLM Integration: Ollama (local deployment)
* NLP: pythainlp for Thai language processing

### System Monitoring

* Process Management: psutil
* Visualization: matplotlib

---

## Project Structure

```
BreatheScan-L/
├── app.py
├── Breathescan.py
├── HarmoMed.py
├── basic_ai_agent_with_memory.py
├── requirements.txt
├── runtime.txt
├── sensor.csv
├── user.csv
├── image_log.csv
├── HarmoMed/
├── templates/
├── static/
├── user/
├── memory/
├── sensor/
└── images/
```

---

## Installation

### Prerequisites

* Python 3.9 or higher
* pip
* Git (optional)
* Arduino IDE (optional)

### Clone Repository

```bash
git clone https://github.com/AImunich20/BreatheScan-L.git
```

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate |(window)
source venv/bin/activate |(linux)
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

### Flask Settings

```python
app.secret_key = "your-secret-key"
UPLOAD_FOLDER = "uploads"
USER_DIR = "user"
```

### Machine Learning Models

* risk_model.pkl
* scaler.pkl

Train models:

```bash
python test_lir_HarmoMed.py
```

### AI Agent

```python
model = "mistral"
temperature = 0.7
```

---

## Running the Application

```bash
python app.py
```

Open browser at:

```
http://127.0.0.1:5000
```

---

## API Endpoints

### Authentication

* POST /login
* POST /register
* GET /logout

### Breath Analysis

* POST /api/breathescan/analyze
* GET /api/breathescan/history

### Chatbot

* POST /api/chat
* GET /api/chat/history

---

## Usage Guide

### Users

* Register account
* Upload breath image
* Submit sensor data
* View results

### Admin

* Manage users
* View reports
* Configure system

---

## Troubleshooting

### Flask Not Found

```bash
pip install flask
```

### Port in Use

```bash
netstat -ano | findstr :5000
```

---

## License

Open-source components follow their respective licenses.

---

## Future Enhancements

* Database migration
* Mobile application
* Advanced ML models

---

Last Updated: January 2, 2026
Version: 2.0.0
