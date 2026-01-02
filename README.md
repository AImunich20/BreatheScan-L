# BreatheScan-L: AI-Powered Breath Analysis System

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)

---

## 🔬 Overview

**BreatheScan-L** is an advanced health diagnostic system that uses breath analysis combined with artificial intelligence to assess potential health risks. The system analyzes:

- **Breath Color**: Captures and processes images of breath indicators to detect color changes
- **Gas Sensors**: Integrates IoT sensors (MQ sensors) for real-time gas concentration detection
- **Medical Questionnaire**: Collects patient health history through structured questionnaires
- **Machine Learning**: Uses logistic regression models to predict health risk levels

### Key Applications
- Early detection of liver disease indicators
- Breath odor and health assessment
- Non-invasive health screening
- Personalized health risk analysis

---

## ✨ Features

### 1. **User Management System**
- User registration and authentication
- Secure password hashing with bcrypt
- Session management
- Role-based access (User/Admin)
- User profile management

### 2. **Breath Analysis Engine (Breathescan.py)**
- Image-based breath color analysis using OpenCV and HSV color space
- Multi-sensor data integration (MQ135, MQ136, MQ137, MQ138)
- Feature extraction pipeline with standardization
- Machine learning-based risk classification
- Real-time analysis with model versioning

### 3. **HarmoMed Medical Module (HarmoMed.py)**
- Advanced image processing for medical diagnostics
- Binary image analysis for visual health indicators
- Time-series data logging and tracking
- Historical analysis and trend detection
- Integration with medical questionnaires

### 4. **AI Agent with Memory (basic_ai_agent_with_memory.py)**
- Intelligent chatbot with conversation history
- Context-aware responses using LangChain
- Integration with Ollama for local LLM deployment
- Memory persistence in JSON format
- Thai language support with pythainlp

### 5. **Web Dashboard**
- Responsive Flask-based web interface
- User dashboard for viewing analysis results
- Admin dashboard for data management
- Real-time data visualization
- Mobile-friendly design with Tailwind CSS
- Knowledge base and research information

### 6. **IoT Sensor Integration**
- Arduino-compatible sensor module (sensor.ino)
- Real-time gas sensor data collection
- CSV-based data logging
- Configurable sensor calibration

---

## 🛠 Technology Stack

### Backend
- **Framework**: Flask 3.1.2
- **Machine Learning**: scikit-learn, joblib
- **Data Processing**: pandas, numpy
- **Computer Vision**: OpenCV (cv2)
- **AI Integration**: LangChain 0.1.0, LangChain Ollama 0.0.1
- **Security**: werkzeug

### Frontend
- **CSS**: Tailwind CSS 3.0
- **JavaScript**: Vanilla JS with responsive design
- **Template Engine**: Jinja2 (Flask)

### IoT & Hardware
- **Sensor SDK**: ultralytics 8.0.230
- **Arduino IDE**: sensor.ino for microcontroller

### Database
- **Format**: CSV-based data storage
- **Image Storage**: File-based with structured directories

### AI/ML
- **Model Type**: Logistic Regression
- **LLM Integration**: Ollama (local deployment)
- **NLP**: pythainlp for Thai language processing

### System Monitoring
- **Process Management**: psutil
- **Visualization**: matplotlib

---

## 📁 Project Structure

```
BreatheScan-L/
├── app.py                           # Main Flask application
├── Breathescan.py                   # Breath analysis engine
├── HarmoMed.py                      # Medical analysis module
├── basic_ai_agent_with_memory.py   # AI chatbot with memory
├── requirements.txt                 # Python dependencies
├── runtime.txt                      # Python version specification
├── sensor.csv                       # Sensor data log
├── user.csv                         # User database
├── image_log.csv                    # Image metadata log
│
├── HarmoMed/                        # HarmoMed module folder
│   ├── HarmoMed.py
│   ├── binary_HarmoMed_img.py
│   ├── binary_path_img.py
│   ├── test_lir_HarmoMed.py
│   └── static/
│       ├── uploads/                 # Uploaded images
│       └── results/                 # Analysis results
│
├── templates/                       # HTML templates
│   ├── home.html
│   ├── HarmoMed.html
│   ├── admin_dashboard.html
│   ├── knowledge.html
│   ├── research.html
│   ├── about.html
│   ├── contact.html
│   ├── test.html
│   └── components/
│       ├── navbar.html
│       ├── chatbot.html
│       ├── login.html
│       └── user_menu.html
│
├── static/                          # Static assets
│   ├── css/
│   │   ├── style.css
│   │   ├── tailwind.css
│   │   ├── input.css
│   │   ├── postcss.config.js
│   │   └── tailwind.config.js
│   ├── js/
│   │   ├── scripts.js
│   │   ├── admin.js
│   │   ├── mobile.js
│   │   └── user_menu.js
│   ├── fonts/
│   ├── uploads/                     # User uploads
│   └── results/                     # Analysis results
│
├── user/                            # User data directory
│   ├── admin/
│   ├── 20/
│   └── munich/
│
├── memory/                          # AI agent memory
│   ├── chat_history.jsonl
│   └── summary.json
│
├── sensor/                          # IoT sensor code
│   └── sensor.ino
│
└── images/                          # Image storage
```

---

## 💾 Installation

### Prerequisites
- **Python 3.9** or higher
- **pip** (Python package manager)
- **Git** (optional, for cloning)
- **Arduino IDE** (optional, for sensor programming)

### Step 1: Clone or Download the Repository

```bash
# Using Git
git clone <repository-url>
cd BreatheScan-L

# Or extract the downloaded ZIP file
unzip BreatheScan-L.zip
cd BreatheScan-L
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- Flask web framework
- Machine learning libraries (scikit-learn, joblib)
- Data processing (pandas, numpy)
- Computer vision (opencv-python)
- AI/LLM integration (langchain, ollama)
- Image processing (pillow)
- System monitoring (psutil)
- Thai language processing (pythainlp)
- And more...

### Step 5: Verify Installation

```bash
python -c "import flask, cv2, pandas, numpy; print('✓ All core dependencies installed successfully')"
```

---

## ⚙️ Configuration

### 1. Flask Application Settings

Edit `app.py` to configure:
```python
app.secret_key = "your-secret-key"  # Change to a strong secret key
UPLOAD_FOLDER = "uploads"            # Path for file uploads
USER_DIR = "user"                    # Path for user data
```

### 2. Database Configuration

The application uses CSV files for data storage:
- `user.csv` - User accounts and credentials
- `sensor.csv` - IoT sensor readings
- `image_log.csv` - Breath image metadata
- `user/{username}/questionnaires/` - Medical questionnaires
- `user/{username}/uploads/` - User analysis data

### 3. Machine Learning Models

Models are stored as pickle files:
- `risk_model.pkl` - Trained logistic regression model
- `scaler.pkl` - Feature scaler for data normalization

Train new models by running:
```bash
python test_lir_HarmoMed.py
```

### 4. AI Agent Configuration

Configure the AI chatbot in `basic_ai_agent_with_memory.py`:
```python
# LLM Model settings
model = "mistral"  # or your preferred Ollama model
temperature = 0.7

# Memory settings
memory_file = "memory/chat_history.jsonl"
summary_file = "memory/summary.json"
```

### 5. IoT Sensor Configuration

For Arduino-based sensors, configure in `sensor/sensor.ino`:
- Adjust sensor pin assignments
- Calibrate MQ sensor constants
- Set baud rate for serial communication

---

## 🚀 Running the Application

### Method 1: Direct Python Execution (Development)

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the Flask application
python app.py
```

**Output:**
```
 * Running on http://127.0.0.1:5000
 * Debug mode: off
```

### Method 2: Using Flask Development Server with Debug Mode

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
```

### Method 3: Production Deployment with Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Method 4: Using Docker (Optional)

```bash
# Build Docker image
docker build -t breathescan-l .

# Run container
docker run -p 5000:5000 breathescan-l
```

### Step-by-Step Running Process

1. **Activate Virtual Environment**
   ```bash
   source venv/bin/activate
   ```

2. **Start the Application**
   ```bash
   python app.py
   ```

3. **Open in Browser**
   - Navigate to `http://localhost:5000`
   - Default admin credentials: username: `admin`, password: (check user.csv)

4. **Access Features**
   - **Home Page**: `/` - Dashboard and overview
   - **Breath Analysis**: `/HarmoMed` - Upload and analyze breath images
   - **Questionnaire**: `/api/knowledge/save_qs` - Medical history form
   - **Admin Panel**: `/admin` - User and data management
   - **Chatbot**: Integrated in the web interface
   - **Knowledge Base**: `/knowledge` - Research and information

---

## 📡 API Endpoints

### Authentication
- `POST /login` - User login
- `POST /register` - User registration
- `GET /logout` - User logout
- `POST /api/auth/check` - Session verification

### Breath Analysis
- `POST /api/breathescan/analyze` - Analyze breath image and sensor data
- `GET /api/breathescan/history` - Get analysis history
- `POST /api/breathescan/upload` - Upload breath image

### Medical Data
- `POST /api/knowledge/save_qs` - Save questionnaire responses
- `GET /api/knowledge/get_qs` - Retrieve questionnaire data
- `POST /api/knowledge/analyze` - Generate analysis report

### User Management
- `GET /api/user/profile` - Get user profile
- `POST /api/user/update` - Update user information
- `GET /api/user/data` - Retrieve user analysis data

### Admin Functions
- `GET /admin` - Admin dashboard
- `POST /admin/users` - Manage users
- `DELETE /admin/users/<user_id>` - Delete user
- `GET /admin/reports` - Generate system reports

### Chatbot
- `POST /api/chat` - Send message to AI agent
- `GET /api/chat/history` - Get conversation history

---

## 📖 Usage Guide

### For Regular Users

#### 1. Create an Account
- Go to the login page
- Click "Register"
- Enter username, email, and password
- Submit to create account

#### 2. Perform Breath Analysis
- Go to "Breath Analysis" section
- Upload a clear breath indicator image
- Connect IoT sensors and record readings
- Complete health questionnaire
- Submit for analysis

#### 3. View Results
- Results appear on dashboard
- Download analysis reports
- Track analysis history
- Compare results over time

#### 4. Chat with AI Agent
- Use integrated chatbot
- Ask health-related questions
- Get personalized recommendations
- View conversation history

### For Administrators

#### 1. Access Admin Dashboard
- Login as admin user
- Navigate to `/admin`
- View system statistics and user data

#### 2. User Management
- View all registered users
- Manage user roles and permissions
- Delete inactive accounts
- Export user data

#### 3. Data Analysis
- Generate system reports
- Analyze usage patterns
- Monitor system performance
- Export analysis results

#### 4. System Configuration
- Update application settings
- Configure sensor calibration
- Manage AI model parameters
- Backup system data

---

## 🔧 Troubleshooting

### Issue 1: Module Not Found Error
**Problem**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Issue 2: Port Already in Use
**Problem**: `Address already in use`

**Solution**:
```bash
# Find and kill process using port 5000
lsof -i :5000          # On macOS/Linux
netstat -ano | findstr :5000  # On Windows

# Kill the process
kill -9 <PID>          # On macOS/Linux
taskkill /PID <PID> /F # On Windows

# Or use a different port
python app.py --port 8000
```

### Issue 3: Permission Denied
**Problem**: `Permission denied` when running scripts

**Solution**:
```bash
# Make script executable
chmod +x app.py

# Run with python explicitly
python app.py
```

### Issue 4: CSV File Not Found
**Problem**: `FileNotFoundError: user.csv not found`

**Solution**:
```bash
# Initialize default CSV files
python << 'EOF'
import csv
import os

# Create user.csv
with open('user.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['username', 'password_hash', 'email', 'role'])

# Create sensor.csv
with open('sensor.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'MQ135', 'MQ136', 'MQ137', 'MQ138'])

# Create image_log.csv
with open('image_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'time', 'day'])

print("CSV files initialized successfully")
EOF
```

### Issue 5: OpenCV/CV2 Error
**Problem**: `ImportError: cannot import name cv2`

**Solution**:
```bash
# Reinstall opencv-python
pip uninstall opencv-python -y
pip install opencv-python

# Or if that fails, try opencv-contrib-python
pip install opencv-contrib-python
```

### Issue 6: Tensor/NumPy Version Conflict
**Problem**: `numpy version error` or `tensor shape mismatch`

**Solution**:
```bash
# Install compatible numpy version
pip install "numpy<2" --force-reinstall

# Verify installation
python -c "import numpy; print(numpy.__version__)"
```

### Issue 7: LLM/Ollama Not Found
**Problem**: `ConnectionRefusedError` when trying to use chatbot

**Solution**:
```bash
# Install Ollama from https://ollama.ai
# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull mistral

# Test connection
python -c "from ollama import Client; print('Ollama working!')"
```

---

## 🔒 Security Recommendations

1. **Change Default Secret Key**
   ```python
   # In app.py, use a strong random key
   app.secret_key = os.urandom(24).hex()
   ```

2. **Use Environment Variables**
   ```bash
   export FLASK_SECRET_KEY="your-secret-key"
   export DATABASE_URL="your-database-url"
   ```

3. **Enable HTTPS in Production**
   - Use SSL/TLS certificates
   - Configure proper CORS headers

4. **Validate User Inputs**
   - Sanitize file uploads
   - Validate form submissions
   - Implement rate limiting

5. **Regular Backups**
   ```bash
   # Backup user data
   tar -czf backup_$(date +%Y%m%d).tar.gz user/ memory/
   ```

---

## 📊 System Requirements

### Minimum Specifications
- **CPU**: Intel Core i5 / AMD Ryzen 5 or equivalent
- **RAM**: 4 GB
- **Storage**: 2 GB free space
- **Python**: 3.9+

### Recommended Specifications
- **CPU**: Intel Core i7 / AMD Ryzen 7 or equivalent
- **RAM**: 8+ GB
- **Storage**: 10+ GB SSD
- **GPU**: NVIDIA CUDA 11.0+ (for faster ML inference)

### Network
- Internet connection for initial setup
- Local network for sensor connectivity

---

## 📝 License & Attribution

This project integrates multiple open-source libraries. See individual package licenses for details.

---

## 👨‍💼 Support & Contact

For issues, questions, or contributions:
- Check the troubleshooting section above
- Review project documentation
- Contact admin or check about.html for contact information

---

## 🎯 Future Enhancements

- [ ] Database migration to PostgreSQL
- [ ] Mobile application development
- [ ] Advanced ML models (Deep Learning)
- [ ] Real-time data streaming
- [ ] Multi-language support expansion
- [ ] Enhanced security features
- [ ] API documentation with Swagger/OpenAPI

---

**Last Updated**: January 2, 2026  
**Version**: 2.0.0
