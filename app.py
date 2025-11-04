from flask import Flask, render_template, request
import os

app = Flask(__name__)

# โฟลเดอร์สำหรับเก็บไฟล์อัปโหลด
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# หน้า Home
@app.route("/")
def home():
    return render_template("home.html")

# หน้า About
@app.route("/about")
def about():
    return render_template("about.html")

# หน้า Research
@app.route("/research")
def research():
    return render_template("research.html")

@app.route("/HarmoMed")
def HarmoMed():
    return render_template("HarmoMed.html")

# หน้า Knowledge
@app.route("/knowledge")
def knowledge():
    return render_template("knowledge.html")

# หน้า Test Model (รองรับ GET และ POST)
@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        # รับไฟล์
        eye_image = request.files.get("eyeImage")
        skin_image = request.files.get("skinImage")
        enose_csv = request.files.get("enoseCSV")
        questionnaire_csv = request.files.get("questionnaireCSV")
        breath_csv = request.files.get("breathData")

        uploaded_files = [eye_image, skin_image, enose_csv, questionnaire_csv, breath_csv]
        saved_files = []

        for f in uploaded_files:
            if f:
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
                f.save(filepath)
                saved_files.append(f.filename)

        # รับข้อมูลฟอร์ม
        age = request.form.get("age")
        gender = request.form.get("gender")
        alcohol = request.form.get("alcohol")
        sleep = request.form.get("sleep")
        exercise = request.form.get("exercise")
        diabetes = "diabetes" in request.form
        obesity = "obesity" in request.form
        hepatitis = "hepatitis" in request.form
        family_history = "family_history" in request.form

        # TODO: เรียกฟังก์ชัน AI Model ของคุณที่นี่
        # result = run_ai_model(...)

        return f"Received data: Age={age}, Gender={gender}, Alcohol={alcohol}, Files={saved_files}"

    return render_template("test.html")

# หน้า Contact
@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # โค้ดรับไฟล์และฟอร์มเหมือนเดิม
    return "Analyzing data..."


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
