from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, jsonify, send_from_directory
)
import os, csv, json
from werkzeug.security import generate_password_hash, check_password_hash
from HarmoMed import HarmoMed_lir
import psutil
import time
from Breathescan import Breathescan_L
from datetime import datetime
import smtplib
from email.message import EmailMessage

print(os.environ.get("GMAIL_APP_PASSWORD"))


app = Flask(__name__)
app.secret_key = "20"

UPLOAD_FOLDER = "uploads"
USER_CSV = "user.csv"
USER_DIR = "user"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/api/knowledge/save_qs", methods=["POST"])
def save_knowledge_qs():
    if "user" not in session:
        return jsonify(success=False, msg="not logged in"), 401

    username = session["user"]["username"]
    user_dir = os.path.join(USER_DIR, username)
    qs_dir = os.path.join(user_dir, "questionnaires")
    os.makedirs(qs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qs_{timestamp}.csv"
    filepath = os.path.join(qs_dir, filename)

    fields = [
        "alcohol", "fat_food", "sulfur_food", "fructose",
        "jaundice", "carotene_food", "breath_odor",
        "abdominal_pain", "fatigue",
        "diabetes", "fatty_liver", "hepatitis", "family_history"
    ]

    row = {}
    for f in fields:
        if f in ["diabetes", "fatty_liver", "hepatitis", "family_history"]:
            row[f] = "1" if request.form.get(f) else "0"
        else:
            row[f] = request.form.get(f, "")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    return jsonify(success=True, filename=filename)

@app.route("/api/qs/list")
def list_qs():
    if "user" not in session:
        return jsonify(files=[])

    username = session["user"]["username"]
    qs_dir = os.path.join(USER_DIR, username, "questionnaires")

    if not os.path.exists(qs_dir):
        return jsonify(files=[])

    files = sorted(os.listdir(qs_dir), reverse=True)
    return jsonify(files=files)



# ---------------- INIT CSV ----------------
if not os.path.exists(USER_CSV):
    with open(USER_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "username", "password",
            "first_name", "last_name",
            "email", "phone"
        ])

def admin_required():
    return "user" in session and session["user"].get("adminst") is True

@app.route("/admin/dashboard")
def admin_dashboard():
    if not admin_required():
        return redirect(url_for("home"))
    return render_template("admin_dashboard.html")
@app.route("/api/admin/server_status")
def server_status():
    if not admin_required():
        return jsonify(error="unauthorized"), 403

    return jsonify({
        "cpu": psutil.cpu_percent(interval=0.5),
        "ram": {
            "used": psutil.virtual_memory().used // (1024**2),
            "total": psutil.virtual_memory().total // (1024**2),
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "used": psutil.disk_usage("/").used // (1024**3),
            "total": psutil.disk_usage("/").total // (1024**3),
            "percent": psutil.disk_usage("/").percent
        },
        "net": {
            "sent": psutil.net_io_counters().bytes_sent // (1024**2),
            "recv": psutil.net_io_counters().bytes_recv // (1024**2)
        }
    })

# ---------------- GLOBAL USER ----------------
@app.context_processor
def inject_user():
    return dict(user=session.get("user"))

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/research")
def research():
    return render_template("research.html")

@app.route("/HarmoMed")
def HarmoMed():
    return render_template("HarmoMed.html")

@app.route("/knowledge")
def knowledge():
    return render_template("knowledge.html")

@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")

@app.route("/send-contact", methods=["POST"])
def send_contact():
    first = request.form["first_name"]
    last = request.form["last_name"]
    email = request.form["email"]
    subject = request.form["subject"]
    message = request.form["message"]

    msg = EmailMessage()
    msg["Subject"] = f"BreatheScan-L Contact | {subject}"
    msg["From"] = "zen20.munich@gmail.com"     # ต้องตรงกับ SMTP
    msg["To"] = "natthanathron@gmail.com"
    msg["Reply-To"] = email

    msg.set_content(f"""
New Contact Message

Name: {first} {last}
Sender Email: {email}
Subject: {subject}

Message:
{message}
""")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(
                "zen20.munich@gmail.com",
                os.environ.get("GMAIL_APP_PASSWORD")
            )
            smtp.send_message(msg)

        flash("Message sent successfully!", "success")

    except Exception as e:
        print("EMAIL ERROR:", e)
        flash("Failed to send message", "error")

    return redirect(url_for("contact"))

# ---------------- LOGIN ----------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or request.form
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify(success=False, msg="Missing data")

    with open(USER_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"] == username and check_password_hash(row["password"], password):

                user_path = os.path.join(USER_DIR, username)
                info_path = os.path.join(user_path, "info.json")

                if os.path.exists(info_path):
                    with open(info_path, encoding="utf-8") as jf:
                        user_info = json.load(jf)
                else:
                    user_info = {
                        "username": username,
                        "first_name": row["first_name"],
                        "last_name": row["last_name"],
                        "email": row["email"],
                        "phone": row["phone"],
                        "medical": {},
                        "adminst": False
                    }

                # ensure boolean
                user_info["adminst"] = bool(user_info.get("adminst", False))

                session["user"] = user_info
                return jsonify(success=True)

    return jsonify(success=False, msg="Invalid username or password")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

# ---------------- USER IMAGE ----------------
@app.route("/user_image/<username>")
def user_image(username):
    image_path = os.path.join(USER_DIR, username, "profile.jpg")

    if not os.path.exists(image_path):
        return send_from_directory("static", "default-avatar.png")

    return send_from_directory(os.path.join(USER_DIR, username), "profile.jpg")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["POST"])
def register():
    data = request.form if request.form else request.get_json()

    required = [
        "username", "password",
        "first_name", "last_name",
        "email", "phone"
    ]
    if not all(k in data for k in required):
        return jsonify(success=False, msg="Incomplete data")

    username = data["username"]

    # check duplicate
    with open(USER_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["username"] == username:
                return jsonify(success=False, msg="Username already exists")

    user_path = os.path.join(USER_DIR, username)
    os.makedirs(user_path, exist_ok=True)

    # save profile image
    photo = request.files.get("photo")
    if photo and photo.filename:
        photo.save(os.path.join(user_path, "profile.jpg"))

    # save csv (hash password)
    hashed_pw = generate_password_hash(data["password"])

    with open(USER_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            username,
            hashed_pw,
            data["first_name"],
            data["last_name"],
            data["email"],
            data["phone"]
        ])

    # create info.json
    info = {
        "username": username,
        "first_name": data["first_name"],
        "last_name": data["last_name"],
        "email": data["email"],
        "phone": data["phone"],
        # "medical": {
        #     "age": "",
        #     "gender": "",
        #     "height_cm": "",
        #     "weight_kg": "",
        #     "blood_type": "",
        #     "chronic_disease": "",
        #     "allergy": "",
        #     "note": ""
        # },
        "adminst": False
    }

    with open(os.path.join(user_path, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return jsonify(success=True)

# ---------------- PROFILE API ----------------
@app.route("/api/profile")
def api_profile():
    if "user" not in session:
        return jsonify({}), 401

    username = session["user"]["username"]
    info_path = os.path.join(USER_DIR, username, "info.json")

    if not os.path.exists(info_path):
        return jsonify({})

    with open(info_path, encoding="utf-8") as f:
        return jsonify(json.load(f))

@app.route("/api/profile/update", methods=["POST"])
def update_profile():
    if "user" not in session:
        return jsonify(success=False, msg="not logged in"), 401

    username = session["user"].get("username")
    if not username:
        return jsonify(success=False, msg="no username"), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify(success=False, msg="invalid data"), 400

    info_path = os.path.join(USER_DIR, username, "info.json")

    info = {}
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)

    info.update(data)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return jsonify(success=True)

# ---------------- TEST ----------------
from datetime import datetime
import shutil

@app.route("/test", methods=["GET", "POST"])
def test():
    if "user" not in session:
        flash("Please login first", "error")
        return redirect(url_for("home"))

    if request.method == "POST":
        username = session["user"]["username"]

        base_user_dir = os.path.join(USER_DIR, username)
        uploads_root = os.path.join(base_user_dir, "uploads")
        qs_root = os.path.join(base_user_dir, "questionnaires")

        os.makedirs(uploads_root, exist_ok=True)

        # ---------- สร้าง run folder ----------
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(uploads_root, run_id)
        result_dir = os.path.join(run_dir, "result")

        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        saved = {}

        # ---------- save uploaded files ----------
        file_map = {
            "eyeImage": "eye.png",
            "skinImage": "skin.png",
            "enoseCSV": "enose.csv",
            "breathData": "breath.csv"
        }

        for key, filename in file_map.items():
            f = request.files.get(key)
            if f and f.filename:
                path = os.path.join(run_dir, filename)
                f.save(path)
                saved[key] = path

        # ---------- questionnaire ----------
        qs_filename = request.form.get("questionnaire_file")
        if not qs_filename:
            flash("Please select questionnaire", "error")
            return redirect(url_for("test"))

        src_qs = os.path.join(qs_root, qs_filename)
        dst_qs = os.path.join(run_dir, "questionnaire.csv")

        if not os.path.exists(src_qs):
            flash("Questionnaire not found", "error")
            return redirect(url_for("test"))

        shutil.copy(src_qs, dst_qs)

        # ---------- AI PIPELINE ----------
        result = Breathescan_L(
            input_img=saved.get("eyeImage"),
            path_img=run_dir,
            sensor=saved.get("enoseCSV"),
            ans=dst_qs,
            result_dir=result_dir
        )

        flash("AI Analysis completed", "success")
        return render_template(
            "test.html",
            result=result,
            run_id=run_id
        )

    return render_template("test.html")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=2020)
