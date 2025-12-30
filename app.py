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

app = Flask(__name__)
app.secret_key = "20"

UPLOAD_FOLDER = "uploads"
USER_CSV = "user.csv"
USER_DIR = "user"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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

@app.route("/contact")
def contact():
    return render_template("contact.html")

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
@app.route("/test", methods=["GET", "POST"])
def test():
    if "user" not in session:
        flash("Please login first", "error")
        return redirect(url_for("home"))

    if request.method == "POST":
        file_keys = [
            "eyeImage",
            "skinImage",
            "enoseCSV",
            "questionnaireCSV",
            "breathData"
        ]

        username = session["user"]["username"]

        base_user_dir = os.path.join(USER_DIR, username)
        upload_dir = os.path.join(base_user_dir, "uploads")
        result_dir = os.path.join(base_user_dir, "result")

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        saved_files = {}

        for key in file_keys:
            f = request.files.get(key)
            if f and f.filename:
                save_path = os.path.join(upload_dir, f.filename)
                f.save(save_path)
                saved_files[key] = save_path

        if not saved_files:
            flash("Please upload files", "error")
            return redirect(url_for("test"))

        # ----------------------------
        # เตรียม input ให้ Breathescan_L
        # ----------------------------
        input_img = saved_files.get("eyeImage")
        path_img = upload_dir

        sensor = saved_files.get("enoseCSV")
        ans = saved_files.get("questionnaireCSV")

        # เรียก AI Pipeline
        result = Breathescan_L(
            input_img=input_img,
            path_img=path_img,
            sensor=sensor,
            ans=ans,
            result_dir=result_dir
        )

        flash("AI Analysis completed successfully", "success")

        return render_template(
            "test.html",
            result=result
        )

    return render_template("test.html")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=2020)
