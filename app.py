from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
import os, csv, json
from HarmoMed.HarmoMed import HarmoMed_lir

app = Flask(__name__)
app.secret_key = "breathe_secret_key"

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
        writer.writerow(["username","password","first_name","last_name","email","phone"])

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("home.html", user=session.get("user"))

@app.route("/about")
def about():
    return render_template("about.html", user=session.get("user"))

@app.route("/research")
def research():
    return render_template("research.html", user=session.get("user"))

@app.route("/HarmoMed")
def HarmoMed():
    return render_template("HarmoMed.html", user=session.get("user"))

@app.route("/knowledge")
def knowledge():
    return render_template("knowledge.html", user=session.get("user"))

@app.route("/contact")
def contact():
    return render_template("contact.html", user=session.get("user"))

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
            if row["username"] == username and row["password"] == password:
                session["user"] = username
                return jsonify(success=True)

    return jsonify(success=False, msg="Invalid username or password")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

@app.route("/user_image/<username>")
def user_image(username):
    user_path = os.path.join(USER_DIR, username)
    return send_from_directory(user_path, "profile.jpg")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["POST"])
def register():
    data = request.form if request.form else request.get_json()

    required = ["username","password","first_name","last_name","email","phone"]
    if not all(k in data for k in required):
        return jsonify(success=False, msg="Incomplete data")

    username = data["username"]

    # 🔍 check duplicate username
    with open(USER_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"] == username:
                return jsonify(success=False, msg="Username already exists")

    user_path = os.path.join(USER_DIR, username)
    os.makedirs(user_path, exist_ok=True)

    # save profile image
    photo = request.files.get("photo")
    if photo and photo.filename:
        photo.save(os.path.join(user_path, "profile.jpg"))

    # save user csv
    with open(USER_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            username,
            data["password"],
            data["first_name"],
            data["last_name"],
            data["email"],
            data["phone"]
        ])

    # save user info json
    with open(os.path.join(user_path, "info.json"), "w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, ensure_ascii=False)

    return jsonify(success=True)

# ---------------- TEST ----------------
@app.route("/test", methods=["GET", "POST"])
def test():
    if "user" not in session:
        flash("Please login first", "error")
        return redirect(url_for("home"))

    if request.method == "POST":
        file_keys = ["eyeImage", "skinImage", "enoseCSV", "questionnaireCSV", "breathData"]
        user_upload = os.path.join(USER_DIR, session["user"], "uploads")
        os.makedirs(user_upload, exist_ok=True)

        saved = False
        for key in file_keys:
            f = request.files.get(key)
            if f and f.filename:
                f.save(os.path.join(user_upload, f.filename))
                saved = True

        if not saved:
            flash("Please upload files", "error")
            return redirect(url_for("test"))

        # 🔬 AI Analysis (placeholder)
        result = "AI Analysis Result: Low liver disease risk"
        flash(result, "success")

    return render_template("test.html", user=session.get("user"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
