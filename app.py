from flask import Flask, render_template, request, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = "breathe_secret_key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


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


# -------------------------------
# TEST MODEL PAGE
# -------------------------------
@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        file_keys = ["eyeImage", "skinImage", "enoseCSV", "questionnaireCSV", "breathData"]
        saved_files = []

        for key in file_keys:
            f = request.files.get(key)
            if f and f.filename != "":
                path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
                f.save(path)
                saved_files.append(f.filename)

        if not saved_files:
            flash("⚠️ Please upload all required files before submitting.", "error")
            return redirect(url_for("test"))

        # TODO: เชื่อมโมเดลจริงที่นี่
        result = "✅ AI Analysis Result: Low liver disease risk"

        flash(result, "success")
        return render_template("test.html", result=result)

    return render_template("test.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
