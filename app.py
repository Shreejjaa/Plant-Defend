
from flask import Flask, render_template, request
import os
from utils import predict_leaf_condition

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        result = predict_leaf_condition(filepath)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
