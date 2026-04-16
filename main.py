import os
import uuid

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER
from src.predict import (
    are_class_names_ready,
    get_class_names_error,
    get_model_error,
    is_model_ready,
    load_class_names,
    predict_image,
)
from src.calorie_estimator import get_calories
from src.data_loader import load_data
from config import TRAIN_DIR, TEST_DIR

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

startup_issues = []

if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not is_model_ready():
    startup_issues.append(
        "Trained model not found at 'models/food_model.h5'. "
        "Run training or place the model file there."
    )

if not are_class_names_ready():
    try:
        _, _, class_names = load_data(TRAIN_DIR, TEST_DIR)
        load_class_names(class_names)
    except Exception:
        startup_issues.append(
            "Class labels are unavailable. Add your dataset at "
            "'dataset/train' and 'dataset/test' before training, or place "
            "'models/class_names.json' beside the trained model."
        )

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if startup_issues:
            error = " ".join(startup_issues)
        elif not file or not file.filename:
            error = "Please choose an image before predicting."
        else:
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(filepath)
            image_path = url_for("static", filename=f"uploads/{unique_name}")

            try:
                food_name, confidence = predict_image(filepath)
                calories = get_calories(food_name)

                result = f"{food_name} ({confidence*100:.2f}%) - Calories: {calories}"
            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html",
        result=result,
        image=image_path,
        error=error,
        startup_issues=startup_issues,
        model_error=get_model_error(),
        class_names_error=get_class_names_error(),
    )

if __name__ == "__main__":
    app.run(debug=True)
