from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch
from torchvision import transforms

from model import FastCNN_PINN


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "banana_pinn.pth"

app = Flask(__name__)


def load_model() -> FastCNN_PINN:
    model = FastCNN_PINN()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()

transform = transforms.Compose(
    [
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ]
)


def get_stage(ripeness_score: float) -> str:
    if ripeness_score < 0.25:
        return "Unripe"
    if ripeness_score < 0.6:
        return "Semi-Ripe"
    if ripeness_score < 0.85:
        return "Ripe"
    return "Overripe"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file in form-data field 'image'."}), 400

    day_value = request.form.get("day")
    if day_value is None:
        return jsonify({"error": "Missing 'day' in form-data."}), 400

    try:
        day = int(day_value)
    except ValueError:
        return jsonify({"error": "'day' must be an integer."}), 400

    if day < 1 or day > 6:
        return jsonify({"error": "'day' must be between 1 and 6."}), 400

    image_file = request.files["image"]

    try:
        image = Image.open(image_file).convert("RGB")
    except Exception:
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    image_tensor = transform(image).unsqueeze(0)
    time_tensor = torch.tensor([[day / 6]], dtype=torch.float32)

    with torch.no_grad():
        ripeness = model(image_tensor, time_tensor).item()

    return jsonify(
        {
            "ripeness": round(ripeness, 3),
            "stage": get_stage(ripeness),
            "day": day,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
