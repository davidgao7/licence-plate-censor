import torch
from torchvision import transforms
from flask import Flask, request, jsonify

# from flask_ngrok import run_with_ngrok
import numpy as np
import cv2

# Load the PyTorch model
model = torch.load("best.pt")

app = Flask(__name__)
# run_with_ngrok(app)  # Start ngrok when the app is run


@app.route("/predict", methods=["POST"])
def predict():
    # Check if the POST request has the file part
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read the image file
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess the image for the model
    # Assuming model expects a 3x224x224 input
    # Convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Process the output (mock processing here)
    # You might need to apply softmax or argmax depending on your model
    prediction = torch.argmax(output, dim=1).item()

    return jsonify({"output": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
