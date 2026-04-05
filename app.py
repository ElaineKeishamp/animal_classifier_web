from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

app = Flask(__name__)


DROPOUT_RATE = 0.3
NUM_CLASSES = 3

def build_mobilenet_classifier():
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in backbone.features.parameters():
        param.requires_grad = False

    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES),
    )
    return backbone


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('animal_classifier.pth', map_location=device, weights_only=False)

model = build_mobilenet_classifier().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

label_classes = checkpoint['label_encoder_classes']


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    scores = {label_classes[i]: round(probs[i].item() * 100, 2) for i in range(len(label_classes))}
    predicted = max(scores, key=scores.get)

    return jsonify({'predicted': predicted, 'scores': scores})

if __name__ == '__main__':
    app.run(debug=True)