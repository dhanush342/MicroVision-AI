from flask import Flask, request, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
from models import get_efficientnet, get_resnet

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 4  # adjust
classes = ["class1","class2","class3","class4"]

eff_model = get_efficientnet(num_classes)
res_model = get_resnet(num_classes)

eff_model.load_state_dict(torch.load("efficientnet_microplastic.pth", map_location=device))
res_model.load_state_dict(torch.load("resnet_microplastic.pth", map_location=device))

eff_model.to(device)
res_model.to(device)

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out1 = torch.softmax(eff_model(image), dim=1)
        out2 = torch.softmax(res_model(image), dim=1)

    final = 0.6 * out1 + 0.4 * out2
    pred = torch.argmax(final, dim=1).item()

    return jsonify({"prediction": classes[pred]})

if __name__ == "__main__":
    app.run(debug=False)