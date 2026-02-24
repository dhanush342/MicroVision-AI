import torch
import torch.nn.functional as F

def ensemble_predict(model1, model2, images, weight1=0.6, weight2=0.4):
    model1.eval()
    model2.eval()

    with torch.no_grad():
        out1 = F.softmax(model1(images), dim=1)
        out2 = F.softmax(model2(images), dim=1)

    final_output = weight1 * out1 + weight2 * out2
    return torch.argmax(final_output, dim=1)