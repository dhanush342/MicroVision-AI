from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def misclassification_analysis(model, loader, classes):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            outputs = model(images)
            _, p = torch.max(outputs, 1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    print(classification_report(labels, preds, target_names=classes))

    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()