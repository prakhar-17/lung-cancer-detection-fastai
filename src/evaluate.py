
from fastai.vision.all import *
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BASE = Path('/content/drive/MyDrive/lung-cancer-detection-fastai')
TEST_PATH = BASE/'data/interim/test'
MODEL_PATH = BASE/'exported/lung_model.pkl'

def evaluate_model():

    BASE = Path('/content/drive/MyDrive/lung-cancer-detection-fastai')
    TEST_PATH = BASE / 'data/interim/test'
    MODEL_PATH = BASE / 'exported/lung_model.pkl'

    # Load trained model
    learn = load_learner(MODEL_PATH)

    # Create test dataloader using the same vocab as training
    test_files = get_image_files(TEST_PATH)

    def get_label_from_folder(file): return file.parent.name

    test_dl = learn.dls.test_dl(test_files, with_labels=True, label_func=get_label_from_folder)

    # Predict
    preds, targets = learn.get_preds(dl=test_dl)
    pred_classes = preds.argmax(dim=1)

    # Use training vocab
    class_labels = learn.dls.vocab

    # Report
    print("📊 Classification Report:\n")
    print(classification_report(targets, pred_classes, target_names=class_labels))

    cm = confusion_matrix(targets, pred_classes)

    # Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
