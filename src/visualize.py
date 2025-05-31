
from fastai.vision.all import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

BASE = Path('/content/drive/MyDrive/lung-cancer-detection-fastai')
MODEL_PATH = BASE / 'exported/lung_model.pkl'
TEST_PATH = BASE / 'data/interim/test'

def show_gradcam(img_path):
    learn = load_learner(MODEL_PATH)
    img = PILImage.create(img_path)

    pred_class, pred_idx, probs = learn.predict(img)
    print(f"🔍 Prediction: {pred_class}")
    print(f"🧠 Confidence: {probs[pred_idx]:.4f}")

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    last_conv = list(learn.model.children())[0][-1]
    h1 = last_conv.register_forward_hook(forward_hook)
    h2 = last_conv.register_backward_hook(backward_hook)

    learn.model.eval()
    dl = learn.dls.test_dl([img])
    inp = dl.one_batch()[0]
    inp.requires_grad_()

    out = learn.model(inp)
    out[0, pred_idx].backward()

    grads = gradients[0][0]
    acts = activations[0][0]
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = (weights * acts).sum(0).detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()

    img_np = np.array(img.resize((224, 224)))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_np)
    ax.imshow(cam, cmap='jet', alpha=0.4)
    ax.set_title(f"Grad-CAM: {pred_class}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    h1.remove()
    h2.remove()
