# Lung Cancer Detection with Deep Learning (FastAI + Grad-CAM)

A high-performance binary classifier to detect **benign** vs **malignant** lung tissue from histopathological images using transfer learning and FastAI, enhanced with Grad-CAM explainability.

---

## 📁 Dataset

- **Source:** [LC25000 Dataset (Kermany et al.)](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- **Total Images Used:** 15,000
- **Classes:**
  - `benign`: Normal lung tissue
  - `malignant`: Includes adenocarcinoma and squamous cell carcinoma

---

## 🧪 Objective

To build a reproducible and interpretable pipeline for early lung cancer detection using image-based classification:
- Binary output: `benign` or `malignant`
- Visual interpretation of model decisions with **Grad-CAM**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **FastAI** | Data loading, augmentation, transfer learning |
| **ResNet34** | Pretrained convolutional base |
| **Grad-CAM** | Visual explanation of predictions |
| **Google Colab Pro** | GPU training |
| **Matplotlib / Seaborn** | Evaluation & heatmaps |

---

##  Pipeline Overview

1. **Data Preprocessing** (`data_prep.py`)
   - Original 3-class lung data → Mapped to `benign` / `malignant`
   - Stratified split into `train`, `valid`, and `test`

2. **Model Training** (`model.py`)
   - `ResNet34` backbone
   - `to_fp16()` mixed precision for speed
   - ~99.7% validation accuracy
   - Exported as `.pkl` for reuse

3. **Evaluation** (`evaluate.py`)
   - Precision, recall, F1-score on unseen test set
   - Confusion matrix

4. **Explainability** (`visualize.py`)
   - Grad-CAM visualization for individual test samples
   - Model focuses on biologically relevant regions

---

## ✅ Results

| Metric     | Value |
|------------|-------|
| Accuracy   | **100%** |
| Precision  | 1.00  |
| Recall     | 1.00  |
| F1-score   | 1.00  |

✅ Model generalizes extremely well on unseen test set.

---

##  Sample Grad-CAM Output

![image](https://github.com/user-attachments/assets/62dccc22-0376-4ec2-ae6c-c4a667bec13b)


*Red regions indicate what the model focuses on for prediction*

---

##  Project Structure

lung-cancer-detection-fastai/
├── data/
│ ├── raw/
│ └── interim/
├── notebooks/
│ └── 01_model_training.ipynb
├── src/
│ ├── data_prep.py
│ ├── model.py
│ ├── evaluate.py
│ └── visualize.py
├── exported/
│ └── lung_model.pkl
├── README.md
├── requirements.txt

## Future Work

- Train on more diverse datasets (e.g., LC500, CAMELYON)
- Deploy with Streamlit or Gradio
- Convert to a real-time biopsy screening app

---

##  Author

**Prakhar Srivastava**  
Graduate Student, MSIS @ NYU  
Aspiring ML & Product professional | [LinkedIn](https://www.linkedin.com/in/your-link)

---

## License

MIT License (Feel free to use this for learning or extensions.)
