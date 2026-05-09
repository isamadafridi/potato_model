# 🥔 Potato Disease Detector

A deep learning web application that detects potato leaf diseases using **MobileNetV2** transfer learning. Upload a potato leaf image and get instant disease classification.

---

## 📋 Classes Detected

| Class | Description |
|---|---|
| 🟠 Early Blight | Caused by *Alternaria solani* fungus |
| 🔴 Late Blight | Caused by *Phytophthora infestans* |
| 🟢 Healthy | No disease detected |

---

## 🏗️ Project Structure

```
potato-disease-project/
│
├── api.py                        ← FastAPI backend
├── frontend.html                     ← Frontend UI
├── requirements.txt               ← Python dependencies
├── potato_model_mobileNetV2.keras ← Trained model
└── README.md
```

---

## ⚙️ Tech Stack

- **Model** — MobileNetV2 (Transfer Learning, pretrained on ImageNet)
- **Training** — TensorFlow / Keras (Google Colab)
- **Backend** — FastAPI + Uvicorn
- **Frontend** — HTML / CSS / JavaScript
- **Dataset** — PlantVillage Dataset (~2,152 images)

---

## 🚀 Setup & Execution

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/potato-disease-detector.git
cd potato-disease-detector
```

---

### Step 2 — Create Conda Environment

```bash
conda create -n potato-disease python=3.11
conda activate potato-disease
```

> ⚠️ Always make sure `(potato-disease)` appears in your terminal before proceeding.

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Add the Trained Model

Download the model file and place it in the project root:

```
potato-disease-project/
├── main.py
├── potato_model_mobileNetV2.keras   ← place here
```

> Model download link: [Google Drive / Release link here]

---

### Step 5 — Run the API Server

Open **Terminal 1**:

```bash
conda activate potato-disease
uvicorn main:app --reload
```

Verify server is running by visiting:
```
http://localhost:8000
```

Expected response:
```json
{"message": "Potato Disease API is running"}
```

---

### Step 6 — Run the Frontend

Open **Terminal 2**:

```bash
python -m http.server 3000
```

Then open your browser and go to:
```
http://localhost:3000/index.html
```

---

### Step 7 — Use the App

1. Click **Choose File** and upload a potato leaf image
2. Click **Detect Disease**
3. View the predicted class and confidence score

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Training Accuracy | 99.4% |
| Validation Accuracy | 100% |
| Test Accuracy | ~99% |
| Architecture | MobileNetV2 + Dense(128) |

### Training Details

| Setting | Value |
|---|---|
| Input Size | 256 × 256 × 3 |
| Batch Size | 32 |
| Optimizer | Adam (lr=0.001) |
| Loss | SparseCategoricalCrossentropy |
| Epochs | 50 (EarlyStopping) |

---

## 🔄 Data Pipeline

```
Raw Images [0-255]
      ↓
   cache (raw images saved to memory)
      ↓
   Augmentation (train only)
   - RandomFlip horizontal
   - RandomRotation 0.15
   - RandomZoom 0.1
   - RandomBrightness 0.1
      ↓
   preprocess_input (scale to [-1, 1])
      ↓
   shuffle + prefetch
      ↓
   MobileNetV2
```

---

## 🧠 Model Architecture

```
Input (256, 256, 3)
      ↓
MobileNetV2 (pretrained, frozen)
      ↓
GlobalAveragePooling2D
      ↓
Dense(128, relu)
      ↓
Dropout(0.3)
      ↓
Dense(3, softmax)  ← Early Blight / Late Blight / Healthy
```

---

## ⚠️ Limitations

- Model trained on **PlantVillage dataset** (controlled lab conditions)
- May not perform well on real-world photos with:
  - Natural backgrounds (soil, grass)
  - Poor lighting or shadows
  - Multiple leaves in frame
- Non-leaf images may still receive a classification (always predicts one of 3 classes)

### For Best Results

```
✅ Single leaf in frame
✅ Plain background (white paper / wall)
✅ Good lighting, no shadows
✅ Leaf fills 80% of the frame
✅ Clear, focused photo
```

---

## 📦 requirements.txt

```
tensorflow==2.20.0
fastapi
uvicorn
python-multipart
pillow
numpy
scikit-learn
matplotlib
```

---

## 🔁 Retraining the Model

1. Open `training.ipynb` in Google Colab
2. Upload dataset to Google Drive
3. Mount Drive and set `DATASET_PATH`
4. Run all cells
5. Download the saved `.keras` model
6. Replace `potato_model_mobileNetV2.keras` in project folder

---

## 📁 Dataset

- **Source** — [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- **Classes Used** — Potato Early Blight, Potato Late Blight, Potato Healthy
- **Total Images** — ~2,152

---

## 👤 Author

**Abdul Samad Afridi**
- GitHub: [@isamadafridi](https://github.com/isamadafridi)

---

## 📄 License

This project is licensed under the MIT License.
