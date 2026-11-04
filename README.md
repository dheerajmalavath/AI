# 🍱 Food AI & ML Notebooks

A collection of machine learning notebooks focused on **food recognition**, **segmentation**, and **calorie estimation** — built with XGBoost, EfficientNetB3, Mask R-CNN, and DeepLabV3+.

---

## 📂 Notebooks

### 1. 🔥 Calories Burnt Prediction — XGBoost
**File:** `calories-burnt-detection_XGBoost.ipynb`

Predicts calories burned during exercise using physiological features (age, gender, height, weight, duration, heart rate, body temperature).

- **Model:** XGBoost Regressor
- **Techniques:** Feature engineering, cross-validation, hyperparameter tuning
- **Dataset:** [Calories Burnt Prediction — Kaggle](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction)

---

### 2. 🍛 Indian Food Image Classification — EfficientNetB3
**File:** `efficientnetb3_classification.ipynb`

Classifies **80+ Indian food categories** from images using transfer learning on EfficientNetB3.

- **Model:** EfficientNetB3 (ImageNet pretrained, fine-tuned)
- **Techniques:** Transfer learning, class-weight balancing, data augmentation
- **Dataset:** [Indian Food Images Dataset — Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset)
- **Trained Model (Kaggle):** [indian-food-model-v2-keras](https://www.kaggle.com/datasets/dheerajmalavath/indian-food-model-v2-keras)

---

### 3. 🎭 Food Instance Segmentation + Calorie Tracking — Mask R-CNN
**File:** `maskrcnn.ipynb`

Performs **instance segmentation** on food items using Mask R-CNN with an EfficientNetB3 classifier backbone to identify and estimate calories per food item on a plate.

- **Model:** Mask R-CNN (PyTorch) + EfficientNetB3 classifier
- **Techniques:** Instance segmentation, before/after plate comparison, per-item calorie delta
- **Dataset:** [FoodSeg103 — Kaggle](https://www.kaggle.com/datasets/ggrill/foodseg103)
- **Trained Model (Kaggle):** [maskrcnn-food-epoch](https://www.kaggle.com/datasets/dheerajmalavath/maskrcnn-food-epoch)

> ⚠️ **Model weights are not included** in this repo due to file size. Download from the Kaggle dataset links above.

---

### 4. 🖼️ Food Semantic Segmentation — DeepLabV3+
**File:** `segmentation_deeplabv3.ipynb`

Performs **semantic segmentation** of food images using DeepLabV3+ with MobileNetV2 backbone, trained on the FoodSeg103 dataset.

- **Model:** DeepLabV3+ with MobileNetV2 backbone
- **Techniques:** Semantic segmentation, pixel-wise food class labeling
- **Dataset:** [FoodSeg103 — Kaggle](https://www.kaggle.com/datasets/ggrill/foodseg103)

> ⚠️ **Model weights (`best_deeplabv3plus_mobilenetv2.pth`) are not included.** Train the model or attach your own weights as a Kaggle dataset before running this notebook.

---

## 📊 Datasets Summary

| Dataset | Used In | Link |
|---|---|---|
| Calories Burnt Prediction | `calories-burnt-detection_XGBoost.ipynb` | [🔗 Kaggle](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction) |
| Indian Food Images Dataset | `efficientnetb3_classification.ipynb` | [🔗 Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset) |
| FoodSeg103 | `maskrcnn.ipynb`, `segmentation_deeplabv3.ipynb` | [🔗 Kaggle](https://www.kaggle.com/datasets/ggrill/foodseg103) |

---

## 🚀 How to Run

All notebooks are designed to run on **Kaggle**. To run them:

1. Go to [kaggle.com](https://www.kaggle.com) and create a new notebook
2. Click **"Add Data"** and attach the dataset(s) listed for that notebook (see table above)
3. Copy-paste or upload the `.ipynb` file
4. Run all cells

> **GPU recommended** for the deep learning notebooks (`efficientnetb3_classification`, `maskrcnn`, `segmentation_deeplabv3`).

---

## 🛠️ Requirements

The notebooks use the standard Kaggle Python environment. Key libraries:

```
tensorflow / keras
torch / torchvision
scikit-learn
xgboost
pandas / numpy
matplotlib / seaborn
Pillow
```

---

## 👤 Author

**Dheeraj Malavath**
- Kaggle: [@dheerajmalavath](https://www.kaggle.com/dheerajmalavath)
- GitHub: [@dheerajmalavath](https://github.com/dheerajmalavath)
