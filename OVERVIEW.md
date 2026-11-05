### 1. Semantic Segmentation — DeepLabV3+
* **Objective:** Performs dense, pixel-level semantic segmentation to identify and isolate distinct food items from complex visual backgrounds.
* **Architecture:** DeepLabV3+ with an Atrous Spatial Pyramid Pooling (ASPP) module atop an efficient MobileNetV2 feature-extraction backbone to preserve multi-scale context without computational overhead.
* **Methodology:** Employs dilated (atrous) convolutions with multi-receptive fields to generate high-resolution feature maps, followed by bilinear upsampling and skip-connection decoding to recover precise boundary delineations.
* **Dataset & Scale:** Trained on the **FoodSeg103 Dataset**, containing 7,118 fine-grained meal images categorized across 103 target ingredient and food classes.

### 2. Fine-Grained Food Image Classification — EfficientNetB3
* **Objective:** Categorizes complex multi-ingredient dishes into specific culinary classes under high intra-class variance and visual similarity.
* **Architecture:** EfficientNetB3 backbone leveraging compound scaling across network depth, width, and input image resolution (300×300 px), initialized with ImageNet pre-trained weights.
* **Methodology:** Customized with global average pooling, dropout regularization (0.4), and a multi-class softmax dense layer. Employs dynamic class-weight penalization to address data imbalance across rarer culinary classes during transfer-learning fine-tuning.
* **Dataset & Scale:** Evaluated on the **Indian Food Images Dataset**, comprising 4,000+ curated images spanning 80+ regional food categories.

### 3. Calorie-Burn Tabular Regression — XGBoost
* **Objective:** Predicts post-activity energy expenditure (net calories burned) based on physiological profiles and session exercise parameters.
* **Architecture:** Extreme Gradient Boosting (XGBoost) Regressor implementing an ensemble of gradient-boosted decision trees (GBDT) optimized via root mean squared error (RMSE) minimization.
* **Methodology:** Models non-linear interactions among tabular features—including duration, heart rate, body temperature, age, gender, height, and weight. Applies regularization parameters (L1/L2) and tree-depth constraints to prevent overfitting on continuous metrics.
* **Dataset & Scale:** Trained and validated on the **Calories Burnt Prediction Dataset**, consisting of 15,000 continuous biometric and exercise transaction records.
