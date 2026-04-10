#  Jet Classification using Multi-Modal Machine Learning

##  Overview
This project focuses on classifying jet particles using both **tabular physics features** and **jet images**. It combines advanced preprocessing techniques with powerful machine learning models to achieve high classification performance on large-scale HDF5 datasets.

---

##  Dataset
- **Format:** HDF5 (`.h5`)
- **Training Files:** 60 (1 corrupt file removed)
- **Validation Files:** 27  

Each file contains:
- `jets_data` → Tabular features (53 features)
- `jetImage` → Grayscale images (100×100)
- `target` → Jet labels (`j_g`, `j_q`, `j_w`, `j_z`, `j_t`)

---

##  Data Pipeline

###  Data Loading
- Loaded multiple `.h5` files using `h5py`
- Skipped corrupt/empty files
- Combined into:
  - Tabular dataset (`train_df`, `validation_df`)
  - Image generator (for CNN)

---

###  Data Preprocessing

####  Missing Value Handling
- Certain columns had **0 values treated as missing**
- Converted `0 → NaN`
- Applied **KNN Imputation (k=12)**

####  Feature Scaling
- **MinMaxScaler** (for imputation)
- **StandardScaler** (before PCA)

####  Dimensionality Reduction
- Applied **PCA**
- Reduced features from **53 → 13**

---

###  Label Encoding
- Converted categorical labels into numeric format using `LabelEncoder`

---

##  Models Used

###  CatBoost Classifier
- Iterations: 4375  
- Early stopping enabled  
- Handles complex tabular relationships efficiently  

###  LightGBM Classifier
- Trees: 1000  
- Early stopping used  
- Fast and scalable gradient boosting model  

---

##  Results

| Model       | Accuracy |
|------------|----------|
| CatBoost   | **80.9%** |
| LightGBM   | 80.3%     |

---

##  Image Data Pipeline (CNN Ready)
- Implemented generator for large-scale image loading  
- Image shape: `(100, 100, 1)`  
- Memory-efficient batch processing for deep learning  

---

##  Key Highlights
-  Efficient handling of large HDF5 datasets  
-  End-to-end ML pipeline  
-  Domain-aware preprocessing (zero → missing)  
-  Advanced models (CatBoost, LightGBM)  
-  ~81% classification accuracy  
-  Scalable CNN-ready data pipeline  

---

##  Future Improvements
-  Train CNN on jet images  
-  Build **hybrid model (CNN + tabular features)**  
-  Add **SHAP explainability**  
-  Hyperparameter tuning  
-  Remove PCA for tree-based models  

---

##  Tech Stack
- Python  
- NumPy, Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- CatBoost  
- LightGBM  
- h5py  

---

##  Conclusion
This project demonstrates a **real-world machine learning workflow** involving large datasets, advanced preprocessing, and high-performance models. It highlights the importance of combining **domain knowledge with modern ML techniques**.

---
