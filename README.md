# HVIface_MMAD

HVIface is a deep learning framework designed to predict interaction interfaces between two protein sequences in human–virus systems.

---

## HVIface Tutorial

Follow the steps below to perform interface prediction:

### 1. Feature Generation
Compile 18 features for the two input protein sequences.  
Refer to the **CoRNeA tutorial** for detailed instructions:
- `CoRNeA_tutorial.pdf`

---

### 2. Load the Trained Model
Load the pre-trained model:
- `HVIface_model.keras`

Use the provided notebook:
- `ANN_Oversampling_under_testing_final_18_06_2024-Copy1.ipynb`

---

### 3. Perform Prediction
Run the notebook to predict pairwise residue interactions between the two protein sequences.

---

### 4. Post-processing
Follow the **CoRNeA tutorial** for post-processing steps:
- Filtering predictions  
- Generating interaction networks  
- Refining interface residues  

---

### 5. Output
- The predicted pairwise interactions are generated in **CSV format**
- Results can be sorted based on **convolution scores** to identify high-confidence interactions

---

## Additional Resources

- Validation data:  
  https://github.com/krishnakantgupta-ai/HVIface_MMAD/tree/main/Validation

---

## Notes

- The model uses **SMOTE-based imbalance handling**
- Training incorporates **early stopping for optimal performance**
- Designed specifically for **human–virus protein interaction interfaces**

---
