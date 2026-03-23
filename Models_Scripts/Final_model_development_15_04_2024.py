# 1. IMPORT LIBRARIES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

np.random.seed(42)


# 2. LOAD DATA

data_folder = "/Users/krishnagupta/Desktop/updated_training_file/"

file_list = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv")]

all_data = []

for file_name in file_list:
    df_temp = pd.read_csv(file_name)

    complex_id = os.path.basename(file_name).split("_")[0]
    df_temp["complex_id"] = complex_id

    # IMPORTANT: EDIT THIS BASED ON YOUR DATA
    if "HIV" in file_name:
        df_temp["virus_family"] = "HIV"
    elif "Influenza" in file_name:
        df_temp["virus_family"] = "Influenza"
    elif "SARS" in file_name:
        df_temp["virus_family"] = "SARS-CoV-2"
    else:
        df_temp["virus_family"] = "Other"

    all_data.append(df_temp)

df_all = pd.concat(all_data, ignore_index=True)

print("Total samples:", df_all.shape)
print("Total complexes:", df_all["complex_id"].nunique())


# 3. FEATURES

feature_cols = [
    "cmi.m$value","cc.m$value","cp.m$value","cp1.m$value",
    "cp2.m$value","hcm.m$value","rsa.m$value","scm.m$value",
    "ssp.m$value","ecc.m$value","ecmi.m$value","ecp.m$value",
    "ecp1.m$value","ecp2.m$value","ehcm.m$value","ersa.m$value",
    "escm.m$value","essp.m$value"
]

target_col = "inf.m$value"

# 4. MODEL FUNCTION

def build_model():
    model = Sequential([
        Dense(32, activation='relu', input_dim=18),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model


# 5. TRAIN + EVALUATE FUNCTION

def train_evaluate(train_df, test_df, return_history=False):

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Scaling
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    model = build_model()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_res, y_train_res,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_prob)

    if return_history:
        return acc, auc, model, scaler, history
    else:
        return acc, auc, model, scaler

# ============================================
# 6. SINGLE SPLIT + SAVE MODEL
# ============================================
print("\n===== SINGLE SPLIT =====")

unique_complexes = df_all["complex_id"].unique()

train_c, test_c = train_test_split(unique_complexes, test_size=0.2, random_state=42)

train_df = df_all[df_all["complex_id"].isin(train_c)]
test_df  = df_all[df_all["complex_id"].isin(test_c)]

acc, auc, model, scaler, history = train_evaluate(train_df, test_df, return_history=True)

print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")


# 7. SAVE TRAINING CURVES

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')

plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Training Curves")
plt.legend()

os.makedirs("results", exist_ok=True)
plt.savefig("results/training_curves.png", dpi=300)
plt.close()

print("Training curves saved!")


# 8. SAVE MODEL + SCALER

os.makedirs("saved_model", exist_ok=True)

model.save("saved_model/HVIface_model.keras")
model.save_weights("saved_model/HVIface_weights.h5")
joblib.dump(scaler, "saved_model/scaler.pkl")

print("Model and scaler saved!")

# ============================================
# 9. 10-FOLD CROSS VALIDATION
# ============================================
print("\n===== 10-FOLD CV =====")

kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_acc = []
cv_auc = []

for i, (train_idx, test_idx) in enumerate(kf.split(unique_complexes)):
    print(f"Fold {i+1}")

    train_c = unique_complexes[train_idx]
    test_c  = unique_complexes[test_idx]

    train_df = df_all[df_all["complex_id"].isin(train_c)]
    test_df  = df_all[df_all["complex_id"].isin(test_c)]

    acc, auc, _, _ = train_evaluate(train_df, test_df)

    cv_acc.append(acc)
    cv_auc.append(auc)

    print(f"Acc: {acc:.4f}, AUC: {auc:.4f}")

print("\n===== CV SUMMARY =====")
print(f"Mean Accuracy: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
print(f"Mean AUC: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")

# Save CV results
cv_df = pd.DataFrame({
    "Fold": range(1, 11),
    "Accuracy": cv_acc,
    "AUC": cv_auc
})

cv_df.to_csv("results/cv_results.csv", index=False)


# 10. LOFO VALIDATION

print("\n===== LOFO VALIDATION =====")

families = df_all["virus_family"].unique()
lofo_results = []

for fam in families:
    print(f"\nTesting on: {fam}")

    train_df = df_all[df_all["virus_family"] != fam]
    test_df  = df_all[df_all["virus_family"] == fam]

    if len(test_df) < 50:
        print("Skipping (too small)")
        continue

    acc, auc, _, _ = train_evaluate(train_df, test_df)

    lofo_results.append((fam, acc, auc))

    print(f"{fam} → Accuracy: {acc:.4f}, AUC: {auc:.4f}")

# Save LOFO results
lofo_df = pd.DataFrame(lofo_results, columns=["Family", "Accuracy", "AUC"])
lofo_df.to_csv("results/lofo_results.csv", index=False)

print("\nAll results saved successfully!")

#SAVE THE MODEL AND LOAD IT FOR THE PREDICTION

model.save("Desktop/ANN_model_oversampling_15_04_2024_testing_7bestfeatures.keras")
model.save_weights("ANN_model_oversampling.weights.h5")

---------------------------------------------------------------------------------------------------------------------------------------

#For LOFO (Leave one family out) you can add Virus family at the beginning of CSV file.


# 1. IMPORT LIBRARIES

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

np.random.seed(42)


# 2. LOAD DATA

data_folder = "/Users/krishnagupta/Desktop/updated_training_file/"
mapping_file = "family_mapping.csv"  # <-- YOU CREATE THIS

file_list = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

# Load family mapping
mapping_df = pd.read_csv(mapping_file)
family_map = dict(zip(mapping_df["filename"], mapping_df["virus_family"]))


# 3. FEATURES

feature_cols = [
    "cmi.m$value","cc.m$value","cp.m$value","cp1.m$value",
    "cp2.m$value","hcm.m$value","rsa.m$value","scm.m$value",
    "ssp.m$value","ecc.m$value","ecmi.m$value","ecp.m$value",
    "ecp1.m$value","ecp2.m$value","ehcm.m$value","ersa.m$value",
    "escm.m$value","essp.m$value"
]

target_col = "inf.m$value"


# 4. MODEL

def build_model():
    model = Sequential([
        Dense(32, activation='relu', input_dim=18),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])
    return model


# 5. TRAIN FUNCTION

def train_model(train_df, test_df):

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Scaling
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    model = build_model()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_res, y_train_res,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)

    return acc, auc, y_test, y_prob, model, scaler


# 6. SINGLE SPLIT (FOR ROC and MODEL SAVE)

print("\n===== SINGLE SPLIT =====")

file_array = np.array(file_list)

train_files, test_files = train_test_split(file_array, test_size=0.2, random_state=42)

train_df = pd.concat([pd.read_csv(os.path.join(data_folder, f)) for f in train_files])
test_df  = pd.concat([pd.read_csv(os.path.join(data_folder, f)) for f in test_files])

acc, auc, y_test, y_prob, model, scaler = train_model(train_df, test_df)

print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")


# 7. ROC CURVE

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

os.makedirs("results", exist_ok=True)
plt.savefig("results/roc_curve.png", dpi=300)
plt.close()

print("ROC curve saved!")


# 8. SAVE MODEL + SCALER

os.makedirs("saved_model", exist_ok=True)

model.save("saved_model/HVIface_model.keras")
model.save_weights("saved_model/HVIface_weights.h5")
joblib.dump(scaler, "saved_model/scaler.pkl")

print("Model saved!")


# 9. 10-FOLD CV TABLE

print("\n===== 10-FOLD CV =====")

kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_results = []

for i, (train_idx, test_idx) in enumerate(kf.split(file_array)):

    train_files = file_array[train_idx]
    test_files  = file_array[test_idx]

    train_df = pd.concat([pd.read_csv(os.path.join(data_folder, f)) for f in train_files])
    test_df  = pd.concat([pd.read_csv(os.path.join(data_folder, f)) for f in test_files])

    acc, auc, _, _, _, _ = train_model(train_df, test_df)

    cv_results.append([i+1, acc, auc])

cv_df = pd.DataFrame(cv_results, columns=["Fold", "Accuracy", "ROC-AUC"])

# Mean ± SD
mean_row = pd.DataFrame([[
    "Mean ± SD",
    f"{cv_df['Accuracy'].mean():.3f} ± {cv_df['Accuracy'].std():.3f}",
    f"{cv_df['ROC-AUC'].mean():.3f} ± {cv_df['ROC-AUC'].std():.3f}"
]], columns=cv_df.columns)

cv_table = pd.concat([cv_df, mean_row], ignore_index=True)

cv_table.to_csv("results/Table1_CV.csv", index=False)

print("\nCV Table saved!")


# 10. LOFO TABLE

print("\n===== LOFO =====")

families = list(set(family_map.values()))
lofo_results = []

for fam in families:

    train_files = [f for f in file_list if family_map[f] != fam]
    test_files  = [f for f in file_list if family_map[f] == fam]

    if len(test_files) < 2:
        continue

    train_df = pd.concat([pd.read_csv(os.path.join(data_folder, f)) for f in train_files])
    test_df  = pd.concat([pd.read_csv(os.path.join(data_folder, f)) for f in test_files])

    acc, auc, _, _, _, _ = train_model(train_df, test_df)

    lofo_results.append([fam, len(test_df), acc, auc])

lofo_df = pd.DataFrame(lofo_results, columns=["Viral Family", "Test Samples", "Accuracy", "ROC-AUC"])

lofo_df.to_csv("results/Table2_LOFO.csv", index=False)

print("LOFO Table saved!")


# 11. PRINT SUMMARY

print("\n===== TABLE 1 (CV) =====")
print(cv_table)

print("\n===== TABLE 2 (LOFO) =====")
print(lofo_df)
