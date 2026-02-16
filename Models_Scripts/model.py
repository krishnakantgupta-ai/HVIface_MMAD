# 1. IMPORT NECESSARY LIBRARIES
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 2. LOAD ALL COMPLEX FILES (BUT DONT CONCATENATE)


data_folder = "YOUR_FOLDER_PATH HAVING CSV FILES"   # <-- change this
file_list = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv")]

all_data = []

for file_name in file_list:
    df_temp = pd.read_csv(file_name)

    # Extract complex ID from filename
    complex_id = os.path.basename(file_name).split("_")[0]
    df_temp["complex_id"] = complex_id

    all_data.append(df_temp)

df_all = pd.concat(all_data, ignore_index=True)

print("Total samples:", df_all.shape)
print("Total complexes:", df_all["complex_id"].nunique())

# 3. SPLIT BY COMPLEX (NO LEAKAGE)

unique_complexes = df_all["complex_id"].unique()

train_complexes, test_complexes = train_test_split(
    unique_complexes,
    test_size=0.2,
    random_state=42
)

train_df = df_all[df_all["complex_id"].isin(train_complexes)]
test_df  = df_all[df_all["complex_id"].isin(test_complexes)]

print("Train complexes:", len(train_complexes))
print("Test complexes:", len(test_complexes))

# 4. DEFINE FEATURES AND LABEL

feature_cols = [
    "cmi.m$value","cc.m$value","cp.m$value","cp1.m$value",
    "cp2.m$value","hcm.m$value","rsa.m$value","scm.m$value",
    "ssp.m$value","ecc.m$value","ecmi.m$value","ecp.m$value",
    "ecp1.m$value","ecp2.m$value","ehcm.m$value","ersa.m$value",
    "escm.m$value","essp.m$value"
]

target_col = "inf.m$value"

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

# 5. APPLY SMOTE (TRAIN ONLY: OVERSAMPLING)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("After SMOTE:", X_train_res.shape)

# 6. SCALE FEATURES (FIT ON TRAIN ONLY)

scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# 7. BUILD ANN MODEL

model = Sequential()

model.add(Dense(32, input_dim=18, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

model.summary()

# 8. EARLY STOPPING

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 9. TRAIN MODEL

history = model.fit(
    X_train_res,
    y_train_res,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 10. EVALUATE MODEL

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Accuracy:", accuracy)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred_prob))
