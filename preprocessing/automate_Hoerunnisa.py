import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from joblib import dump

def preprocess_data(data_path, target_col, save_dir="liver_cancer_preprocessing"):
    df = pd.read_csv(data_path)

    # Hapus duplikasi
    df = df.drop_duplicates()
    os.makedirs(save_dir, exist_ok=True)

    # Simpan header kolom untuk inverse transform
    header_path = os.path.join(save_dir, "data_columns.csv")
    df.drop(columns=[target_col]).head(0).to_csv(header_path, index=False)

    # Pisahkan fitur numerik & kategorikal
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    if target_col in numeric_features:
        numeric_features.remove(target_col)
    if target_col in categorical_features:
        categorical_features.remove(target_col)

    # Tangani outlier numerik (IQR)
    for col in numeric_features:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        median_val = df[col].median()
        df[col] = np.where((df[col] < lower) | (df[col] > upper), median_val, df[col])

    # Pisahkan fitur dan target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Bagi kolom kategorikal
    label_cols = ["gender"]
    ohe_cols = ["alcohol_consumption", "smoking_status", "physical_activity_level"]

    # Pipeline numerik & kategorikal
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    label_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    ohe_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('label', label_transformer, label_cols),
        ('ohe', ohe_transformer, ohe_cols)
    ])

    # Fit-transform data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # SMOTE balancing
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_transformed, y_train)

    # Simpan pipeline
    pipeline_path = os.path.join(save_dir, "preprocessor_pipeline.joblib")
    dump(preprocessor, pipeline_path)

    # Dapatkan nama kolom setelah transformasi
    label_names = label_cols
    ohe_names = preprocessor.named_transformers_['ohe']['encoder'].get_feature_names_out(ohe_cols)
    all_columns = numeric_features + label_names + list(ohe_names) + [target_col]
    
    # Simpan hasil preprocessing
    train_res = pd.DataFrame(np.hstack([X_train_res, y_train_res.values.reshape(-1, 1)]), columns=all_columns)
    test_res = pd.DataFrame(np.hstack([X_test_transformed, y_test.values.reshape(-1, 1)]), columns=all_columns)

    # Konversi kolom biner ke integer
    binary_cols = list(ohe_names) + label_names + [target_col] + ['hepatitis_b', 'hepatitis_c', 'cirrhosis_history', 'family_history_cancer', 'diabetes']
    train_res[binary_cols] = train_res[binary_cols].astype(int)
    test_res[binary_cols] = test_res[binary_cols].astype(int)

    train_res.to_csv(os.path.join(save_dir, "train_liver.csv"), index=False)
    test_res.to_csv(os.path.join(save_dir, "test_liver.csv"), index=False)

    print(f"Hasil preprocessing disimpan di folder: {save_dir}")

    return train_res, test_res, pipeline_path

# Jalankan jika dieksekusi langsung
if __name__ == "__main__":
    data_path = "../liver_cancer_dataset.csv"
    target_col = "liver_cancer"
    train_res, test_res, pipeline_path = preprocess_data(data_path, target_col)

    print("=== FILE HASIL ===")
    print(os.listdir("liver_cancer_preprocessing"))
