# https://www.kaggle.com/datasets/akashdogra/cic-iot-2023
# CIC-IoT2023 raw

# ==================== IMPORTS ====================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==================== TIỀN XỬ LÝ CIC-IoT-2023 ====================
def preprocess_ciciot2023(df):
    # 1. Đổi tên cột nhãn về chuẩn
    if 'label' in df.columns:
        df.rename(columns={'label': 'Label'}, inplace=True)
    # không có các cộ t định danh 
    # 2. Ép kiểu dữ liệu số
    for col in df.columns:
        if col not in ['Label']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Xử lý giá trị NaN và Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 5. Loại bỏ cột chỉ có 1 giá trị duy nhất
    unique_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if unique_cols:
        df.drop(columns=unique_cols, inplace=True)
        print(f"Đã xoá {len(unique_cols)} cột constant: {unique_cols}")

    # 6. Chuẩn hóa tất cả cột số (trừ nhãn)
    numeric_cols = df.columns.difference(['Label'])
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 7. Loại bỏ cột trùng lặp
    df_T = df.T.drop_duplicates()
    duplicate_cols = [col for col in df.columns if col not in df_T.T.columns]
    if duplicate_cols:
        df.drop(columns=duplicate_cols, inplace=True)
        print(f"Đã xoá {len(duplicate_cols)} cột trùng lặp: {duplicate_cols}")

    # 8. Mã hoá nhãn
    label_order = df['Label'].unique()
    label_names = {label: idx for idx, label in enumerate(label_order)}
    df['Label_encode'] = df['Label'].map(label_names)

    print("Tiền xử lý dữ liệu CIC-IoT-2023 xong")
    return df, label_names

# ==================== LOAD & SAVE ====================
def load_data(path):
    if os.path.isdir(path):
        return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(path, "*.csv"))], ignore_index=True)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Chỉ hỗ trợ đọc CSV hoặc thư mục chứa CSV.")

def save_result(df, output_path):
    df.to_parquet(output_path, index=False)

# ==================== CÂN BẰNG DỮ LIỆU ====================
def balance_data(X, y, labels, max_per_class=150000, test_size=0.2, random_state=42):
    df = X.copy()
    df["Label_encode"] = y
    df["Label"] = labels

    # 1. Lấy mẫu lại (giữ nguyên nếu ít hơn max_per_class, còn nhiều hơn thì random sample)
    balanced_parts = []
    for lbl, group in df.groupby("Label"):
        if len(group) > max_per_class:
            balanced_parts.append(group.sample(n=max_per_class, random_state=random_state))
        else:
            balanced_parts.append(group)
    df_balanced = pd.concat(balanced_parts).reset_index(drop=True)

    print("Phân bố sau khi dùng kỹ thuật lấy mẫu RUS:")
    print(df_balanced["Label"].value_counts())

    # 2. Chia train/test (stratify theo nhãn)
    X = df_balanced.drop(columns=["Label", "Label_encode"])
    y = df_balanced["Label_encode"]
    labels = df_balanced["Label"]

    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
        X, y, labels, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Phân bố y_train:")
    print(y_train.value_counts())

    # 2. SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("Train sau SMOTE:", X_train_smote.shape)
    print(y_train_smote.value_counts())

    # 3. Tomek Links
    tl = TomekLinks()
    X_train_resampled, y_train_resampled = tl.fit_resample(X_train_smote, y_train_smote)
    print("Train sau TomekLinks:", X_train_resampled.shape)
    print(y_train_resampled.value_counts())

    # 4. Mapping lại Label gốc
    label_mapping = pd.DataFrame({'Label_encode': y_train, 'Label': labels_train}).drop_duplicates()
    label_mapping = label_mapping.set_index('Label_encode')['Label'].to_dict()
    train_balanced_labels = y_train_resampled.map(label_mapping)

    # 5. Tạo DataFrame kết quả
    train_df = X_train_resampled.copy()
    train_df['Label_encode'] = y_train_resampled
    train_df['Label'] = train_balanced_labels

    test_df = X_test.copy()
    test_df['Label_encode'] = y_test
    test_df['Label'] = labels_test

    return train_df, test_df
# ==================== RUN ====================
if __name__ == "__main__":
    input_path = '/kaggle/input/dataset_iot.csv'
    output_path = '/kaggle/output/cleaned_iot.parquet'

    # Load + tiền xử lý
    df = load_data(input_path)
    df_clean, label_names = preprocess_ics_data(df)
    save_result(df_clean, output_path)

    # Cân bằng dữ liệu
    X = df_clean.drop(columns=['Label', 'Label_encode'])
    y = df_clean['Label_encode']
    labels = df_clean['Label']
    train_df, test_df = balance_data(X, y, labels)

    # Lưu dữ liệu train/test
    train_path = '/kaggle/output/train_balanced.parquet'
    test_path = '/kaggle/output/test.parquet'
    save_result(train_df, train_path)
    save_result(test_df, test_path)

