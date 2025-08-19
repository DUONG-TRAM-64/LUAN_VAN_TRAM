
#https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv/data
# CSE raw

# ==================== IMPORTS ====================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import glob, os

# ==================== TIỀN XỬ LÝ DỮ LIỆU ====================
def preprocess_ics_data(df):
    # 1. Xoá các cột không cần thiết
    columns_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # 2. Các cột kiểu int
    int_columns = [
        'Dst Port', 'Protocol', 'Flow Duration',
        'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
        'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
        'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
        'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Tot', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Tot', 'Bwd IAT Max', 'Bwd IAT Min',
        'Fwd PSH Flags', 'Bwd PSH Flags',
        'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Len', 'Bwd Header Len',
        'Pkt Len Min', 'Pkt Len Max',
        'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
        'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
        'CWE Flag Count', 'ECE Flag Cnt',
        'Down/Up Ratio',
        'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
        'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
        'Subflow Fwd Pkts', 'Subflow Fwd Byts',
        'Subflow Bwd Pkts', 'Subflow Bwd Byts',
        'Init Fwd Win Byts', 'Init Bwd Win Byts',
        'Fwd Act Data Pkts', 'Fwd Seg Size Min',
        'Active Max', 'Active Min',
        'Idle Max', 'Idle Min'
    ]

    # 3. Các cột kiểu float
    float_columns = [
        'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
        'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
        'Flow Byts/s', 'Flow Pkts/s',
        'Flow IAT Mean', 'Flow IAT Std',
        'Fwd IAT Mean', 'Fwd IAT Std',
        'Bwd IAT Mean', 'Bwd IAT Std',
        'Fwd Pkts/s', 'Bwd Pkts/s',
        'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
        'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
        'Active Mean', 'Active Std',
        'Idle Mean', 'Idle Std'
    ]

    # 4. Ép kiểu dữ liệu
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # 5. Gộp nhãn
    label_mapping = {
        'Benign': 'BENIGN',
        'DDOS attack-HOIC': 'DDoS',
        'DDoS attacks-LOIC-HTTP': 'DDoS',
        'DDOS attack-LOIC-UDP': 'DDoS',
        'DoS attacks-Hulk': 'DoS',
        'DoS attacks-SlowHTTPTest': 'DoS',
        'DoS attacks-GoldenEye': 'DoS',
        'DoS attacks-Slowloris': 'DoS',
        'Bot': 'Bot',
        'FTP-BruteForce': 'Brute Force',
        'SSH-Bruteforce': 'Brute Force',
        'Brute Force -Web': 'Web Attack',
        'Brute Force -XSS': 'Web Attack'
    }
    df['Label'] = df['Label'].replace(label_mapping)

    # 6. Mã hoá nhãn
    label_order = ['BENIGN', 'DDoS', 'DoS', 'Bot', 'Brute Force', 'Web Attack']
    label_names = {label: idx for idx, label in enumerate(label_order)}
    df['Label_encode'] = df['Label'].map(label_names)

    # 7. Chuyển đổi tất cả cột số
    exclude_cols = ['Dst Port', 'Protocol', 'Label', 'Label_encode']
    numeric_cols = df.columns.difference(exclude_cols)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 8. Xoá các cột không thay đổi (constant)
    unique_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if unique_cols:
        df.drop(columns=unique_cols, inplace=True)
        print(f"Đã xoá {len(unique_cols)} cột chỉ có 1 giá trị duy nhất: {unique_cols}")

    # 9. Chuẩn hóa cột số
    scaler = MinMaxScaler()
    scaled_cols = df.columns.difference(['Label', 'Label_encode'])
    df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

    # 10. Xoá cột trùng lặp
    df_T = df.T.drop_duplicates()
    duplicate_cols = [col for col in df.columns if col not in df_T.T.columns]
    if duplicate_cols:
        df.drop(columns=duplicate_cols, inplace=True)
        print(f"Đã xoá {len(duplicate_cols)} cột trùng nhau: {duplicate_cols}")

    print("Tiền xử lý dữ liệu basic xong")
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
    input_path = '/kaggle/input/dataset_cse.csv'
    output_path = '/kaggle/output/cleaned_cse.parquet'

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

