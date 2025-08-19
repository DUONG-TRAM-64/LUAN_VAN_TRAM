#https://www.kaggle.com/datasets/alirezadehlaghi/icssim
#ICS dataset raw

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import glob, os

# ==================== TIỀN XỬ LÝ DỮ LIỆU ====================
def preprocess_ics_data(df):

    # 1. Xoá các cột không cần thiết
    columns_to_drop = [
        'sAddress', 'rAddress', 'sMACs', 'rMACs', 'sIPs', 'rIPs', 'protocol',
        'startDate', 'endDate', 'start', 'end', 'startOffset', 'endOffset',
        'IT_B_Label', 'NST_B_Label', 'Label2'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # 2. Các cột kiểu int
    int_columns = [
        'sPackets', 'rPackets',
        'sBytesSum', 'rBytesSum',
        'sBytesMax', 'rBytesMax', 'sBytesMin', 'rBytesMin',
        'sBytesAvg', 'rBytesAvg',
        'sPayloadSum', 'rPayloadSum',
        'sPayloadMax', 'rPayloadMax', 'sPayloadMin', 'rPayloadMin',
        'sPayloadAvg', 'rPayloadAvg',
        'sttl', 'rttl'
    ]

    # 3. Các cột kiểu float
    float_columns = [
        'duration', 'sLoad', 'rLoad',
        'sInterPacketAvg', 'rInterPacketAvg',
        'sAckRate', 'rAckRate', 'sFinRate', 'rFinRate',
        'sPshRate', 'rPshRate', 'sSynRate', 'rSynRate',
        'sRstRate', 'rRstRate',
        'sWinTCP', 'rWinTCP',
        'sAckDelayMax', 'rAckDelayMax',
        'sAckDelayMin', 'rAckDelayMin',
        'sAckDelayAvg', 'rAckDelayAvg'
    ]

    # 4. Xử lý cột nhãn
    label_col = 'IT_M_Label'
    label2_col = 'NST_M_Label'

    if label_col in df.columns:
        df = df.rename(columns={label_col: 'Label'})
        label_col = 'Label'
    if label2_col in df.columns:
        df = df.rename(columns={label2_col: 'Label2'})
        label2_col = 'Label2'

    # 5. Ép kiểu & xử lý missing values
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val).round().astype('Int64')

    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)

    # 6. Mã hóa nhãn
    label_names = []
    if label_col in df.columns:
        le = LabelEncoder()
        df['Label_encode'] = le.fit_transform(df[label_col])
        label_names = list(le.classes_)
        print(f"Các nhãn đã mã hóa: {label_names}")

    # 7. Xử lý riêng cho Label2 (loại bỏ inconsistency)
    if 'Label2' in df.columns and 'Label_encode' in df.columns:
        df = df[~((df['Label_encode'] != 0) & (df['Label2'] == 'Normal'))]
        print("Đã lọc bỏ các hàng không hợp lệ theo điều kiện Label2.")

    # 8. Xoá cột chỉ có 1 giá trị duy nhất
    unique_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if unique_cols:
        df.drop(columns=unique_cols, inplace=True)
        print(f"Đã xoá {len(unique_cols)} cột chỉ có 1 giá trị duy nhất: {unique_cols}")

    # 9. Xoá cột trùng lặp
    df_T = df.T.drop_duplicates()
    duplicate_cols = [col for col in df.columns if col not in df_T.T.columns]
    if duplicate_cols:
        df.drop(columns=duplicate_cols, inplace=True)
        print(f"Đã xoá {len(duplicate_cols)} cột trùng nhau: {duplicate_cols}")

    # 10. Chuẩn hóa dữ liệu số
    numeric_columns = df.select_dtypes(include=['float64', 'int64', 'Int64']).columns
    cols_to_scale = [col for col in numeric_columns if col not in [label_col, 'Label_encode']]
    if cols_to_scale:
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        print(f"Đã chuẩn hóa {len(cols_to_scale)} cột số.")

    print("Tiền xử lý dữ liệu cơ ban xonggg")
    return df, label_names


# ==================== LOAD & SAVE ====================
def load_data(path):
    """ Đọc CSV hoặc folder chứa nhiều CSV """
    if os.path.isdir(path):
        return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(path, "*.csv"))], ignore_index=True)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Chỉ hỗ trợ đọc CSV hoặc thư mục chứa CSV.")

def save_result(df, output_path):
    df.to_parquet(output_path, index=False)


# ==================== CÂN BẰNG DỮ LIỆU ====================
def balance_data(X, y, labels, test_size=0.2, random_state=42):
    # 1. Chia train/test
    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
        X, y, labels,
        test_size=test_size, random_state=random_state, stratify=y
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
    input_path = '/kaggle/input/dataset_ics.csv'
    output_path = '/kaggle/output/cleaned_ics.parquet'

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
