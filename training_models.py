# ---------------------------------------------------------------------------- #
#                                  ADD LIBRARY                                 #
# ---------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import time
from time import sleep
import os
import shutil
import gc
import subprocess
import psutil
import GPUtil
import threading
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Dropout, Flatten, Dense, BatchNormalization, GRU, Input
from keras.optimizers import RMSprop, Adam

# ---------------------------------------------------------------------------- #
#                                DATA PROCESSING                               #
# ---------------------------------------------------------------------------- #
def load_data():
    print("\n- Đang đọc dữ liệu...")
    
    # #CSE
    # train_df = pd.read_parquet('/kaggle/input/cse/train_balanced.parquet')
    # test_df = pd.read_parquet('/kaggle/input/cse/test.parquet')

    # #IoT
    # train_df = pd.read_parquet('/kaggle/input/iot23/train_balanced.parquet')
    # test_df = pd.read_parquet('/kaggle/input/iot23/test.parquet')

    #ICS
    train_df = pd.read_parquet('/kaggle/input/ics-flow/train_balanced.parquet')
    test_df = pd.read_parquet('/kaggle/input/ics-flow/test.parquet')
    # Kiem tra tap Train-Test
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True) 
    
    return train_df, test_df


def preprocess_data(train_df, test_df, model_num):
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42)
   
    print("Validation shape:", val_df.shape)
    print("\n- Phân bố nhãn trong tập train:")
    print(train_df['Label'].value_counts())
    print("\n- Phân bố nhãn trong tập validation:")
    print(val_df['Label'].value_counts())
    print("\n- Phân bố nhãn trong tập test:")
    print(test_df['Label'].value_counts())
    
    X_train = train_df.drop(['Label', 'Label_encode'], axis=1).values
    y_train = train_df['Label_encode'].values
    X_val = val_df.drop(['Label', 'Label_encode'], axis=1).values
    y_val = val_df['Label_encode'].values
    X_test = test_df.drop(['Label', 'Label_encode'], axis=1).values
    y_test = test_df['Label_encode'].values
    
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_test)
    # Reshape cho phu hợp với kiến trúc yêu cầu đa chiều , (CNN) , (Nhom RNN)
    if model_num in [1, 2, 3, 5, 6, 7]:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    classes = np.array(sorted(train_df['Label_encode'].unique()))
    class_weights_arr = class_weight.compute_class_weight('balanced', classes=classes, y=train_df['Label_encode'].values)
    class_weights = dict(zip(classes, class_weights_arr))
    label_names = list(train_df.sort_values('Label_encode')['Label'].unique())
    
    return X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, class_weights, label_names, y_test

# ---------------------------------------------------------------------------- #
#                                 COnfig model                                 #
# ---------------------------------------------------------------------------- #
def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    print("\n- Bắt đầu quá trình huấn luyện...")
    # Neu dung early stopping : Chỉnh  cho phù hợp ví dụ : patience=15
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=99999, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=99999, min_lr=1e-7, verbose=1)
    ]
    start = time.time()
    # Chinh cac SIEU THAM SO truoc khi huan luyen 
    history = model.fit(X_train, y_train,
                        batch_size=256, epochs=100,
                        validation_data=(X_val, y_val), callbacks=callbacks,
                        class_weight=class_weights, verbose=1)
    train_time = time.time() - start
    print(f"- Thời gian huấn luyện: {train_time:.2f} giây")
    return history, train_time

# Biến toàn cục cho cấu hình số nơ-ron (256)
neuron_per_layer = {
    1: [256],
    2: [128, 128],
    3: [64, 64, 128],
    4: [32, 32, 64, 128],
    5: [32, 32, 64, 64, 64]
}
def model_CNN(n, input_shape, num_classes):
    print(f"\n- Xây dựng mô hình CNN - {n} Conv1D layers")
    model = Sequential()
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong neuron_per_layer. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    # Thêm Conv1D(basic) + Dropout giữa các layer
    for i, filters in enumerate(layer_list):
        model.add(Conv1D(
            filters, kernel_size=2, activation='relu', padding='same',
            input_shape=input_shape if i == 0 else None,
            name=f'conv1d_{i+1}'
        ))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))

    # BatchNormalization trước khi Flatten
    model.add(BatchNormalization(name='batch_norm'))
    # Giam so DIM(3) cua CNN : To + Dense(output)
    model.add(Flatten(name='flatten'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    # Compile model
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
# ---------------------------------------------------------------------------- #
#                                  MODEL LSTM                                  #
# ---------------------------------------------------------------------------- #
def model_LSTM(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình LSTM - {n} LSTM layers")
    model = Sequential()
    #return_sequences --> TRUE(DIM=3) or FALSE(DIM=2)--> giữ trạng thái cuối cùng của HIDDEN layer cuối cùng (bỏ Timestep Giữ :(Batchzie, Hidden_last))
    #return_sequences=False --> là DIM(2) nên không cần Flatten như CNN vì nhóm RNN có cơ chế như vậy
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    for i, units in enumerate(layer_list):
        # Thêm LSTM(basic) + Dropout giữa các layer
        if i == 0:
            model.add(LSTM(units, input_shape=input_shape, return_sequences=n > 1, name=f'lstm_{i+1}'))
        else:
            model.add(LSTM(units, return_sequences=i < len(layer_list) - 1, name=f'lstm_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))
    
    model.add(BatchNormalization(name='batch_normalization'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                   MODEL GRU                                  #
# ---------------------------------------------------------------------------- #
def model_GRU(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình GRU - {n} GRU layers")
    model = Sequential()
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    for i, units in enumerate(layer_list):
        # Thêm GRU(basic) + Dropout giữa các layer
        if i == 0:
            model.add(GRU(units, input_shape=input_shape, return_sequences=n > 1, name=f'gru_{i+1}'))
        else:
            model.add(GRU(units, return_sequences=i < len(layer_list) - 1, name=f'gru_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))

    
    model.add(BatchNormalization(name='batch_normalization'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                   MODEL MLP                                  #
# ---------------------------------------------------------------------------- #
def model_MLP(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình MLP - {n} Dense layers")
    model = Sequential()
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")

    layer_list = neuron_per_layer[n]
    for i, units in enumerate(layer_list):
        # Thêm Dense + Dropout giữa các layer
        if i == 0:
            model.add(Dense(units, activation='relu', input_shape=input_shape, name=f'dense_MLP_{i+1}'))
        else:
            model.add(Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))
        
    model.add(BatchNormalization(name='batch_normalization'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                MODEL HYBRID                                  #
# ---------------------------------------------------------------------------- #
def model_CNN_LSTM(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình CNN-LSTM - {n} LSTM layers")
    model = Sequential()
    
    #  CNN layers trich xuat dac trung 
    model.add(Conv1D(32, kernel_size=2, activation='relu', input_shape=input_shape, name='conv1d_1'))
    model.add(Conv1D(64, kernel_size=2, activation='relu', name='conv1d_2'))

    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    # LSTM layers #return_sequences --> TRUE or FALSE
    for i, units in enumerate(layer_list):
        model.add(LSTM(units, return_sequences=i < len(layer_list) - 1, name=f'lstm_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))
    
    model.add(BatchNormalization(name='batch_normalization'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    # Compile model
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                   MODEL CNN-GRU                              #
# ---------------------------------------------------------------------------- #
def model_CNN_GRU(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình CNN-GRU - {n} GRU layers")
    model = Sequential()
    
    # Fixed CNN layers trich xuat DT
    model.add(Conv1D(32, kernel_size=2, activation='relu', input_shape=input_shape, name='conv1d_1'))
    model.add(Conv1D(64, kernel_size=2, activation='relu', name='conv1d_2'))

    
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    # GRU layers
    for i, units in enumerate(layer_list):
        model.add(GRU(units, return_sequences=i < len(layer_list) - 1, name=f'gru_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))
    
    model.add(BatchNormalization(name='batch_normalization'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    # Compile model
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def model_CNN_MLP(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình CNN-MLP - {n} Dense layers")
    model = Sequential()
    # CNN layers 
    model.add(Conv1D(32, kernel_size=2, activation='relu', input_shape=input_shape, name='conv1d_1'))
    model.add(Conv1D(64, kernel_size=2, activation='relu', name='conv1d_2'))
    # GIAM so chieu cua CNN dim(3) (gom thanh 1 vecto + batchsize)=dim(2) --> phu hop voi dense
    model.add(Flatten(name='flatten'))
    
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong neuron_per_layer. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    # Dense layers 
    for i, units in enumerate(layer_list):
        model.add(Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_dense_{i+1}'))
    
    model.add(BatchNormalization(name='batch_normalization'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model
# # ---------------------------------------------------------------------------- #
#                                   MODEL NAMES                                 #
# ---------------------------------------------------------------------------- #
def get_model_name(num):
    model_names = {
        1: "CNN", 2: "LSTM", 3: "GRU", 4: "MLP", 5: "CNN-LSTM",  7: "CNN-MLP" 
    }
    return model_names.get(num, "Không tìm thấy mô hình tương ứng")

def build_model(n, model_num, input_shape, num_classes):
    if model_num == 1:
        return model_CNN(n, input_shape, num_classes)
    elif model_num == 2:
        return model_LSTM(n, input_shape, num_classes)
    elif model_num == 3:
        return model_GRU(n, input_shape, num_classes)
    elif model_num == 4:
        return model_MLP(n, input_shape, num_classes)
    elif model_num == 5:
        return model_CNN_LSTM(n, input_shape, num_classes)
    elif model_num == 6:
        return model_CNN_MLP(n, input_shape, num_classes)
    else:
        raise ValueError("model_num không hợp lệ. Vui lòng chọn từ 1 đến 6.")
    
# ---------------------------------------------------------------------------- #
#                                Evalute on TEST                               #
# ---------------------------------------------------------------------------- #
def evaluate_model(model, X_test, y_test_cat, y_true, label_names, output_dir, n, model_num):
    print("\n- Đánh giá mô hình...")   
    # Evaluate model
    model_name = get_model_name(model_num)

    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    
    start = time.time()
    y_pred = model.predict(X_test, verbose=0)
    infer_time = time.time() - start
    y_pred_classes = np.argmax(y_pred, axis=1)
   
    print(f"- Accuracy: {test_accuracy * 100:.2f}%")
    print(f"- Suy luận trung bình: {infer_time / X_test.shape[0] * 1000:.4f} ms")
   
    # Generate classification report
    print("\n- Classification Report:")
    report = classification_report(y_true, y_pred_classes, digits=4, output_dict=True)
    print(classification_report(y_true, y_pred_classes, digits=4))
    
    # Convert report to DataFrame and save to CSV
    report_df = pd.DataFrame(report).transpose().round(4)
    report_file_path = f"{output_dir}/report_01_class_{model_name}_{n}_lay.csv"
                       
    report_df.to_csv(report_file_path, index=True)
    print(f"- Classification report saved to {report_file_path}")
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    return test_accuracy, infer_time, y_pred_classes, report, cm

# ---------------------------------------------------------------------------- #
#                               Training_history                               #
# ---------------------------------------------------------------------------- #
def save_training_history(history, output_dir, n, model_num):
    model_name = get_model_name(model_num)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title(f"Accuracy ({model_name} - {n} Layers)")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title(f"Loss ({model_name} - {n} Layers)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_history_{model_name}_{n}_layers.png")
    plt.close()
    print(f"Đã lưu ảnh training history cho {model_name} với {n} layers.")
# ---------------------------------------------------------------------------- #
#                              confusion_matrices                              #
# ---------------------------------------------------------------------------- #
def save_confusion_matrices(cm, label_names, output_dir, n, model_num):
    model_name = get_model_name(model_num)
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(10, 8))
    ax1 = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - Counts ({model_name} - {n} Layers)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    count_path = os.path.join(output_dir, f'{model_name}_{n}_layers_matrix.png')
    plt.savefig(count_path)
    plt.close()
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    ax2 = sns.heatmap(np.round(cm_percent, 2), annot=True, fmt='.2f', cmap='YlGnBu',
                      xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - Percentage ({model_name} - {n} Layers)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    percent_path = os.path.join(output_dir, f'{model_name}_{n}_layers_matrix_percent.png')
    plt.savefig(percent_path)
    plt.close()
    print(f"Saved confusion matrices for {model_name} with {n} layers.")

# ---------------------------------------------------------------------------- #
#                            RESOURCE MONITORING                               #
# ---------------------------------------------------------------------------- #
def get_gpu_memory_used_nvidia_smi():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        memory_str = result.stdout.strip()
        if memory_str:
            return float(memory_str.split(' ')[0])
        return 0.0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Cảnh báo: Không thể lấy thông tin bộ nhớ GPU qua nvidia-smi: {e}")
        return 0.0
    except Exception as e:
        print(f"Cảnh báo: Lỗi không xác định khi lấy bộ nhớ GPU: {e}")
        return 0.0

def monitor_cpu(cpu_log_file, stop_event):
    """Thread để ghi log CPU usage"""
    with open(cpu_log_file, 'w') as f:
        f.write("timestamp,cpu_percent,memory_percent\n")
        while not stop_event.is_set():
            try:
                timestamp = time.strftime("%Y/%m/%d %H:%M:%S")
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                f.write(f"{timestamp},{cpu_percent},{memory_percent}\n")
                f.flush()
                time.sleep(1)  # Giống với nvidia-smi loop=1
            except Exception as e:
                print(f"Lỗi khi ghi CPU log: {e}")
                break

def start_monitor(output_dir, model_name, n_layers):
    gpu_log_file = os.path.join(output_dir, f"gpu_usage_log_{model_name}_{n_layers}_layers.csv")
    cpu_log_file = os.path.join(output_dir, f"cpu_usage_log_{model_name}_{n_layers}_layers.csv")
    
    print(f"- File log GPU: {gpu_log_file}")
    print(f"- File log CPU: {cpu_log_file}")
    
    # Xóa file cũ nếu tồn tại
    for file in [gpu_log_file, cpu_log_file]:
        if os.path.exists(file):
            os.remove(file)
            print(f"- Đã xóa file log cũ: {file}")
    
    nvidia_smi_command = [
        "nvidia-smi",
        "--loop=1",
        f"--query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,utilization.memory",
        "--format=csv,noheader"
    ]
    
    gpu_process = None
    gpu_log_fd = None
    cpu_thread = None
    stop_event = threading.Event()
    
    try:
        # Bắt đầu ghi log GPU
        gpu_log_fd = open(gpu_log_file, 'a')
        gpu_process = subprocess.Popen(
            nvidia_smi_command,
            stdout=gpu_log_fd,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        
        # Bắt đầu thread ghi log CPU
        cpu_thread = threading.Thread(target=monitor_cpu, args=(cpu_log_file, stop_event))
        cpu_thread.start()
        
        print("- Bắt đầu ghi log GPU và CPU...")
    except Exception as e:
        print(f"Lỗi khi bắt đầu ghi log: {e}")
    
    p = psutil.Process(os.getpid())
    ram_before = p.memory_info().rss / (1024 * 1024) 
    gpus = GPUtil.getGPUs()
    gpu_mem_before = gpus[0].memoryUsed if gpus else get_gpu_memory_used_nvidia_smi()
    start_time = time.time()
    
    return gpu_process, ram_before, gpu_mem_before, start_time, gpu_log_fd, cpu_thread, stop_event

def end_monitor(gpu_process, ram_before, gpu_mem_before, start_time, gpu_log_fd, cpu_thread=None, stop_event=None):
    total_time = time.time() - start_time
    print(f"- Dừng ghi log và tính toán tài nguyên sử dụng ({total_time:.2f}s)...")
    
    # Dừng CPU thread
    if stop_event:
        stop_event.set()
    if cpu_thread:
        cpu_thread.join(timeout=2)
    
    # Dừng GPU process
    if gpu_process:
        gpu_process.terminate()
        try:
            gpu_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            print("Cảnh báo: Tiến trình nvidia-smi không dừng trong 2 giây, đang kill.")
            gpu_process.kill()
        if gpu_log_fd:
            gpu_log_fd.close()
    
    # Tính toán tài nguyên
    p = psutil.Process(os.getpid())
    ram_after = p.memory_info().rss / (1024 * 1024) 
    ram_increase = ram_after - ram_before
    
    gpus = GPUtil.getGPUs()
    gpu_mem_after = gpus[0].memoryUsed if gpus else get_gpu_memory_used_nvidia_smi()
    gpu_ram_increase = gpu_mem_after - gpu_mem_before
    
    print(f"  - RAM hệ thống tăng: {ram_increase:.2f} MB")
    if gpus:
        print(f"  - GPU RAM đã dùng thêm: {gpu_ram_increase:.2f} MB")
    
    return total_time, ram_increase, gpu_ram_increase


def plot_gpu_cpu_usage(model_name, num_layers, output_dir):
    """Vẽ biểu đồ GPU và CPU usage từ 2 file log riêng biệt"""
    gpu_log_file = os.path.join(output_dir, f"gpu_usage_log_{model_name}_{num_layers}_layers.csv")
    cpu_log_file = os.path.join(output_dir, f"cpu_usage_log_{model_name}_{num_layers}_layers.csv")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(gpu_log_file) or not os.path.exists(cpu_log_file):
        print(f"Cảnh báo: Thiếu file log GPU hoặc CPU. Bỏ qua việc vẽ biểu đồ.")
        return
    
    try:
        # Đọc dữ liệu GPU
        gpu_df = pd.read_csv(gpu_log_file, names=['Timestamp', 'Power_Draw', 'Temperature', 'GPU_Utilization', 'Memory_Utilization'])
        
        # Xử lý dữ liệu GPU
        gpu_df['Power_Draw'] = gpu_df['Power_Draw'].str.replace(' W', '').astype(float)
        gpu_df['Temperature'] = gpu_df['Temperature'].astype(float)
        gpu_df['GPU_Utilization'] = gpu_df['GPU_Utilization'].str.replace(' %', '').astype(float)
        gpu_df['Memory_Utilization'] = gpu_df['Memory_Utilization'].str.replace(' %', '').astype(float)
        gpu_df['Step'] = range(len(gpu_df))
        
        # Đọc dữ liệu CPU
        cpu_df = pd.read_csv(cpu_log_file)
        cpu_df['Step'] = range(len(cpu_df))
        
    except Exception as e:
        print(f"Lỗi khi đọc file log: {e}")
        return
    
    # Vẽ 6 biểu đồ
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # GPU metrics (4 biểu đồ)
    gpu_metrics = [
        (gpu_df, 'Temperature', 'Nhiệt độ GPU (°C)', 'red', 'GPU Temperature'),
        (gpu_df, 'Power_Draw', 'Công suất tiêu thụ (W)', 'purple', 'GPU Power Draw'),
        (gpu_df, 'GPU_Utilization', '% Sử dụng GPU', 'green', 'GPU Utilization'),
        (gpu_df, 'Memory_Utilization', '% Sử dụng bộ nhớ GPU', 'blue', 'GPU Memory Utilization')     
    ]    
    # CPU metrics (2 biểu đồ)
    cpu_metrics = [
        (cpu_df, 'cpu_percent', '% Sử dụng CPU', 'orange', 'CPU Utilization'),
        (cpu_df, 'memory_percent', '% Sử dụng RAM', 'brown', 'System Memory Utilization')
    ] 
    # Vẽ GPU metrics
    for i, (df, column, ylabel, color, title) in enumerate(gpu_metrics):
        ax = axes[i]
        ax.plot(df['Step'], df[column], color=color, linewidth=1.5)
        ax.set_title(f'{title} ({model_name} - {num_layers} layers)')
        # ax.set_xlabel('Lần ghi (Step)')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        if "Utilization" in column:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Vẽ CPU metrics
    for i, (df, column, ylabel, color, title) in enumerate(cpu_metrics, start=4):
        ax = axes[i]
        ax.plot(df['Step'], df[column], color=color, linewidth=1.5)
        ax.set_title(f'{title} ({model_name} - {num_layers} layers)')
        # ax.set_xlabel('Lần ghi (Step)')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'system_usage_plot_{model_name}_{num_layers}_layers.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu biểu đồ hệ thống vào {output_file}")

def save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir, model_num, total_time, ram_increase, gpu_ram_increase):
    model_name = get_model_name(model_num)
    gpu_log_file = os.path.join(output_dir, f"gpu_usage_log_{model_name}_{n}_layers.csv")
    cpu_log_file = os.path.join(output_dir, f"cpu_usage_log_{model_name}_{n}_layers.csv")
    
    # Khởi tạo giá trị mặc định
    avg_gpu_util = 0.0
    max_gpu_mem_util = 0.0
    avg_gpu_temp = 0.0
    avg_power_draw = 0.0
    avg_cpu_util = 0.0
    avg_memory_cpu = 0.0
    
    # Xử lý file log GPU
    if os.path.exists(gpu_log_file):
        try:
            gpu_data = pd.read_csv(gpu_log_file, names=['timestamp', 'power_draw', 'temperature_gpu', 'utilization_gpu', 'utilization_memory'])
            
            # Làm sạch dữ liệu GPU
            gpu_data['power_draw'] = gpu_data['power_draw'].str.replace(' W', '').astype(float)
            gpu_data['temperature_gpu'] = gpu_data['temperature_gpu'].astype(float)
            gpu_data['utilization_gpu'] = gpu_data['utilization_gpu'].str.replace(' %', '').astype(float)
            gpu_data['utilization_memory'] = gpu_data['utilization_memory'].str.replace(' %', '').astype(float)
            
            # Tính toán GPU metrics
            avg_gpu_util = gpu_data['utilization_gpu'].mean()
            max_gpu_mem_util = gpu_data['utilization_memory'].max()
            avg_gpu_temp = gpu_data['temperature_gpu'].mean()
            avg_power_draw = gpu_data['power_draw'].mean()
            
        except Exception as e:
            print(f"Cảnh báo: Không thể xử lý file log GPU: {e}")
    
    # Xử lý file log CPU
    if os.path.exists(cpu_log_file):
        try:
            cpu_data = pd.read_csv(cpu_log_file)
            
            # Tính toán CPU metrics
            avg_cpu_util = cpu_data['cpu_percent'].mean()
            avg_memory_cpu = cpu_data['memory_percent'].mean()
            
        except Exception as e:
            print(f"Cảnh báo: Không thể xử lý file log CPU: {e}")
    
    # Lưu báo cáo văn bản
    with open(f"{output_dir}/report_02_{model_name}_{n}_layers.txt", 'w') as f:
        f.write(f"TÓM TẮT KẾT QUẢ - {model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Tên mô hình: {model_name}\n")
        f.write(f"Số layer: {n}\n")
        f.write(f"Số mẫu train: {X_train.shape[0]:,}\n")
        f.write(f"Số mẫu test: {X_test.shape[0]:,}\n")
        f.write(f"Số epoch đã train: {len(history.history['loss'])}\n")
        f.write(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}\n")
        f.write(f"Test accuracy cuối cùng: {test_accuracy:.4f}\n")
        f.write(f"Số lượng tham số: {model.count_params():,}\n")
        f.write(f"Thời gian huấn luyện: {train_time:.2f} giây\n")
        f.write(f"Tổng thời gian suy luận: {infer_time:.2f} giây\n")
        f.write(f"Suy luận trung bình mỗi mẫu: {infer_time / X_test.shape[0] * 1000:.4f} ms\n")
        f.write("\n- Tổng kết tài nguyên:\n")
        f.write(f"- Tổng thời gian chạy: {total_time:.2f} giây\n")
        f.write(f"- RAM tăng: {ram_increase:.2f} MB\n")
        f.write(f"- GPU RAM đã dùng thêm: {gpu_ram_increase:.2f} MB\n")
        f.write(f"- Mức sử dụng GPU trung bình: {avg_gpu_util:.2f} %\n")
        f.write(f"- Mức sử dụng bộ nhớ GPU tối đa: {max_gpu_mem_util:.2f} %\n")
        f.write(f"- Nhiệt độ GPU trung bình: {avg_gpu_temp:.2f} °C\n")
        f.write(f"- Công suất tiêu thụ trung bình: {avg_power_draw:.2f} W\n")
        f.write(f"- Mức sử dụng CPU trung bình: {avg_cpu_util:.2f} %\n")
        f.write(f"- Mức sử dụng RAM trung bình: {avg_memory_cpu:.2f} %\n")
    
    # Lưu báo cáo CSV
    report_data = {
        "Train_Time_s": [round(train_time, 2)],
        "Inference_Time_s": [round(infer_time, 2)],
        "Best_Val_Accuracy": [round(max(history.history['val_accuracy']), 4)],
        "Test_Accuracy": [round(test_accuracy, 4)],
        "Avg_GPU_Util_%": [round(avg_gpu_util, 2)],
        "Max_GPU_Mem_Util_%": [round(max_gpu_mem_util, 2)],
        "Avg_GPU_Temp_C": [round(avg_gpu_temp, 2)],
        "Avg_Power_Draw_W": [round(avg_power_draw, 2)],
        "Avg_CPU_Util_%": [round(avg_cpu_util, 2)],
        "Avg_Memory_CPU_%": [round(avg_memory_cpu, 2)]
    }
    
    report_df = pd.DataFrame(report_data)
    csv_file_path = f"{output_dir}/report_02_power_{model_name}_{n}_lay.csv"
    report_df.to_csv(csv_file_path, index=False)
    print(f"Đã lưu báo cáo CSV cho {model_name} với {n} layers.")


def main():
    print("- Bắt đầu quá trình huấn luyện các mô hình...")
    output_dir = "/kaggle/working/output"
    if os.path.exists(output_dir):
        print(f"- Đang xóa thư mục output cũ: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"- Thư mục output mới đã được tạo: {output_dir}")
    # train_df, test_df = load_data()  # Tải dữ liệu một lần
   # 1: "CNN", 2: "LSTM", 3: "GRU", 4: "MLP", 5: "CNN-LSTM", 6: "CNN-GRU", 7: "CNN-MLP",    bỏ 8: "MLP", 9: "DBN"
    model_nums = [1 ] 
    # model_nums = [1,4,5,7,2,3]
    # model_nums = [1,4,5,7]
    # layers_list = [ 1,2,3,4,5]
    layers_list = [5]
    
    for model_num in model_nums:
       
        for n in layers_list:
            train_df, test_df = load_data()  # Tải dữ liệu Nhiều lần --> Mới sửa
            model_name = get_model_name(model_num)
            print(f"\n=== Bắt đầu huấn luyện mô hình {model_name} với {n} layer ===")
            # Bắt đầu theo dõi tài nguyên
            # process, ram_before, gpu_mem_before, start_time, gpu_log_fd = start_monitor(output_dir, model_name, n)
            process, ram_before, gpu_mem_before, start_time, gpu_log_fd, cpu_thread, stop_event = start_monitor(output_dir, model_name, n)
            # Xử lý dữ liệu
            # X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, class_weights, label_names, y_test = preprocess_data(train_df, test_df)
            X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, class_weights, label_names, y_test = preprocess_data(train_df, test_df, model_num)
            input_shape = X_train.shape[1:]
            num_classes = y_train_cat.shape[1]
            # Xây dựng và huấn luyện mô hình
            model = build_model(n, model_num, input_shape, num_classes)
            history, train_time = train_model(model, X_train, y_train_cat, X_val, y_val_cat, class_weights)
            # Đánh giá mô hình
            test_accuracy, infer_time, y_pred_classes, report, cm = evaluate_model(model, X_test, y_test_cat, y_test, label_names, output_dir, n, model_num)
            # Kết thúc theo dõi tài nguyên
            # total_time, ram_increase, gpu_ram_increase = end_monitor(process, ram_before, gpu_mem_before, start_time, gpu_log_fd)
            total_time, ram_increase, gpu_ram_increase = end_monitor(process, ram_before, gpu_mem_before, start_time, gpu_log_fd, cpu_thread, stop_event)
            # Lưu kết quả
            save_training_history(history, output_dir, n, model_num)
            save_confusion_matrices(cm, label_names, output_dir, n, model_num)
            save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir, model_num, total_time, ram_increase, gpu_ram_increase)

            # Vẽ và lưu biểu đồ sử dụng GPU
            plot_gpu_cpu_usage(
                model_name=model_name,
                num_layers=n,
                output_dir=output_dir
            )         
            
            print(f"\n=== Kết thúc mô hình {model_name} với {n} layer ===")
            del model, history, y_pred_classes, report, cm, X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, y_test , train_df, test_df
            del process, ram_before, gpu_mem_before, start_time, gpu_log_fd ,test_accuracy, infer_time,train_time,total_time , ram_increase, gpu_ram_increase
            gc.collect()
            print(f"- Đã xóa các biến tạm thời sau khi train mô hình {model_name} {n} layer(s)")
            sleep(5) 
    
    gc.collect()
    print("- Đã xóa dữ liệu sau khi hoàn thành tất cả model.")

if __name__ == '__main__':
    main()