import csv
import os

model_list = ["CNN", "LSTM", "GRU", "DNN", "DBN", "MLP", "CNN-LSTM", "CNN-GRU", "CNN-DNN"]

# Gộp từng nhóm model thành file get_pow{n}.csv
for n in range(1, 6): 
    output_file = f"get_pow{n}.csv"
    header_written = False

    with open(output_file, 'w', newline='') as fout:
        writer = csv.writer(fout)

        for model_name in model_list:
            filename = f"report_02_power_{model_name}_{n}_lay.csv"

            if not os.path.exists(filename):
                print(f"Bỏ qua: {filename} không tồn tại.")
                continue

            with open(filename, 'r', newline='') as fin:
                reader = list(csv.reader(fin))
                if not reader:
                    print(f"Bỏ qua: {filename} rỗng.")
                    continue

                if not header_written:
                    writer.writerow(['Model', 'Source_File'] + reader[0])  # Ghi tiêu đề từ dòng đầu tiên
                    header_written = True

                writer.writerow([model_name, filename] + reader[-1])  # Ghi dòng cuối cùng của file

# Gộp tất cả các get_pow*.csv thành get_all_pow.csv
all_output_file = "get_all_pow.csv"
header_written = False

with open(all_output_file, 'w', newline='') as fout:
    writer = csv.writer(fout)

    for n in range(1, 6):
        input_file = f"get_pow{n}.csv"

        if not os.path.exists(input_file):
            print(f"Bỏ qua: {input_file} không tồn tại.")
            continue

        with open(input_file, 'r', newline='') as fin:
            reader = csv.reader(fin)
            for i, row in enumerate(reader):
                if i == 0 and header_written:
                    continue  # Bỏ qua header nếu đã ghi
                elif i == 0 and not header_written:
                    header_written = True
                writer.writerow(row)

print("✅ Đã nối tất cả các file vào get_all_pow.csv.")
