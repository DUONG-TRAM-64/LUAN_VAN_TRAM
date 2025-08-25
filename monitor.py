import os, time, psutil, subprocess, threading, GPUtil

# ---------------- GPU ---------------- #
def start_gpu_monitor(log_path):
    if os.path.exists(log_path): os.remove(log_path)
    fd = open(log_path, 'a')
    proc = subprocess.Popen(
        ["nvidia-smi","--loop=1",
         "--query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used",
         "--format=csv,noheader,nounits"],
        stdout=fd, stderr=subprocess.STDOUT, universal_newlines=True
    )
    try:
        gpus = GPUtil.getGPUs()
        gpu_before = gpus[0].memoryUsed if gpus else 0.0
    except:
        gpu_before = 0.0
    return proc, fd, gpu_before

# ---------------- CPU ---------------- #
def start_cpu_monitor(log_path):
    if os.path.exists(log_path): os.remove(log_path)
    stop_event = threading.Event()

    def cpu_loop():
        with open(log_path, 'w') as f:
            f.write("timestamp,cpu_percent,memory_percent\n")
            while not stop_event.is_set():
                ts = time.strftime("%Y/%m/%d %H:%M:%S")
                f.write(f"{ts},{psutil.cpu_percent(0.1)},{psutil.virtual_memory().percent}\n")
                f.flush(); time.sleep(1)

    t = threading.Thread(target=cpu_loop)
    t.start()
    return t, stop_event

# ---------------- START / END ---------------- #
def start_monitor(out_dir, model, n_layers):
    gpu_log = os.path.join(out_dir, f"gpu_usage_log_{model}_{n_layers}_layers.csv")
    cpu_log = os.path.join(out_dir, f"cpu_usage_log_{model}_{n_layers}_layers.csv")

    gpu_proc, gpu_fd, gpu_before = start_gpu_monitor(gpu_log)
    cpu_thread, stop_event = start_cpu_monitor(cpu_log)

    ram_before = psutil.Process(os.getpid()).memory_info().rss/1024**2
    start_time = time.time()
    return gpu_proc, ram_before, gpu_before, start_time, gpu_fd, cpu_thread, stop_event

def end_monitor(gpu_proc, ram_before, gpu_before, start_time, gpu_fd, cpu_thread, stop_event):
    stop_event.set(); cpu_thread.join(timeout=2)
    if gpu_proc: gpu_proc.terminate(); gpu_fd.close()

    total_time = time.time() - start_time
    ram_inc = psutil.Process(os.getpid()).memory_info().rss/1024**2 - ram_before
    try:
        gpus = GPUtil.getGPUs()
        gpu_after = gpus[0].memoryUsed if gpus else 0.0
    except:
        gpu_after = 0.0
    gpu_inc = gpu_after - gpu_before
    return total_time, ram_inc, gpu_inc
