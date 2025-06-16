from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, TrainerCallback
from transformers import DataCollatorForSeq2Seq
from transformers.trainer import Trainer
import torch
from torch.utils.data import Dataset
import yaml
import sys
import time
import threading
import os
import warnings
import numpy as np

# Try to import pynvml, only if GPU is available
try:
    import pynvml
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

# Early Stopping Callback
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_score = float('inf')
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            eval_loss = metrics.get('eval_loss')
            if eval_loss is not None:
                if eval_loss < self.best_score:
                    self.best_score = eval_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        control.should_early_stop = True
        return control

# GPU Monitor
def gpu_monitor(log_file_path):
    if not GPU_AVAILABLE:
        return
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    with open(log_file_path, 'w') as log_file:
        log_file.write("time,gpu_id,memory_used_MB,utilization_percent,temperature_C,power_draw_W\n")
        try:
            while True:
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{i},{memory_info.used // (1024 * 1024)},{utilization.gpu},{temperature},{power_draw}\n")
                    log_file.flush()
                time.sleep(1)
        except Exception as e:
            print(f"GPU Monitor Stopped: {e}")
        finally:
            pynvml.nvmlShutdown()

# Dataset loading functions (iguals que abans)...
# ...
# ...

# Load config
if len(sys.argv) > 1:
    configfile = sys.argv[1]
else:
    configfile = "config.yaml"

with open(configfile, 'r', encoding="utf-8") as stream:
    configYAML = yaml.load(stream, Loader=yaml.FullLoader)

# Params
model_name = configYAML["model_name"]
max_length = configYAML["max_length"]
output_dir = configYAML["output_dir"]
eval_strategy = configYAML["eval_strategy"]
save_strategy = configYAML["save_strategy"]
per_device_train_batch_size = configYAML["per_device_train_batch_size"]
per_device_eval_batch_size = configYAML["per_device_eval_batch_size"]
num_train_epochs = configYAML["num_train_epochs"]
weight_decay = configYAML["weight_decay"]
logging_dir = configYAML["logging_dir"]
logging_steps = configYAML["logging_steps"]
load_best_model_at_end = configYAML["load_best_model_at_end"]
metric_for_best_model = configYAML["metric_for_best_model"]
patience = configYAML["patience"]
output_model_name = configYAML["output_model_name"]
output_tokenizer_name = configYAML["output_tokenizer_name"]
gpu_log_file = configYAML.get("gpu_log_file", "gpu_usage.log")

# Monitor GPU only if available
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_model_name, exist_ok=True)
os.makedirs(output_tokenizer_name, exist_ok=True)
if GPU_AVAILABLE:
    gpu_thread = threading.Thread(target=gpu_monitor, args=(gpu_log_file,))
    gpu_thread.daemon = True
    gpu_thread.start()

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply language aliases (igual)...
# Load corpora (igual)...
# Dataset i collator (igual)...

# Training arguments amb fp16 adaptat
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy=eval_strategy,
    save_strategy=save_strategy,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    logging_dir=logging_dir,
    logging_steps=logging_steps,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model=metric_for_best_model,
    save_total_limit=2,
    fp16=torch.cuda.is_available()  # Només activa fp16 si hi ha GPU
)

# Trainer
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(patience=patience)]
)

# Train and save
trainer.train()
trainer.save_model(output_model_name)
tokenizer.save_pretrained(output_tokenizer_name)
