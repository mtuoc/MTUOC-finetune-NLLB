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

# Dataset loading utilities
def load_moses_corpus(src_file, tgt_file, weight_file=None):
    with open(src_file, 'r', encoding='utf-8') as f_src, open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        src_sentences = f_src.readlines()
        tgt_sentences = f_tgt.readlines()
    assert len(src_sentences) == len(tgt_sentences), "Source and Target files must have the same number of lines"
    if weight_file:
        with open(weight_file, 'r', encoding='utf-8') as f_w:
            weights = [float(line.strip()) for line in f_w.readlines()]
        assert len(weights) == len(src_sentences), "Mismatch between weights and number of lines"
    else:
        weights = [1.0] * len(src_sentences)
    return src_sentences, tgt_sentences, weights

def load_multiple_corpora(corpora_config, options):
    all_src_texts, all_tgt_texts, all_src_langs, all_tgt_langs, all_weights = [], [], [], [], []
    use_weights = options.get("use_weights", False)
    for corpus in corpora_config:
        src = corpus["src_file"]
        tgt = corpus["tgt_file"]
        src_lang = corpus["src_lang_code"]
        tgt_lang = corpus["tgt_lang_code"]
        weight_file = corpus.get("weights") if use_weights else None
        src_lines, tgt_lines, weights = load_moses_corpus(src, tgt, weight_file)
        all_src_texts.extend(src_lines)
        all_tgt_texts.extend(tgt_lines)
        all_src_langs.extend([src_lang] * len(src_lines))
        all_tgt_langs.extend([tgt_lang] * len(tgt_lines))
        all_weights.extend(weights)
    return all_src_texts, all_tgt_texts, all_src_langs, all_tgt_langs, all_weights

# El resto del script continúa como está, se puede copiar y exportar desde aquí


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

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights", None)
        outputs = model(**inputs)
        loss = outputs.loss
        if weights is not None:
            logits = outputs.logits
            labels = inputs["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss_vals = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_vals = loss_vals.view(shift_labels.size())
            per_sample_loss = loss_vals.sum(dim=1) / (shift_labels != -100).sum(dim=1)
            loss = (per_sample_loss * weights).mean()
        return (loss, outputs) if return_outputs else loss

class CustomCollator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, batch):
        weights = [x.pop("weights") for x in batch if "weights" in x]
        input_texts = [x["input_text"] for x in batch]
        label_texts = [x["label_text"] for x in batch]

        encoded = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            label_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        encoded["labels"] = labels
        if weights:
            encoded["weights"] = torch.tensor(weights, dtype=torch.float32)
        return encoded

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_langs, tgt_langs, weights, tokenizer, max_length=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_langs = src_langs
        self.tgt_langs = tgt_langs
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = f"{self.src_langs[idx]} {self.src_texts[idx].strip()}"
        tgt_text = f"{self.tgt_langs[idx]} {self.tgt_texts[idx].strip()}"

        inputs = self.tokenizer(
            src_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        labels = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )["input_ids"]

        item = {key: val.squeeze() for key, val in inputs.items()}
        item["labels"] = labels.squeeze()
        if self.weights:
            item["weights"] = torch.tensor(self.weights[idx], dtype=torch.float32)
        return item

class CustomCollator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, batch):
        weights = [x.pop("weights") for x in batch if "weights" in x]
        batch = self.tokenizer.pad(batch, return_tensors="pt")
        if weights:
            batch["weights"] = torch.tensor(weights, dtype=torch.float32)
        return batch

        

if __name__ == "__main__":
    configfile = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(configfile, 'r', encoding="utf-8") as stream:
        configYAML = yaml.load(stream, Loader=yaml.FullLoader)

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
    use_weights = configYAML.get("use_weights", False)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_model_name, exist_ok=True)
    os.makedirs(output_tokenizer_name, exist_ok=True)

    if GPU_AVAILABLE:
        gpu_thread = threading.Thread(target=gpu_monitor, args=(gpu_log_file,))
        gpu_thread.daemon = True
        gpu_thread.start()

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    aliases = configYAML.get("language_aliases", {})
    if aliases:
        print("\nApplying language aliases:")
        embedding_layer = model.get_input_embeddings()
        added_tokens = []
        for new_lang, base_lang in aliases.items():
            if new_lang not in tokenizer.get_vocab():
                added_tokens.append(new_lang)
                print(f"  - Added token {new_lang} copying from {base_lang}")
        if added_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})
            model.resize_token_embeddings(len(tokenizer))
            for new_token in added_tokens:
                base_token = aliases[new_token]
                base_id = tokenizer.convert_tokens_to_ids(base_token)
                new_id = tokenizer.convert_tokens_to_ids(new_token)
                with torch.no_grad():
                    embedding_layer.weight[new_id] = embedding_layer.weight[base_id].clone()

    def load_corpora_block(config_entry, use_weights_flag):
        return load_multiple_corpora(configYAML.get(config_entry, []), {"use_weights": use_weights_flag})

    train_src, train_tgt, train_src_langs, train_tgt_langs, train_weights = load_corpora_block("corpora", use_weights)
    val_src, val_tgt, val_src_langs, val_tgt_langs, _ = load_corpora_block("val_corpora", False)

    train_dataset = TranslationDataset(train_src, train_tgt, train_src_langs, train_tgt_langs, train_weights, tokenizer, max_length)
    val_dataset = TranslationDataset(val_src, val_tgt, val_src_langs, val_tgt_langs, None, tokenizer, max_length)

    data_collator = CustomCollator(tokenizer=tokenizer, model=model)

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
        fp16=torch.cuda.is_available()
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(patience=patience)]
    )

    trainer.train()
    trainer.save_model(output_model_name)
    tokenizer.save_pretrained(output_tokenizer_name)
