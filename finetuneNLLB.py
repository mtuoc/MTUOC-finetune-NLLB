from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
import torch
from torch.utils.data import Dataset
import yaml
import sys

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

# Load Moses-format parallel corpus
def load_moses_corpus(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as f_src, open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        src_sentences = f_src.readlines()
        tgt_sentences = f_tgt.readlines()
    assert len(src_sentences) == len(tgt_sentences), "Source and Target files must have the same number of lines"
    return src_sentences, tgt_sentences

# Load config
if len(sys.argv) > 1:
    configfile = sys.argv[1]
else:
    configfile = "config.yaml"

stream = open(configfile, 'r', encoding="utf-8")
configYAML = yaml.load(stream, Loader=yaml.FullLoader)

train_src_file = configYAML["train_src_file"]
train_tgt_file = configYAML["train_tgt_file"]
val_src_file = configYAML["val_src_file"]
val_tgt_file = configYAML["val_tgt_file"]
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
src_lang_code = configYAML["src_lang_code"]
tgt_lang_code = configYAML["tgt_lang_code"]

# Load sentences
train_src_sentences, train_tgt_sentences = load_moses_corpus(train_src_file, train_tgt_file)
val_src_sentences, val_tgt_sentences = load_moses_corpus(val_src_file, val_tgt_file)

# Custom dataset
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx].strip()
        tgt_text = self.tgt_texts[idx].strip()

        # Tokenize source and target sentences
        model_inputs = self.tokenizer(
            src_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        labels = self.tokenizer(
            text_target=tgt_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )["input_ids"]

        model_inputs["labels"] = labels.squeeze()
        return {key: val.squeeze() for key, val in model_inputs.items()}

# Ensure the model is properly set up for NLLB
#model = MarianMTModel.from_pretrained(model_name)

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    src_lang=src_lang_code,
    tgt_lang=tgt_lang_code,
)


# Create datasets
train_dataset = TranslationDataset(train_src_sentences, train_tgt_sentences, tokenizer, max_length=max_length)
val_dataset = TranslationDataset(val_src_sentences, val_tgt_sentences, tokenizer, max_length=max_length)

# Training arguments
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
    fp16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(patience=patience)]
)

# Train and save
trainer.train()
trainer.save_model(output_model_name)
tokenizer.save_pretrained(output_tokenizer_name)
