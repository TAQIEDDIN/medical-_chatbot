from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# المسارات
model_name = "./base_model"
output_dir = "./adapter"
data_path = "./medical_qa.jsonl"

# تحميل البيانات
dataset = load_dataset("json", data_files=data_path, split="train")

# ناخد فقط أول 5000 مثال
dataset = dataset.select(range(14000))

# تجهيز نص التدريب (دمج instruction مع output)
def preprocess(example):
    full_text = f"Soru: {example['instruction']}\nCevap: {example['output']}"
    return {"text": full_text}

dataset = dataset.map(preprocess)

# تحميل التوكنيزر والموديل مع تعيين استخدام CPU فقط
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"}  # إجبار على استخدام CPU فقط
)

model = prepare_model_for_kbit_training(base_model)

# إعداد LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# تجهيز البيانات بالتوكنيزيشن مع labels وخفض طول النص لـ 256 لتسريع التدريب
def tokenize_function(example):
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# إعداد التدريب مع batch صغير و gradient accumulation لتقليل استهلاك الذاكرة
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=20,
    save_steps=100,
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=False,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# بدء التدريب
trainer.train()

# حفظ الموديل والتوكنيزر
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
