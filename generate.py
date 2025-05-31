from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from peft import PeftModel

base_model = "./base_model"
adapter_path = "./adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", load_in_4bit=True)
model = PeftModel.from_pretrained(model, adapter_path)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Soru: Mide ağrısı için ne yapmalıyım?\nCevap:"
result = generator(prompt, max_new_tokens=150, temperature=0.7, top_p=0.9, do_sample=True)

print(result[0]["generated_text"])
