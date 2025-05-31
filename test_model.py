from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model ve tokenizer yolları
model_yolu = "./adapter"

# Tokenizer ve model yükle
tokenizer = AutoTokenizer.from_pretrained(model_yolu)
model = AutoModelForCausalLM.from_pretrained(model_yolu, device_map={"": "cpu"})

print("=== Tıbbi Asistan'a Hoş Geldiniz ===")
print("Bir soru yazın. Çıkmak için 'çık' yazın.\n")

while True:
    soru = input("Soru: ")

    if soru.lower() == "çık":
        print("Görüşmek üzere!")
        break

    # Prompt oluştur
    girdi_metni = f"Soru: {soru}\nCevap:"
    girdi = tokenizer(girdi_metni, return_tensors="pt").to("cpu")

    # Cevap üret
    with torch.no_grad():
        cikti = model.generate(
            **girdi,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    cevap = tokenizer.decode(cikti[0], skip_special_tokens=True)

    # Sadece cevabı göstermek için prompt'ı çıkar
    if "Cevap:" in cevap:
        cevap = cevap.split("Cevap:")[-1].strip()

    print(f"\n>>> Cevap: {cevap}\n")
