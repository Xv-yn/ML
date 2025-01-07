from bpe_tokenizer import get_encoder

print("Input Text: ")
text = "Hello, world. Is this-- a test?"

orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")

integers = orig_tokenizer.encode(text)

print("Encoded: ")
print(integers)

strings = orig_tokenizer.decode(integers)

print("Decoded: ")
print(strings)
