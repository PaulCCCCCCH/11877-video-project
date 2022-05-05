from transformers import BertTokenizer, GPT2Tokenizer


bert = BertTokenizer.from_pretrained('bert-base-uncased')
token_dict = bert("Hello, my name is paul.", padding='max_length', max_length=77, truncation=True).items()
print(token_dict)

gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
gpt2.pad_token = gpt2.eos_token
token_dict = gpt2("Hello, my name is paul.", padding='max_length', max_length=77, truncation=True).items()
print(token_dict)