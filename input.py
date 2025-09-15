# Step 1 preprocessing the input 
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2") # we will be the using the one used by GPT-2
# print(tokenizer.n_vocab) # 50257
# x = tokenizer.encode("Hello, my dog is cute sgaskfhasih")

# creating input target pairs
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

enc_text = tokenizer.encode(text)
print(len(enc_text)) 