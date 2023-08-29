#@ Implementation of character tokenizer 
with open("wizard_of_oz.txt", 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))           # provide sorted set of all characters present in wizard_of_oz.txt
print(chars)
print(len(chars))

# character tokenizer : less vocab
string_to_int = { ch:i for i,ch in enumerate(chars)}      # encoding each character from 0 to len(chars) e.g 'A' : 0
int_to_string = {i:ch for i,ch in enumerate(chars)}       # encoding each index from starting character to end 0 : 'A'
encode = lambda s: [string_to_int[c] for c in s]          # lamda function : takes string and encode each char to integer
decode = lambda l: ''.join([int_to_string[i] for i in l]) # takes a list of integer and encode it to char, form string using .join()

encoded_hello = encode('hello')
encoded_hello
decoded_number = decode(encoded_hello)
decoded_number

