import tensorflow as tf

shakespeare_url = "https://homl.info/shakespeare"                                                         # webpage for text 
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

print(shakespeare_text[:80])

#encoding of text
text_vec_layer = tf.keras.layers.TextVectorization(split="character",standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]
