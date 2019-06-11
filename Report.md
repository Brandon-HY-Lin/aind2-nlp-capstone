[model_1_rnn]: https://raw.githubusercontent.com/Brandon-HY-Lin/aind2-nlp-capstone/d87103a5076703c2b891153a7b0056610a6209a8/images/rnn.png "Model 1: Simple RNN"

[model_2_rnn_embed]: https://raw.githubusercontent.com/Brandon-HY-Lin/aind2-nlp-capstone/d87103a5076703c2b891153a7b0056610a6209a8/images/embedding.png "Model 2: Embedding"

[model_3_bi_rnn]: https://raw.githubusercontent.com/Brandon-HY-Lin/aind2-nlp-capstone/d87103a5076703c2b891153a7b0056610a6209a8/images/bidirectional.png "Model 3: Bidirectional RNNs"

[model_4_encoder_decoder]: https://github.com/Brandon-HY-Lin/aind2-nlp-capstone/blob/master/images/encoder_decoder.png "Model 4: Encoder-Decoder"

# Project: Machine Translation

# Abstract
This work adopts 2 Bidirectional GRU and 1 fully-connected layer to translation English sentence to French sentence. The dataset is a subset of [WMT](http://www.statmt.org/). The experiment shows that the validation accuracy is 97.2%.


# Introduction
Machine translation is a popular topic in research. This work utilizes deep neural networks to perform word-level translation and translate English into French. Different combinations of GRU, Bidirectional GRU and fully-connected layer are implemented. Before training, words in a sentence are tokenized and encodded. Then every sentence is padded to insure equal-length. The max English sentence length is 15 and max French sentence length is 21. Note that, the vocabulary sizes of English and French are 199 and 344, respectively.


# Architectures
There are 4 basic type of architectures. In this project, all 4 types are implemented. Furthurmore, the combinations of these 4 types are also implemented.

* Model 1: RNN

![Model 1][model_1_rnn]

code snippet:
```
x_rnn = keras.layers.GRU(units=hidden_size, return_sequences=True)(x)
y = keras.layers.TimeDistributed(keras.layers.Dense(output_size, activation='softmax'))(x_rnn)
```

- Model 2: Embedding

Adding embedding layer after input.

![Model 2][model_2_rnn_embed]

code snippet:
```
x_embed = keras.layers.embeddings.Embedding(input_dim=input_size, output_dim=embed_size, input_length=sequence_length)(x)
x_rnn = keras.layers.GRU(units=hidden_size, return_sequences=False)(x_embed)
y = keras.layers.TimeDistributed(keras.layers.Dense(output_size, activation='softmax'))(x_rnn)
```

* Model 3: Bidirectional RNN

One restriction of a RNN is that it can't see the future input, only the past. This is where bidirectional recurrent neural networks come in. They are able to see the future data.

![Model 3][model_3_bi_rnn]

code snippet:
```
x_embed = keras.layers.embeddings.Embedding(input_dim=input_size, output_dim=embed_size, input_length=sequence_length)(x)
x_rnn = keras.layers.Bidirectional(keras.layers.GRU(units=hidden_size, return_sequences=True))(x_embed)
y = keras.layers.TimeDistributed(keras.layers.Dense(output_size, activation='softmax'))(x_rnn)
```

* Model 4: Encoder-Decoder

This model is made up of an encoder and decoder. The encoder creates a matrix representation of the sentence. The decoder takes this matrix as input and predicts the translation as output.

![Model 4][model_4_encoder_decoder]


code snippet:
```
x_embed = keras.layers.embeddings.Embedding(input_dim=input_size, output_dim=embed_size, input_length=sequence_length)(x)
x_rnn_encoder = keras.layers.GRU(units=hidden_size, return_sequences=False)(x_bembed)

x_rep = keras.layers.RepeatVector(sequence_length)(x_rnn_encoder)

x_rnn_decoder = keras.layers.GRU(units=hidden_size, return_sequences=True)(x_rep)
y = keras.layers.TimeDistributed(keras.layers.Dense(output_size, activation='softmax'))(x_rnn_decoder)
```

# Results
The results are shown in the following table. The best result is 2 bidirectional GRU + 1 fully-connected layer as shown in the last row. The validation accuracy is 0.9726.


| Architecture                      | With Embedding    | Note              | val_acc   |
|-------------------------------    |----------------   |-----------------  |---------  |
| Simple RNN                        | No                | hidden_size=256   | 0.6841    |
| Simple RNN                        | Yes               | hidden_size=256   | 0.9087    |
| Bidirectional RNN                 | No                | N/A               | 0.7211    |
| Bidirectional RNN                 | Yes               | N/A               | 0.9429    |
| Encoder-Decoder RNN               | No                | concat mode       | 0.6427    |
| Encoder-Decoder RNN               | Yes               | concat mode       | 0.6813    |
| Recursive Encoder-Decoder RNN     | Yes               | sum mode          | 0.904     |
| 2 Bidir GRU + FC                  | Yes               | N/A               | 0.9726    |


# Conclusion
In this work, different architectures are implemented. The architecture with highest validation accuracy is 2 layer of bidirectional GRU followed by 1 fully-connected layer. Its validation accuracy is 97.26%.


# Future Works
Implement attention models:
  * [Luong's work](https://arxiv.org/abs/1508.04025)
  * [Bahdanau' work](https://arxiv.org/abs/1409.0473)


# Appendix
#### Hyper-Parameters

* 2 Bidirectional GRU + FC
    * Embedding:
        * input size: 199
        * embed size: 128
	* Bidirectional GRU 1
	    * unit of neurons: 256
    * Bidirectional GRU 2
        * unit of neurons: 256
	* Fully-connected layer
		* layer: 1
		* input size: 256
		* output size: 344
		
		
#### Key APIs
* Tokenize: keras.preprocessing.text.Tokenizer().fit_on_texts(text).texts_to_sequences(text)
* Dictionary: keras.preprocessing.text.Tokenizer().fit_on_texts(text).texts_to_sequences(text).word_index
* Padding: keras.preprocessing.sequence.pad_sequences(sequences, maxlen, padding='post')
* Keras:
	* keras.layers.Input(input_size)
	* keras.layers.Embedding(input_dim, output_dim, input_length)
	* keras.layers.GRU(units=hidden_size, return_sequences=True)
	* keras.layers.TimeDistributed()
	* keras.layers.RepeatVector()  # for encoder-decoder architecture
	* keras.layers.Dense()
	* keras.models.Model(inputs, outputs)
	
