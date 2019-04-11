[model_1_rnn]: https://raw.githubusercontent.com/Brandon-HY-Lin/aind2-nlp-capstone/d87103a5076703c2b891153a7b0056610a6209a8/images/rnn.png "Model 1: Simple RNN"

[model_2_rnn_embed]: https://raw.githubusercontent.com/Brandon-HY-Lin/aind2-nlp-capstone/d87103a5076703c2b891153a7b0056610a6209a8/images/embedding.png "Model 2: Embedding"



# Abstract
This work adopts 2 Bidirectional GRU and 1 fully-connected layer to translation English sentence to French sentence. The dataset is a subset of [WMT](http://www.statmt.org/). The experiment shows that the validation accuracy is 97.2% and BLEU-4 score is 0.617.


# Introduction
Machine translation is a popular topic in research. This work utilizes deep neural networks to perform word-level translation and translate English into French. Different combinations of GRU, Bidirectional GRU and fully-connected layer are implemented. Before training, words in a sentence are tokenized and encodded. Then every sentence is padded to insure equal-length.

# Architectures

* Model 1: RNN

![Model 1][model_1_rnn]

* Model 2: Embedding


# Deep Speech 2
The flow chart of [deep speech 2][deep_speech_2] is shown below. The input is spectrogram and then fed into 2-layer 2d-CNN. The result is passed into 1-layer RNN and 1 fully-connected layer. Each output of layer has batch-norm except final output.


![Flow chart of deep speech 2][flow_deep_speech_2] 

*Flow chart of __deep speech 2__*


# Implementation

* Architecture:
The architecture is (1-layer Conv2D + 2-layer Bidirectional GRU + TimeDistributed Dense). To reducing overfitting, batch-normalization and dropout layers are added between each layer. Futhurmore, the dropout feature in GRU cell is also enabled.

* Steps:
    * Input: The spectrogram with shape (None, TimeSeqLength, 161, 1).
    * Conv2D: Served as encoder to extract features.
            Stride=2, Out-channel=8, padding=same
            output shape = (None, TimeSeqLength/2, 81, 8)
    * BatchNorm: Speed-up traininig at first few epochs.
    * Dropout: Reduce strong overfitting caused by Conv2D.
    * Lambda: reshape the 4-dim data to 3-dim data in order to fit into GRU layer.
            output shape = (None, TimeSeqLength/2, 81 * 8)
    * Bidirectional GRU with dropout (layer-0) : served as decoder.
            output shape = (None, TimeSeqLength/2, 512)
    * BatchNorm: Speed-up training
    * Dropout: Reduce overfitting of 2-layer GRU
    * Bidirectional GRU with dropout (layer-1) : served as decoder.
            output shape = (None, TimeSeqLength/2, 512)
    * BatchNorm: Speed-up training
    * Dropout: Reduce overfitting of 2-layer GRU
    * TimeDistributed Dense: convert time series to characters.
            output shape = (None, TimeSeqLength/2, 29)


# Results

The model name of __deep speech 2__ is _model_cnn2d_dropout_, and the validation loss is 93.7 which is better than other combinations of CNN, RNN, and FC. The detail results are shown below.


| Model               	| Architecture                                	| Training loss @ Epoch-20 	| Validation Loss @ Epoch-20 	| Lowest Validation Loss 	| Epoch of lowest Valid Loss 	|
|---------------------	|---------------------------------------------	|--------------------------	|----------------------------	|------------------------	|----------------------------	|
| model_0             	| RNN                                         	| 778.4                    	| 752.1                      	| 752.2                  	| 20                         	|
| model_1             	| RNN + Dense                                 	| 119.1                    	| 137.3                      	| 137.3                  	| 18                         	|
| model_2             	| CNN-1D + RNN + Dense                        	| 73.4                     	| 149.8                      	| 134.5                  	| 8                          	|
| model_3             	| 3-layer RNN + Dense                         	| 93.1                     	| 134.5                      	| 132.7                  	| 16                         	|
| model_4             	| Bidirectional RNN + Dense                   	| 102.1                    	| 135.7                      	| 134.3                  	| 17                         	|
| model_cnn2d         	| CNN-2D + RNN + Dense                        	| 76.7                     	| 140.7                      	| 128.3                  	| 9                          	|
| model_deep_cnn2d    	| 2-layer CNN-2D + RNN + Dense                	| 66.5                     	| 191.9                      	| 151.1                  	| 6                          	|
| model_cnn2d_dropout 	| CNN2D + 2-layer Bidirectional GRU + Dropout 	| 75.7                     	| 93.7                       	| 93.7                   	| 20                         	|


# Conclusion
In this work, the similar architecture in [deep sppech 2][deep_speech_2] is implemented and achieve validation loss of 93.7 which is much better compared to other combinations of RNN and CNN.


# Future Works
Implement google's [WaveNet paper](https://arxiv.org/abs/1609.03499).


# Appendix
#### Hyper-Parameters

* model_cnn2d_dropout
	* CNN
		* #layer: 1
	    * #input width: 161
	    * #input channel: 1
	    * #output channel: 8
	    * kernel size: 11
	    * stride: 2
	    * padding: same
	    * dropout: 0.2
	* Bidirectional GRU
		* #layer: 2
	    * unit of neurons: 256
	    * dropout: 0.2
	* Fully-connected layer
		* layer: 1
		* input size: 256
		* output size: 29