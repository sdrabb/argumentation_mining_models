# Argumentation Mining Models

This repository provides three types of Neural Networks that can be used to solve the Context Independent Claim Detection.
We employ our models on the  [IBM dataset](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml) and for each of them we consider the pretrained word embeddings built with [*Glove model*](https://nlp.stanford.edu/projects/glove/). For the **Tree-LSTM** model we follow the code of the Stanford [Tree-Structured Long Short-Term Memory Networks](https://github.com/stanfordnlp/treelstm).


## Repository Structure
* [LSTM](https://github.com/sdrabb/argumentation_mining_models/tree/master/lstm) implementation of the *LSTM*: the model is defined in the file **lstm.py**, **scores.py** is used to evaluate the model. The considered topics are listed in **considered_topic.txt**.

* [RNN](https://github.com/sdrabb/argumentation_mining_models/tree/master/rnn) implementation of the *RNN*: the model is defined in the file **rnn.py**, **scores_and_charts.py** is used to evaluate the model. 

* [Tree-LSTM](https://github.com/sdrabb/argumentation_mining_models/tree/master/tree_lstm) contains the changes made to the [Tree-Structured Long Short-Term Memory Networks](https://github.com/stanfordnlp/treelstm) to fit their implementation to our task. 


## Built With

* [Tensorflow](https://www.tensorflow.org/) - *Long Short Term Memory* and *Recursive Neural Network*
* [Torch](http://torch.ch/) - *Tree Structured Long Short Term Memory*

## References 

* Argumentation Mining (https://dl.acm.org/citation.cfm?id=2850417)
* Tree-structured Long Short-Term Memory networks (https://arxiv.org/abs/1503.00075)
