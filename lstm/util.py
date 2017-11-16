import numpy as np
import os

def build_dictionaries(embedding_path):
    word_to_id = {}
    embeddings = []
    embedding_dim = int(embedding_path.split('.')[2].replace('d',''))

    with open(embedding_path) as f:
        content = f.readlines()

    id_counter = 0

    ukn_embedding = listofzeros = [0] * embedding_dim
    word_to_id['<ukn>'] = id_counter
    embeddings.append(ukn_embedding)
    id_counter += 1

    print ('loading embeddings and vocabularies')

    for l in content:

        word = l.split(' ')[0]
        embedding = [float(f) for f in l.split(' ')[1:]]

        word_to_id[word] = id_counter
        embeddings.append(embedding)
        id_counter += 1


    return word_to_id, np.array(embeddings), embedding_dim

class Set:
    def __init__(self, examples, labels):
        self.examples = examples
        self.labels = labels




def load_set(set_path,word_to_id):

    print('loading '+ set_path)

    with open(set_path + '/toks.txt') as f:
        sents = f.readlines()
    ukn_counter = 1
    words_counter = 1

    n_examples = len(sents)

    print (n_examples)

    max_len = -1
    avg_len = 0

    for s in sents:
        if len(s.split(' ')) > max_len:
            max_len = len(s.split(' '))
        avg_len += len(s.split(' '))

    print (avg_len)
    print ('avg sent len: {}'.format(float(avg_len) / float(n_examples)))


    x = np.zeros((n_examples,max_len),dtype=np.int)

    samples_counter = 0
    for s in sents:
        indexes_toks = []
        for t in s.split(' '):
            try:
                indexes_toks.append(word_to_id[t.strip().lower()])

            except KeyError:
                indexes_toks.append(word_to_id['<ukn>'])
                ukn_counter += 1

            words_counter +=1

        x[samples_counter,0:len(indexes_toks)] = np.array(indexes_toks)
        samples_counter += 1

    print (samples_counter)

    with open(set_path + '/labels.txt') as f:
        labels = f.readlines()

    y = []

    for l in labels:
        y.append(int(l.split(' ')[0]))


    print ('loading: {}  uknown tokens: {} of {}'.format(set_path,ukn_counter,words_counter))
    labels = np.array(y)
    labels[labels==-1] += 1

    return Set(x, labels  )

class Config:
    def __init__(self,batch_size, learning_rate, sequence_length, num_cell, num_layers, reg_constant,embeddings_fname,restore_model,save_model):
        self.save_model = save_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.num_cell = num_cell
        self.num_layers = num_layers
        self.restore_model = restore_model
        self.reg_constant = reg_constant
        self.output_dir = 'log/' + 'test_tesi_bs_{}_lr_{}_nl_{}_nc_{}_sl_{}'.format(batch_size,learning_rate,num_layers,num_cell,sequence_length) + '/'
        self.model_dir = self.output_dir + 'models/'
        self.predictions_dir = self.output_dir + 'predictions/'
        self.embeddings_fname = 'data/glove/' + embeddings_fname
        create_folder(self.output_dir)
        create_folder(self.model_dir)
        create_folder(self.predictions_dir)
        self.save_config_to_file()

    def save_config_to_file(self):
        info = 'lr: {} \n sequence len: {} \n num_cell: {} \n num_layers: {} \n reg_costant: {} \n '.format(self.learning_rate,
                                                                                                            self.sequence_length,
                                                                                                            self.num_cell,
                                                                                                            self.num_layers,
                                                                                                            self.reg_constant)
        with open(self.output_dir + 'info.txt', 'w') as file:
            file.write(info)

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print 'folder arleady exists'

