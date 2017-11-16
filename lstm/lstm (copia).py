import tensorflow as tf
import numpy as np
import sys
from sklearn.metrics import roc_auc_score

class LSTM:
    def __init__(self,embeddings, config):
        self.config = config
        # self.batch_size = config.batch_size
        # self.sequence_length = config.sequence_lenght
        # self.num_layers = config.num_layers
        # self.num_cell = config.num_cell
        self.embeddings = embeddings
        # self.learning_rate = config.learning_rate
        self.num_classes = 2
        # self.reg_constant = config.reg_constant
        # self.output_dir = config.output_dir
        # self.model_dir = config.model_dir
        # self.predictions_dir = config.predictions_dir
        self.build_lstm()




    def build_lstm(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/gpu:0'):
                # Define weights
                weights = {
                    'out': tf.Variable(tf.truncated_normal([self.config.num_cell, self.num_classes],-0.1,0.1))
                }
                biases = {
                    'out': tf.Variable(tf.zeros([self.num_classes]))
                }

                self.data = tf.placeholder(tf.int32, [None, self.config.sequence_length])
                self.input = tf.nn.embedding_lookup(tf.constant(self.embeddings), self.data)


                self.label = tf.placeholder(tf.int32, [None, 1])
                self.target = tf.one_hot(self.label,depth=2)

                # # Define a lstm cell with tensorflow
                # cell = tf.contrib.rnn.BasicLSTMCell(self.num_cell, state_is_tuple=True)

                stacked_rnn = []
                for iiLyr in range(self.config.num_layers):
                    stacked_rnn.append(tf.contrib.rnn.BasicLSTMCell(self.config.num_cell,state_is_tuple=True))
                multilayer_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)



                # Get lstm cell output
                outputs, states = tf.nn.dynamic_rnn(multilayer_cell, tf.cast(self.input,tf.float32), dtype=tf.float32)

                transposed_output = tf.transpose(outputs,[1,0,2])

                #take last output
                last = transposed_output[-1,:,:]

                logits = tf.matmul(last, weights['out']) + biases['out']

                self.prediction = tf.nn.softmax(logits)

                # cross_entropy = - tf.reduce_sum(self.target * tf.log(tf.clip_by_value(self.prediction, 1e-10, 1.0)))

                cross_entropy = - tf.reduce_sum(self.target * tf.log(self.prediction))

                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

                self.loss = cross_entropy + self.config.reg_constant * sum(reg_losses)



                learning_rate = tf.Variable(self.config.learning_rate)
                optimizer = tf.train.AdamOptimizer( learning_rate = learning_rate )
                self.minimize = optimizer.minimize(self.loss)



    def train_an_epoch(self, dataset,session,topic_number,epoch_number ):
        n_examples = dataset.examples.shape[0]

        # n_examples = 103

        indexes = np.arange(0,n_examples)
        np.random.shuffle(indexes)

        n_batch = -1
        if (n_examples % self.config.batch_size)==0:
            n_batch = n_examples / self.config.batch_size
        else:
            n_batch = (n_examples / self.config.batch_size)+1

        loss = 0

        saver = tf.train.Saver()


        if self.config.restore_model:
            saver.restore(session, self.config.model_dir +'model_topic_{}_epoch_{}.ckpt'.format(topic_number,epoch_number))
            print ('\n restored model_topic_{}_epoch_{}.ckpt'.format(topic_number,epoch_number))

        else:

            batch_count = .0
            for i in range(0,n_batch):

                if (((i*self.config.batch_size)+self.config.batch_size) <= n_examples):
                    adaptive_batch_size = self.config.batch_size

                if (((i * self.config.batch_size) + self.config.batch_size) > n_examples):
                    adaptive_batch_size = n_examples - ((i*self.config.batch_size))

                # print ((i*batch_size) + adaptive_batch_size)

                batch_x = dataset.examples[indexes[(i*self.config.batch_size):(i*self.config.batch_size) + adaptive_batch_size -1],0:self.config.sequence_length]



                batch_y = np.expand_dims(dataset.labels[indexes[(i*self.config.batch_size):(i*self.config.batch_size) + adaptive_batch_size - 1]], axis=1)

                feed_dict = {self.data: batch_x, self.label: batch_y}
                batch_loss, m, =session.run([self.loss, self.minimize], feed_dict=feed_dict)

                loss += batch_loss
                batch_count += 1

                sys.stdout.write('\r trained examples: {} / {}'.format(((i*self.config.batch_size) + adaptive_batch_size),n_examples))

            if self.config.save_model:
                saver.save(session,self.config.model_dir +'model_topic_{}_epoch_{}.ckpt'.format(topic_number,epoch_number))

            print('\n TRAINING SET loss: {}'.format(loss/float(batch_count)))


    def evaluate_set(self, dataset, session):

        n_examples = dataset.examples.shape[0]

        if (n_examples % self.config.batch_size)==0:
            n_batch = n_examples / self.config.batch_size
        else:
            n_batch = (n_examples / self.config.batch_size)+1

        batch_losses = np.zeros((n_batch ))
        predictions = np.zeros((n_examples))

        batch_count = 0

        for i in range(0, n_batch):

            if (((i*self.config.batch_size)+self.config.batch_size) <= n_examples):
                adaptive_batch_size = self.config.batch_size
            if (((i * self.config.batch_size) + self.config.batch_size) > n_examples):
                adaptive_batch_size = n_examples - ((i*self.config.batch_size))

            batch_x = dataset.examples[(i*self.config.batch_size):(i*self.config.batch_size) + adaptive_batch_size - 1, 0:self.config.sequence_length]
            batch_y = np.expand_dims(dataset.labels[(i*self.config.batch_size):(i*self.config.batch_size) + adaptive_batch_size - 1], axis=1)

            feed_dict = {self.data: batch_x, self.label: batch_y}
            p, batch_loss,t = session.run([self.prediction, self.loss,self.target],feed_dict=feed_dict)


            predictions[(i*self.config.batch_size):(i*self.config.batch_size) + adaptive_batch_size -1] = p[:,1]
            batch_losses[batch_count] = batch_loss
            batch_count += 1

            sys.stdout.write('\r forwarded examples: {} / {}'.format(((i * self.config.batch_size) + adaptive_batch_size), n_examples))




        # print('\n loss: {}'.format(np.mean(batch_losses)))
        print('\n AUC: {}'.format(roc_auc_score(dataset.labels, predictions)))

        return predictions
