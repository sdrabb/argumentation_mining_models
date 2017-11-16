import itertools
import preprocess
from nltk.tree import ParentedTree,Tree
import random
import os
import numpy as np
import tensorflow as tf
import pickle
import time
from sklearn.metrics import roc_auc_score

from time import sleep
import sys
import math
MODEL_STR = 'rnn_embed=%d_l2=%f_lr=%f.weights'
SAVE_DIR = './weights/'

def restart_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

class Config(object):
    """Holds models hyperparams and data information.
    Model objects are passed a Config() object at instantiation.
    """

    def __init__(self,embeddings,embedding_size):
        self.save_restore_file = False
        self.embeddings = embeddings
        self.embed_size = embedding_size
        self.label_size = 2
        self.early_stopping = 2

        self.max_epochs = 10
        self.lr = 0.00001
        self.l2 = 0.
        self.post_tag_vocab_dim = 74
        self.max_num_children = 3
        self.hsize = 250

        self.test_path = 'test_es_{}_lr_{}_l2_{}_mc_{}_hsize_{}'.format(self.embed_size,self.lr,self.l2,self.max_num_children,self.hsize).replace('.','_')

        self.predictions_dir = os.path.join(self.test_path, 'predictions' )
        self.models_dir = os.path.join(self.test_path,'models')
        self.summaries_dir = os.path.join(self.test_path, 'summaries')


        self.create_dir(self.test_path)
        self.create_dir(self.predictions_dir)
        self.create_dir(self.models_dir)

    def create_dir(self,dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


        # model_name = MODEL_STR % (embed_size, l2, lr)


class RecursiveNetStaticGraph():

    def __init__(self, config):
        self.config = config





        # add input placeholders
        self.is_preterminal_placeholder = tf.placeholder(
            tf.bool, (None), name='is_preterminal_placeholder')

        self.is_leaf_placeholder = tf.placeholder(
            tf.bool, (None), name='is_leaf_placeholder')

        self.children_placeholder = tf.placeholder(
            tf.int32, (None), name='children_placeholder')

        self.node_word_indices_placeholder = tf.placeholder(
            tf.int64, (None), name='node_word_indices_placeholder')

        self.labels_placeholder = tf.placeholder(
            tf.int32, (None), name='labels_placeholder')

        self.node_post_tag_placeholder = tf.placeholder(
            tf.int32, (None), name='node_post_tag_placeholder')



        # # add models variables
        with tf.variable_scope('Embeddings'):
            embeddings = tf.cast(tf.constant(self.config.embeddings), tf.float32)
            # embeddings = tf.cast(tf.get_variable('embeddings',initializer=self.config.embeddings),tf.float32)

        with tf.variable_scope('leaf_composition'):
            W1 = tf.get_variable('W1',[ self.config.post_tag_vocab_dim +  self.config.hsize, self.config.hsize ],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
            b1 = tf.get_variable('b1', [1, self.config.hsize],initializer=tf.zeros_initializer())

        with tf.variable_scope('node_composition'):
            W2 = tf.get_variable('W2',
                                 [ self.config.post_tag_vocab_dim +  self.config.hsize * self.config.max_num_children, self.config.hsize ],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
            b2 = tf.get_variable('b2', [1, self.config.hsize],initializer=tf.zeros_initializer())

        with tf.variable_scope('Projection'):
            U = tf.get_variable('U', [ self.config.hsize, self.config.label_size],initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32))
            bs = tf.get_variable('bs', [1, self.config.label_size],initializer=tf.zeros_initializer())

        # self.node_is_preterminal = tf.gather(self.is_preterminal_placeholder, 0)
        # self.node_is_leaf = tf.gather(self.is_leaf_placeholder, 0)
        # self.node_word_index = tf.gather(self.node_word_indices_placeholder, 0)
        #
        # self.node_pos_tag_one_hot = tf.expand_dims(tf.one_hot(tf.gather(self.node_post_tag_placeholder, 0), depth=config.post_tag_vocab_dim),0)
        # self.left_child = tf.gather(self.left_children_placeholder, 0)
        # self.right_child = tf.gather(self.right_children_placeholder, 0)



        # build recursive graph

        tensor_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)

        tensor_array = tensor_array.write(0,tf.zeros([1, self.config.hsize]))

        def embed_word(word_index,pos_tag):
            with tf.device('/cpu:0'):
                # embedded_word = tf.expand_dims(tf.gather(embeddings, word_index), 0)
                embedded_word = tf.expand_dims(tf.nn.embedding_lookup(embeddings,word_index), 0)
                padded_embedding = tf.pad(embedded_word,[[0,0],[0,self.config.hsize-tf.shape(embedded_word)[1]]])

                return padded_embedding

        def combine_preterminal(preterminal_tensor,node_pos_tag):
            return tf.nn.relu(tf.matmul(tf.concat([node_pos_tag,preterminal_tensor[0,:]],1), W1) + b1)

        def combine_nodes(children,node_pos_tag):
            return tf.nn.relu(tf.matmul( tf.concat([node_pos_tag,tf.reshape(tf.squeeze(children),[1,self.config.hsize*config.max_num_children])],1), W2) + b2)

        def combine_children(is_preterminal,children,node_pos_tag):
            return tf.cond(is_preterminal,lambda: combine_preterminal(children,node_pos_tag),lambda: combine_nodes(children,node_pos_tag))

        def loop_body(tensor_array, i):


            node_is_preterminal = tf.gather(self.is_preterminal_placeholder, i)
            node_is_leaf = tf.gather(self.is_leaf_placeholder, i)
            node_word_index = tf.gather(self.node_word_indices_placeholder, i)

            node_pos_tag_one_hot = tf.expand_dims(tf.one_hot(tf.gather(self.node_post_tag_placeholder,i),depth=config.post_tag_vocab_dim),0)

            children = self.children_placeholder[:self.config.max_num_children,i]


            node_tensor = tf.cond(
                node_is_leaf ,
                lambda: embed_word(node_word_index,node_pos_tag_one_hot),

                lambda: combine_children(node_is_preterminal,
                                         tensor_array.gather(children),
                                         node_pos_tag_one_hot))

            tensor_array = tensor_array.write(i, node_tensor)
            i = tf.add(i, 1)


            return tensor_array, i



        loop_cond = lambda tensor_array, i: \
            tf.less(i, tf.squeeze(tf.shape(self.is_leaf_placeholder)))


        self.tensor_array, _ = tf.while_loop(
            loop_cond, loop_body, [tensor_array, 1], parallel_iterations=10)



        # add projection layer
        self.root_logits = tf.matmul(
            self.tensor_array.read(self.tensor_array.size() - 1), U) + bs



        # self.root_prediction = tf.squeeze(tf.argmax(self.root_logits, 1))

        self.root_prediction = tf.nn.softmax(tf.squeeze(self.root_logits))

        # add loss layer
        regularization_loss = self.config.l2 * (tf.nn.l2_loss(W2)+
                                                tf.nn.l2_loss(W1) + tf.nn.l2_loss(U))


        self.root_loss = tf.reduce_sum( regularization_loss +
                                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                                           logits= self.root_logits, labels=self.labels_placeholder[-1:]))

        # add training op
        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(
            self.root_loss)



        with tf.name_scope('summaries'):
            with tf.name_scope('train'):
                with tf.name_scope('loss'):
                    self.train_loss = tf.placeholder(tf.float32, (None))
                    tf.summary.scalar('cross_entropy', self.train_loss)
                with tf.name_scope('auc'):
                    self.train_auc = tf.placeholder(tf.float32, (None))
                    tf.summary.scalar('auc', self.train_auc)
                    
            with tf.name_scope('test'):
                with tf.name_scope('loss'):
                    self.test_loss = tf.placeholder(tf.float32, (None))
                    tf.summary.scalar('cross_entropy', self.test_loss)
                with tf.name_scope('auc'):
                    self.test_auc = tf.placeholder(tf.float32, (None))
                    tf.summary.scalar('auc', self.test_auc)

            with tf.name_scope('val'):
                with tf.name_scope('loss'):
                    self.val_loss = tf.placeholder(tf.float32, (None))
                    tf.summary.scalar('cross_entropy', self.val_loss)
                with tf.name_scope('auc'):
                    self.val_auc = tf.placeholder(tf.float32, (None))
                    tf.summary.scalar('auc', self.val_auc)

        self.merged = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(self.config.summaries_dir)



    def train_epoch(self,data,session):


        n_examples = len(data['childrens'])
        indexes = [i for i in range(0,n_examples)]
        random.shuffle(indexes)



        epoch_loss = np.zeros([n_examples])

        start = time.time()
        for i in range(0,n_examples):

            feed_dict = {
                self.is_preterminal_placeholder: data['is_preterminals'][indexes[i]],
                self.is_leaf_placeholder: data['is_leaf'][indexes[i]],
                self.children_placeholder: data['childrens'][indexes[i]],
                self.node_word_indices_placeholder: data['nodes_word_id'][indexes[i]],
                self.node_post_tag_placeholder: data['nodes_pos_tag_id'][indexes[i]],
                self.labels_placeholder: data['labels'][indexes[i]]
            }


            _, loss = session.run([self.train_op, self.root_loss],
                               feed_dict=feed_dict)
            epoch_loss[i] = loss

            if (i%5 ==0):
                sys.stdout.write('processed data {} of {}'.format(i,n_examples))
                sys.stdout.flush()
                restart_line()



        done = time.time()
        elapsed = done - start
        print 'epoch finished train loss: {} in time: {} sec'.format(np.mean(epoch_loss),elapsed)



        return epoch_loss


    def train(self,train_set,test_set,val_set,topic,sess):

        saver = tf.train.Saver(max_to_keep=None)

        for i in range(0,self.config.max_epochs):
            print 'training epoch: {} topic {}'.format(i,topic)

            if os.path.isfile(os.path.join(self.config.models_dir,"model_topic_{}_epoch_{}.ckpt.index".format(topic,i))) and self.config.save_restore_file == True:
                # saver = tf.train.import_meta_graph(os.path.join(self.config.models_dir,"model_topic_{}_epoch_{}.ckpt".format(topic,i)))
                saver.restore(sess, os.path.join(self.config.models_dir,"model_topic_{}_epoch_{}.ckpt".format(topic,i)))
                print "model_topic_{}_epoch_{} restored".format(topic,i)
            else:
                self.train_epoch(train_set,sess)
                save_path = saver.save(sess, os.path.join(self.config.models_dir,"model_topic_{}_epoch_{}.ckpt".format(topic,i)))
                print "model_topic_{}_epoch_{} saved".format(topic, i)

                print 'predict train epoch: {} topic {}'.format(i, topic)
                train_loss, train_predictions, train_auc, train_labels = self.predict(train_set, sess)

                np.save(os.path.join(self.config.predictions_dir, 'labels_epoch_{}_topic_{}_train.npy'.format(i, topic)),train_labels)
                np.save(os.path.join(self.config.predictions_dir,'predictions_epoch_{}_topic_{}_train.npy'.format(i,topic)), train_predictions)
                np.save(os.path.join(self.config.predictions_dir,'loss_epoch_{}_topic_{}_train.npy'.format(i,topic)), train_loss)

                print 'predict test epoch: {} topic {}'.format(i, topic)
                test_loss ,test_predictions,test_auc, test_labels = self.predict(test_set,sess)

                np.save(os.path.join(self.config.predictions_dir, 'labels_epoch_{}_topic_{}_test.npy'.format(i, topic)),test_labels)
                np.save(os.path.join(self.config.predictions_dir, 'predictions_epoch_{}_topic_{}_test.npy'.format(i, topic)),test_predictions)
                np.save(os.path.join(self.config.predictions_dir, 'loss_epoch_{}_topic_{}_test.npy'.format(i, topic)),test_loss)

                print 'predict val epoch: {} topic {}'.format(i, topic)
                val_loss, val_predictions,val_auc, val_labels = self.predict(val_set,sess)

                np.save(os.path.join(self.config.predictions_dir, 'labels_epoch_{}_topic_{}_val.npy'.format(i, topic)),val_labels)
                np.save(os.path.join(self.config.predictions_dir,'predictions_epoch_{}_topic_{}_val.npy'.format(i, topic)), val_predictions)
                np.save(os.path.join(self.config.predictions_dir,'loss_epoch_{}_topic_{}_val.npy'.format(i, topic)), val_loss)

                summaries_values = {self.train_loss: np.mean(train_loss),
                                    self.train_auc: train_auc,
                                    self.test_loss: np.mean(test_loss),
                                    self.test_auc:test_auc,
                                    self.val_loss: np.mean(val_loss),
                                    self.val_auc: val_auc,

                }

                summary = sess.run(self.merged, feed_dict=summaries_values)
                self.writer.add_summary(summary,i)

    def predict(self,data,session):

        n_examples = len(data['childrens'])

        labels = np.zeros([n_examples,1])
        predictions = np.zeros([n_examples,self.config.label_size])
        loss = np.zeros([n_examples])

        start = time.time()
        for i in range(0, n_examples):

            feed_dict = {
                self.is_preterminal_placeholder: data['is_preterminals'][i],
                self.is_leaf_placeholder: data['is_leaf'][i],
                self.children_placeholder: data['childrens'][i],
                self.node_word_indices_placeholder: data['nodes_word_id'][i],
                self.node_post_tag_placeholder: data['nodes_pos_tag_id'][i],
                self.labels_placeholder: data['labels'][i]
            }

            prediction, batch_loss = session.run([self.root_prediction, self.root_loss],
                                  feed_dict=feed_dict)


            loss[i] = batch_loss
            predictions[i,:] = prediction
            labels[i,:] = data['labels'][i]

            if (i % 5 == 0):
                sys.stdout.write('processed data {} of {}'.format(i, n_examples))
                sys.stdout.flush()
                restart_line()


        done = time.time()
        elapsed = done - start
        print 'epoch finished loss: {} in time: {} sec'.format((np.mean(loss)), elapsed)


        print('\n AUC: {}'.format(roc_auc_score(labels, predictions[:,1])))

        return loss,predictions, roc_auc_score(labels, predictions[:,1]),labels





if __name__ == '__main__':

    with open('dataset.pickle', 'rb') as handle:
         dataset = pickle.load(handle)

    # print dataset['val_set']['childrens'][0][0,:]

    with open('embeddings_dim_50.pickle', 'rb') as handle:
         embeddings = pickle.load(handle)
    config = Config(embeddings['embeddings'],embeddings['embedding_dim'])

    config.save_restore_file = True

    model = RecursiveNetStaticGraph(config)

    init_op = tf.global_variables_initializer()

    if os.path.isfile('experiments.pickle'):
        with open('experiments.pickle', 'rb') as handle:
            experiments = pickle.load(handle)
    else:
        experiments = {}

    with tf.Session() as sess:

        test_topics = [key.split('_')[1] for key, value in dataset.iteritems() if 'test' in key]


        for topic in test_topics:
            sess.run(init_op)



            if 'topic_{}_'.format(dataset['topic_{}_test'.format(topic)]['topic_name'].split('/')[1])+config.test_path not in experiments:

                model.train(dataset['topic_{}_train'.format(topic)],dataset['topic_{}_test'.format(topic)],dataset['val_set'],topic,sess)
                os.rename(config.test_path,'topic_{}_'.format(dataset['topic_{}_test'.format(topic)]['topic_name'].split('/')[1])+config.test_path)

            if 'topic_{}_'.format(dataset['topic_{}_test'.format(topic)]['topic_name'].split('/')[1])+config.test_path not in experiments:
                experiments['topic_{}_'.format(dataset['topic_{}_test'.format(topic)]['topic_name'].split('/')[1])+config.test_path] = topic

            with open('experiments.pickle', 'wb') as handle:
                pickle.dump(experiments, handle, protocol=pickle.HIGHEST_PROTOCOL)