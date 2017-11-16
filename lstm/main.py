from util import build_dictionaries
from util import load_set
from lstm import LSTM
from util import Config
import numpy as np
import tensorflow as tf





if __name__ == '__main__':

    config = Config(
        batch_size=10,
        learning_rate=0.00001,
        num_layers=3,
        num_cell=300,
        sequence_length=35,
        reg_constant=0.0,
        embeddings_fname='glove.27B.50d.txt',

        restore_model = False,
        save_model = False
    )




    # load embedding
    word_to_id, embeddings, embedding_dim = build_dictionaries(config.embeddings_fname)



    for topic in range(38,39):
        # load dataset
        training_set = load_set('data/arg_mining_new/topic_{}_train'.format(topic),word_to_id)
        test_set = load_set('data/arg_mining_new/topic_{}_test'.format(topic), word_to_id)
        validation_set = load_set('data/arg_mining_new/validation',word_to_id)



        lstm = LSTM(embeddings,config)

        with tf.Session(graph=lstm.graph) as session:


            session.run(tf.global_variables_initializer())

            for epoch in range(0, 20):

                print ('train epoch {}'.format(epoch))
                lstm.train_an_epoch(training_set,session=session,epoch_number=epoch,topic_number=topic)

                print ('testing Training SET epoch {}'.format(epoch))
                p_training = lstm.evaluate_set(training_set, session=session)

                print ('testing VALIDATION SET epoch {}'.format(epoch))
                p_validation = lstm.evaluate_set(validation_set, session=session)

                print ('testing TEST SET epoch {}'.format(epoch))
                p_test = lstm.evaluate_set(test_set, session)

                np.save(config.predictions_dir + 'predictions_training_topic{}_epoch_{}.npy'.format(topic, epoch),p_training)
                np.save(config.predictions_dir + 'predictions_validation_topic{}_epoch_{}.npy'.format(topic, epoch),p_validation)
                np.save(config.predictions_dir + 'predictions_test_topic{}_epoch_{}.npy'.format(topic, epoch),p_test)

                if (epoch==0):
                    np.save(config.predictions_dir + 'training_labels_topic_{}.npy'.format(topic),training_set.labels)
                    print training_set.labels.shape

                    np.save(config.predictions_dir+'validation_labels_topic_{}.npy'.format(topic),validation_set.labels)
                    np.save(config.predictions_dir + 'test_labels_topic_{}.npy'.format(topic), test_set.labels)
