import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import re
import matplotlib.pyplot as plt
import seaborn as sns


def draw_chart(chart_name,measure,axis,val_ordinate,train_ordinate,test_ordinate,dst_folder):
    plt.style.use('seaborn')
    sns.set(font_scale=1.2)
    sns.set_style({'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(8, 8))
    ttl = ax.title
    ttl.set_position([.5, 1.05])
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)

    ax.set_title(' '.join(chart_name.replace('_test_es_50_lr_1e-05_l2_0_0_mc_3_hsize_250','').split('_')[1:]))
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 1)
    plt.xticks(np.arange(0, 11, 1))
    ax.set_xlim(0, 10)
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel(measure)
    if val_ordinate is not None:
        ax.plot(axis, val_ordinate, color=sns.xkcd_rgb["pale red"], marker='.', label='validation')  # plotting t, a separately
    if train_ordinate is not None:
        ax.plot(axis, train_ordinate,color=sns.xkcd_rgb["medium green"],  marker='.', label='train')  # plotting t, b separately
    if test_ordinate is not None:
        ax.plot(axis, test_ordinate, color=sns.xkcd_rgb["denim blue"], marker='.', label='test')  # plotting t, c separately
    ax.legend()
    plt.savefig(os.path.join(dst_folder,'{}_{}.pdf'.format('_'.join(chart_name.split('_')[1:]).replace('_test_es_50_lr_1e-05_l2_0_0_mc_3_hsize_250',''),measure)),dpi=300,bbox_inches='tight')
    # plt.show()

def evaluate():
    dirs = [os.path.join(d,'predictions') for d in os.listdir('.') if os.path.isdir(os.path.join('.', d))]

    best_auc_scores = []
    best_precision_scores = []
    best_recall_scores = []

    with open('considered_topic.txt') as f:
        considered_topic = f.readlines()
    considered_topic = [x.strip() for x in considered_topic]


    for j,d in enumerate(dirs):

        if d.replace('topic_','').replace('_test_es_50_lr_1e-05_l2_0_0_mc_3_hsize_250/predictions','') not in considered_topic:
            print considered_topic
            continue


        best_val_auc = 0.
        best_epoch = 0



        p_train_files = sorted([filename for filename in os.listdir(d) if filename.startswith('predictions_epoch_') and filename.split('_')[5] == 'train.npy'])
        l_train_files = sorted([filename for filename in os.listdir(d) if filename.startswith('labels_epoch_') and filename.split('_')[5] == 'train.npy'])

        p_val_files = sorted([filename for filename in os.listdir(d) if filename.startswith('predictions_epoch_') and filename.split('_')[5]=='val.npy'])
        l_val_files = sorted([filename for filename in os.listdir(d) if filename.startswith('labels_epoch_') and filename.split('_')[5]=='val.npy'])

        p_test_files = sorted([filename for filename in os.listdir(d) if filename.startswith('predictions_epoch_') and filename.split('_')[5] == 'test.npy'])
        l_test_files = sorted([filename for filename in os.listdir(d) if filename.startswith('labels_epoch_') and filename.split('_')[5]=='test.npy'])

        x = np.arange(len(l_train_files) + 1)
        val_y = np.zeros((len(l_train_files) + 1, 4))
        test_y = np.zeros((len(l_train_files) + 1, 4))
        train_y = np.zeros((len(l_train_files) + 1, 4))


        print '------------- {} --------------'.format(d)
        for i,f in enumerate(p_val_files):

            p_val_scores = np.load(os.path.join(d,p_val_files[i]))
            l_val = np.load(os.path.join(d,l_val_files[i]))

            val_top_predictions_scores = p_val_scores[:,1].argsort()[::-1]
            val_top_labels = l_val[val_top_predictions_scores]

            val_precision_at_200 = np.sum(val_top_labels[:200]) / 200.
            val_recall_at_200 = np.sum(val_top_labels[:200]) / float(np.sum(l_val))

            p_test_scores = np.load(os.path.join(d, p_test_files[i]))
            l_test = np.load(os.path.join(d, l_test_files[i]))

            test_top_predictions_scores = p_test_scores[:, 1].argsort()[::-1]
            test_top_labels = l_test[test_top_predictions_scores]

            test_precision_at_200 = np.sum(test_top_labels[:200]) / 200.
            test_recall_at_200 = np.sum(test_top_labels[:200]) / float(np.sum(l_test))

            p_train_scores = np.load(os.path.join(d, p_train_files[i]))
            l_train = np.load(os.path.join(d, l_train_files[i]))

            train_top_predictions_scores = p_train_scores[:, 1].argsort()[::-1]
            train_top_labels = l_train[train_top_predictions_scores]

            train_precision_at_200 = np.sum(train_top_labels[:200]) / 200.
            train_recall_at_200 = np.sum(train_top_labels[:200]) / float(np.sum(l_train))

            val_scores = (roc_auc_score(l_val, p_val_scores[:, 1]), val_precision_at_200, val_recall_at_200, 2*((val_precision_at_200*val_recall_at_200)/(val_precision_at_200+val_recall_at_200)))
            test_scores = (roc_auc_score(l_test, p_test_scores[:, 1]), test_precision_at_200, test_recall_at_200,2*((test_precision_at_200*test_recall_at_200)/(test_precision_at_200+test_recall_at_200)))
            train_scores = (roc_auc_score(l_train, p_train_scores[:, 1]), train_precision_at_200,train_recall_at_200, 2*((train_precision_at_200*train_recall_at_200)/(train_precision_at_200+train_recall_at_200)))


            val_y[i + 1, :] = val_scores
            train_y[i + 1, :] = train_scores
            test_y[i + 1, :] = test_scores

            # print 'VAL -> AUC: {} , P: {} , R: {} {0: >5} TEST -> AUC: {}, P: {} , R: {}'.format(roc_auc_score(l_val,p_val_scores[:,1]),val_precision_at_200,val_recall_at_200,roc_auc_score(l_test,p_test_scores[:,1]),test_precision_at_200,test_recall_at_200)

            current_auc_val = roc_auc_score(l_val,p_val_scores[:,1])

            if current_auc_val > best_val_auc :
                best_val_auc = current_auc_val
                best_epoch = i
            # else:
            #     break

        draw_chart(chart_name=dirs[j].split('/')[0], measure='AUROC', axis=x, val_ordinate=val_y[:, 0],
                   train_ordinate=train_y[:, 0],
                   test_ordinate=test_y[:, 0], dst_folder='../charts_rnn')

        draw_chart(chart_name=dirs[j].split('/')[0], measure='PRECISION', axis=x, val_ordinate=None,
                   train_ordinate=None,
                   test_ordinate=test_y[:, 1], dst_folder='../charts_rnn')

        draw_chart(chart_name=dirs[j].split('/')[0], measure='RECALL', axis=x, val_ordinate=None,
                   train_ordinate=None,
                   test_ordinate=test_y[:, 2], dst_folder='../charts_rnn')

        draw_chart(chart_name=dirs[j].split('/')[0], measure='F1', axis=x, val_ordinate=None,
                   train_ordinate=None,
                   test_ordinate=test_y[:, 3], dst_folder='../charts_rnn')

        print '--> BEST EPOCH: {} VAL_AUC: {} TEST_AUC: {}'.format(best_epoch,best_val_auc,roc_auc_score(l_test,p_test_scores[:,1]))

        p_test_scores = np.load(os.path.join(d, p_test_files[best_epoch]))
        l_test = np.load(os.path.join(d, l_test_files[best_epoch]))

        best_auc_scores.append(roc_auc_score(l_test,p_test_scores[:,1]))
        best_precision_scores.append(test_y[best_epoch,1])
        best_recall_scores.append(test_y[best_epoch, 2])

    print 'AUC: {}'.format(np.mean(np.array(best_auc_scores)))

    avg_precision = np.mean(np.array(best_precision_scores))
    print 'PRECISION: {}'.format(np.mean(np.array(best_precision_scores)))

    avg_recall = np.mean(np.array(best_recall_scores))
    print 'RECALL: {}'.format(np.mean(np.array(best_recall_scores)))
    print 'F1: {}'.format(2*((avg_precision*avg_recall)/(avg_precision+avg_recall)))


if __name__ == "__main__":
    evaluate()
