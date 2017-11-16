import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import re
import matplotlib.pyplot as plt
import seaborn as sns

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text)]

def draw_chart(chart_name,measure,axis,val_ordinate,train_ordinate,test_ordinate,dst_folder):

    plt.style.use('seaborn')
    sns.set(font_scale=1.4)
    sns.set_style({'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(8, 8))
    ttl = ax.title
    ttl.set_position([.5, 1.05])
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    ax.set_title(' '.join(chart_name.split('_')[0:]))
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 1)
    plt.xticks(np.arange(0, 21, 2))
    ax.set_xlim(0, 20)
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
    plt.savefig(os.path.join(dst_folder,'{}_{}.pdf'.format(''.join(chart_name.replace(' ','_').split(':')).strip(),measure)),dpi=300,bbox_inches='tight')
    # plt.show()


def evaluate(epochs,path):

    # with open('log/test_tesi_3/associations.txt') as f:
    #
    #
    # content.sort(key=natural_keys)

    # print content
    with open('considered_topic.txt') as f:
        considered_topic = f.readlines()
    considered_topic = [x.strip() for x in considered_topic]

    best_auc_scores = []
    best_precision_scores = []
    best_recall_scores = []
    best_f1_scores = []


    for j in range(0,38):

        with open('data/arg_mining_new/topic_{}_test/topic_name.txt'.format(j)) as f:
            topic_name = f.readlines()[0]

        if topic_name not in considered_topic:
            continue

        x = np.arange(epochs+1)
        val_y = np.zeros((epochs+1, 3))
        test_y = np.zeros((epochs+1, 4))
        train_y = np.zeros((epochs+1, 3))

        for i in range(0,epochs):
            best_val_auc = 0.
            best_epoch = 0



            print '-------------topic: {} epoch: {} --------------'.format(j,i)


            # print p_val_files
            # print l_val_files

            p_train_scores = np.load(os.path.join(path, 'predictions_training_topic{}_epoch_{}.npy'.format(j, i)))
            l_train = np.load(os.path.join(path, 'training_labels_topic_{}.npy'.format(j)))

            # print p_train_scores.shape
            # print l_train.shape

            train_top_predictions_scores = np.argsort(p_train_scores)[::-1]
            train_top_labels = l_train[train_top_predictions_scores]


            train_precision_at_200 = np.sum(train_top_labels[:200]) / 200.
            train_recall_at_200 = np.sum(train_top_labels[:200]) / float(np.sum(l_train))



            p_val_scores = np.load(os.path.join(path,'predictions_validation_topic{}_epoch_{}.npy'.format(j,i)))
            l_val = np.load(os.path.join(path,'validation_labels_topic_{}.npy'.format(j)))

            val_top_predictions_scores = np.argsort(p_val_scores)[::-1]
            val_top_labels = l_val[val_top_predictions_scores]

            val_precision_at_200 = np.sum(val_top_labels[:200]) / 200.
            val_recall_at_200 = np.sum(val_top_labels[:200]) / float(np.sum(l_val))

            p_test_scores = np.load(os.path.join(path, 'predictions_test_topic{}_epoch_{}.npy'.format(j,i)))
            l_test = np.load(os.path.join(path, 'test_labels_topic_{}.npy'.format(j)))

            test_top_predictions_scores = np.argsort(p_test_scores)[::-1]
            test_top_labels = l_test[test_top_predictions_scores]

            test_precision_at_200 = np.sum(test_top_labels[:200]) / 200.
            test_recall_at_200 = np.sum(test_top_labels[:200]) / float(np.sum(l_test))

            f1_test = 2* ((test_precision_at_200*test_recall_at_200)/(test_precision_at_200+test_recall_at_200))

            val_scores = (roc_auc_score(l_val,p_val_scores),val_precision_at_200,val_recall_at_200)
            test_scores = (roc_auc_score(l_test,p_test_scores),test_precision_at_200,test_recall_at_200,f1_test)
            train_scores = (roc_auc_score(l_train,p_train_scores),train_precision_at_200,train_recall_at_200)

            print test_scores

            val_y[i+1,:] = val_scores
            train_y[i+1,:] = train_scores
            test_y[i + 1, :] = test_scores

            current_auc_val = roc_auc_score(l_val,p_val_scores)

            if current_auc_val > best_val_auc :
                best_val_auc = current_auc_val
                best_epoch = i
            # else:
            #     break

        print train_y


        draw_chart(chart_name=topic_name, measure='AUROC',axis=x,val_ordinate=val_y[:,0],train_ordinate=train_y[:,0],
                   test_ordinate=test_y[:,0],dst_folder='log/test_tesi_bs_10_lr_1e-05_nl_3_nc_300_sl_35/charts')

        draw_chart(chart_name=topic_name, measure='PRECISION',axis=x,val_ordinate=None,train_ordinate=None,
                   test_ordinate=test_y[:,1],dst_folder='log/test_tesi_bs_10_lr_1e-05_nl_3_nc_300_sl_35/charts')

        draw_chart(chart_name=topic_name, measure='RECALL',axis=x,train_ordinate=None,val_ordinate=None,
                   test_ordinate=test_y[:,2],dst_folder='log/test_tesi_bs_10_lr_1e-05_nl_3_nc_300_sl_35/charts')

        draw_chart(chart_name=topic_name, measure='F1',axis=x,train_ordinate=None,val_ordinate=None,
                   test_ordinate=test_y[:,3],dst_folder='log/test_tesi_bs_10_lr_1e-05_nl_3_nc_300_sl_35/charts')


        p_test_scores = np.load(os.path.join(path, 'predictions_test_topic{}_epoch_{}.npy'.format(j,best_epoch)))
        l_test = np.load(os.path.join(path, 'test_labels_topic_{}.npy'.format(j)))

        test_top_predictions_scores = np.argsort(p_test_scores)[::-1]
        test_top_labels = l_test[test_top_predictions_scores]

        test_precision_at_200 = np.sum(test_top_labels[:200]) / 200.
        test_recall_at_200 = np.sum(test_top_labels[:200]) / float(np.sum(l_test))
        f1_test = 2* ((test_precision_at_200*test_recall_at_200)/(test_precision_at_200+test_recall_at_200))

        best_auc_scores.append(roc_auc_score(l_test,p_test_scores))
        best_precision_scores.append(test_precision_at_200)
        best_recall_scores.append(test_recall_at_200)
        best_f1_scores.append(f1_test)


    print 'AUC: {}'.format(np.mean(np.array(best_auc_scores)))
    print 'PRECISION: {}'.format(np.mean(np.array(best_precision_scores)))
    print 'RECALL: {}'.format(np.mean(np.array(best_recall_scores)))
    print 'F1: {}'.format(np.mean(np.array(best_f1_scores)))

if __name__ == "__main__":
    evaluate(epochs = 20,path='log/test_tesi_3/predictions')
