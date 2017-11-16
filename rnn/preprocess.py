from nltk.tree import ParentedTree,Tree
import pickle
import numpy as np
import os


def parse_tree(ptree):

    nodes_position = ptree.treepositions(order='postorder')
    leaves_position = ptree.treepositions(order='leaves')

    # print nodes_position

    n_nodes = len(nodes_position) +1


    nodes_tag = [''] * n_nodes
    is_preterminal = [False] * n_nodes
    is_leaf = [False] * n_nodes
    node_words = [''] * n_nodes

    preterminals = [-1] * n_nodes


    n_max_children = 0
    for i in range(0,n_nodes-1):
        children_counter = 0
        try:
            ptree[nodes_position[i]].label()
            if ptree[nodes_position[i]].height() == 2:
                preterminals[i] = ptree[nodes_position[i]].treeposition()

            for child in ptree[nodes_position[i]]:
                children_counter += 1

        except AttributeError:
            pass
            # if nodes_position[i] not in leaves_position:
            #     preterminals[i] = nodes_position[i]

        if children_counter > n_max_children:
            n_max_children = children_counter





    children = np.zeros([n_max_children,n_nodes])

    leaf_counter = 0

    for i in range(0,n_nodes-1):
        try:
            nodes_tag[i+1] = ptree[nodes_position[i]].label()

            if ptree[nodes_position[i]].height() == 2:
                is_preterminal[i+1] = True

                for n, child in enumerate(ptree[nodes_position[i]]):
                    child_position = ptree.leaf_treeposition(leaf_counter)


                    for index in range(0, n_nodes - 1):

                        if child_position == nodes_position[index]:
                             children[n, i + 1] = index + 1
                leaf_counter+=1
                #
                #             for c in range(0,n_max_children):
                #                 must_have_position = nodes_position[i]
                #
                #                 # print child_position
                #                 # print must_have_position
                #                 # print '---------'
                #
                #                 if child_position == must_have_position:
                #

            for n,child in enumerate(ptree[nodes_position[i]]):
                for index in range(0,n_nodes-1):
                    if child.treeposition() == nodes_position[index]:
                        children[n,i+1] = index+1



        except AttributeError:

            if nodes_position[i] in leaves_position:
                is_leaf[i+1] = True
                node_words[i+1] = ptree[nodes_position[i]]




    if not (children.shape[1] == len(is_preterminal) == len(is_leaf) == len(node_words) == len(nodes_tag)):
        print children.shape[1]
        print len(is_preterminal)
        print len(is_leaf)
        print len(node_words)
        print len(nodes_tag)
        raise Exception('size must agree!')

    # print children[1,1:]
    # print is_preterminal[1:]
    # print is_leaf[1:]
    # print node_words[1:]
    # print nodes_tag[1:]

    return children,is_preterminal,is_leaf,node_words,is_leaf,nodes_tag



def build_vocab(l):
    vocab = set()
    vocab |= set(l)
    voc = {}
    for i,w in enumerate(sorted(vocab)):
        voc[w] = i

    return voc

def build_tag_vocab(data_path):
    topics = [x[0] for x in os.walk(data_path)]
    vocab = set()
    n_max_children = 0
    for topic in topics[1:]:
        with open(os.path.join(topic, "trees.txt")) as f:
            trees = f.readlines()
        labels = []
        for t in trees:
            ptree = ParentedTree.fromstring(t)
            nodes_position = ptree.treepositions(order='postorder')
            for n in nodes_position:
                children_counter = 0
                try:
                    labels.append(ptree[n].label())
                    for child in ptree[n]:
                        children_counter += 1
                except AttributeError:
                    pass
                if children_counter > n_max_children:
                    n_max_children = children_counter



        vocab |= set(labels)
    trees_info = {}
    voc = {}
    for i,w in enumerate(sorted(vocab)):
        voc[w] = i

    trees_info['tag_vocab'] = voc
    trees_info['max_n_children'] = n_max_children


    with open('trees_info.pickle', 'wb') as handle:
        pickle.dump(trees_info, handle, protocol=pickle.HIGHEST_PROTOCOL)



def load_embeddings(embedding_path):
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

    embed = {}

    embed['word_to_id'] = word_to_id
    embed['embeddings'] = np.array(embeddings)
    embed['embedding_dim'] = embedding_dim

    with open('embeddings_dim_{}.pickle'.format(embedding_dim), 'wb') as handle:
        pickle.dump(embed,handle, protocol=pickle.HIGHEST_PROTOCOL)



def load_set(set_path,word_to_id,pos_tag_voc,max_num_children):

    # print('loading '+ set_path)

    with open(set_path + '/trees.txt') as f:
        trees = f.readlines()

    with open(set_path + '/dataset.txt') as f:
        dataset = f.readlines()

    with open(set_path + '/documents.txt') as f:
        documents = f.readlines()


    with open(set_path + '/labels.txt') as f:
        labels = f.readlines()

    with open(set_path + '/vp.txt') as f:
        vps = f.readlines()


    ukn_counter = 1
    words_counter = 1

    childrens = []
    is_preterminals = []
    ys = []
    nodes_word_id = []
    nodes_pos_tag_id = []
    docs = []
    is_leaves = []


    for tree,label,vp,doc in zip(trees,labels,vps,documents):

        if int(vp) != 0:
            children, is_preterminal, is_leaf, node_words, is_leaf, nodes_tag = parse_tree(ParentedTree.fromstring(tree))
            y = np.asarray([int(label)])


            padded_children = np.zeros([max_num_children,children.shape[1]])
            padded_children[:children.shape[0],:] = children

            node_word_indices = []
            for word in node_words:
                try:
                    node_word_indices.append(word_to_id[word])
                except KeyError:
                    node_word_indices.append(word_to_id['<ukn>'])
                    ukn_counter += 1

            node_pos_tag_indices = [pos_tag_voc[pos_tag] if pos_tag else -1 for pos_tag in nodes_tag]

            childrens.append(padded_children)
            is_preterminals.append(is_preterminal)
            ys.append(y)
            nodes_word_id.append(node_word_indices)
            nodes_pos_tag_id.append(node_word_indices)
            docs.append(doc)
            is_leaves.append(is_leaf)


    return childrens,is_preterminals,ys,nodes_word_id,nodes_pos_tag_id,docs,dataset[0].strip(),is_leaves

def load_dataset(dataset_path,embedding_file,trees_info_file):
    topics = [x[0] for x in os.walk(dataset_path)]

    with open(embedding_file, 'rb') as handle:
        embeddings = pickle.load(handle)

    with open(trees_info_file, 'rb') as handle:
        trees_info = pickle.load(handle)

    parsed_children = []
    parsed_is_preterminals = []
    parsed_ys = []
    parsed_nodes_word_id = []
    parsed_nodes_pos_tag_id = []
    parsed_docs = []
    parsed_type = []
    parsed_is_leaf = []

    for i, t in enumerate(topics[1:]):
        print ('preprocessing topic {} : {} / {}'.format(t, i, len(topics[1:])))
        childrens, is_preterminals, ys, nodes_word_id, nodes_pos_tag_id, docs, type, is_leaf = load_set(t, embeddings['word_to_id'], trees_info['tag_vocab'], trees_info['max_n_children'])
        parsed_children.append(childrens)
        parsed_is_preterminals.append(is_preterminals)
        parsed_ys.append(ys)
        parsed_nodes_word_id.append(nodes_word_id)
        parsed_nodes_pos_tag_id.append(nodes_pos_tag_id)
        parsed_docs.append(docs)
        parsed_type.append(type)
        parsed_is_leaf.append(is_leaf)





    topic_number = 0

    val_built = False

    dataset = {}
    while topic_number < len(topics[1:]):

        train_children = []
        train_is_preterminals = []
        train_ys = []
        train_nodes_word_id = []
        train_nodes_pos_tag_id = []
        train_docs = []
        train_is_leaf = []

        val_children = []
        val_is_preterminals = []
        val_ys = []
        val_nodes_word_id = []
        val_nodes_pos_tag_id = []
        val_docs = []
        val_is_leaf = []


        article_test_list = []
        article_train_list = []
        article_val_list = []

        for i,t in enumerate(topics[1:]):
                childrens, is_preterminals, ys, nodes_word_id, nodes_pos_tag_id, docs, type, is_leaf = parsed_children[i],parsed_is_preterminals[i],parsed_ys[i],parsed_nodes_word_id[i],parsed_nodes_pos_tag_id[i],parsed_docs[i],parsed_type[i],parsed_is_leaf[i]



                if topic_number == i and  type == 'train and test':

                    dataset['topic_{}_test'.format(topic_number)] = {'childrens':childrens,
                                                                     'is_preterminals':is_preterminals,
                                                                     'labels':ys,
                                                                     'nodes_word_id':nodes_word_id,
                                                                     'nodes_pos_tag_id':nodes_pos_tag_id,
                                                                     'docs':docs,
                                                                     'type':type,
                                                                     'topic_name':t,
                                                                     'is_leaf':is_leaf
                                                                     }

                    for d in docs:
                        if d not in article_test_list:
                            article_test_list.append(d)



                if topic_number != i and type == 'train and test':

                    for k in range(0,len(childrens)):

                        if docs[k] not in article_test_list and docs[k] not in article_val_list:

                            train_children.extend(childrens[k])
                            train_is_preterminals.extend(is_preterminals[k])
                            train_ys.extend(ys[k])
                            train_nodes_word_id.extend(nodes_word_id[k])
                            train_nodes_pos_tag_id.extend(nodes_pos_tag_id[k])
                            train_docs.extend(docs[k])
                            train_type = type
                            train_is_leaf.extend(is_leaf[k])

                            if docs[k] not in article_train_list:
                                article_train_list.append(docs[k])



                if type == 'held-out' and val_built == False:

                    for k in range(0, len(childrens)):

                        if docs[k] not in article_train_list:

                            val_children.extend(childrens[k])
                            val_is_preterminals.extend(is_preterminals[k])
                            val_ys.extend(ys[k])
                            val_nodes_word_id.extend(nodes_word_id[k])
                            val_nodes_pos_tag_id.extend(nodes_pos_tag_id[k])
                            val_docs.extend(docs[k])
                            val_type = type
                            val_is_leaf.extend(is_leaf[k])

                            if docs[k] not in article_val_list:
                                article_val_list.append(docs[k])

        dataset['topic_{}_train'.format(topic_number)] = {'childrens': train_children,
                                                         'is_preterminals': train_is_preterminals,
                                                         'labels': train_ys,
                                                         'nodes_word_id': train_nodes_word_id,
                                                         'nodes_pos_tag_id': train_nodes_pos_tag_id,
                                                         'docs': train_docs,
                                                         'type': train_type,
                                                          'is_leaf': train_is_leaf
                                                         }

        if val_built == False:

            dataset['val_set'] = {'childrens': val_children,
                                                              'is_preterminals': val_is_preterminals,
                                                              'labels': val_ys,
                                                              'nodes_word_id': val_nodes_word_id,
                                                              'nodes_pos_tag_id': val_nodes_pos_tag_id,
                                                              'docs': val_docs,
                                                              'type': val_type,
                                                              'is_leaf': val_is_leaf
                                                              }
            val_built = True

        topic_number += 1


    with open('dataset.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":


    # ptree = ParentedTree.fromstring('(ROOT (S (NP (NP (NN Substance) (NN abuse)) (, ,) (VP (ADVP (RB also)) (VBN known) (PP (IN as) (NP (NN drug) (NN abuse)))) (, ,)) (VP (VBZ refers) (PP (TO to) (NP (NP (DT a) (JJ maladaptive) (NN pattern)) (PP (IN of) (NP (NP (NN use)) (PP (IN of) (NP (NP (DT a) (NN substance)) (PRN (-LRB- -LRB-) (NN drug) (-RRB- -RRB-)) (SBAR (WHNP (WDT that)) (S (VP (VBZ is) (RB not) (VP (VBN considered) (S (ADJP (JJ dependent)))))))))))))) (. .)))')
    #
    # ptree.pretty_print()
    # parse_tree(ptree)

    # build_tag_vocab('ibm_v3')
    load_embeddings('glove/glove.6B.50d.txt')


    # with open('embeddings_dim_50.pickle', 'rb') as handle:
    #     embeddings = pickle.load(handle)
    #
    # with open('trees_info.pickle', 'rb') as handle:
    #     trees_info = pickle.load(handle)
    #
    # load_set('ibm_v3/This_house_believes_all_nations_have_a_right_to_nuclear_weapons',embeddings['word_to_id'],trees_info['tag_vocab'],trees_info['max_n_children'])



    load_dataset('ibm_v3','embeddings_dim_50.pickle','trees_info.pickle')







