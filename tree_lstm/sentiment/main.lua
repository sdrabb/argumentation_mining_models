--[[

  Tree-LSTM training script for sentiment classication on the Stanford
  Sentiment Treebank

--]]
-- require 'cudnn'
require('..')
metrics = require 'metrics'
io = require 'io'


function get_neg_pos_from_set(set)
   local pos_examples = {}
   local neg_examples = {}
   --print (set.labels[1])
   for i = 1, set.size do

	if set.labels[i] == 2 then

		table.insert(pos_examples, i)
	end
	if set.labels[i] == 1 then
		table.insert(neg_examples,i)
	end
   end
   return pos_examples,neg_examples
end

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

function remap_labels_for_auroc(labels)
  local labels_size = labels:size(1)

  local remapped_labels = labels:clone()
  remapped_labels:apply(function(label)
      if label == 1 then
        return -1
      else
        return 1
      end
    end)

  return remapped_labels
end

function remap_pred_for_auroc(preds)
  local remapped_preds = preds:clone()

  remapped_preds:apply(function(pred)
      if pred > 1-pred  then
        return 1
      else
        return -1
      end
    end)
  return remapped_preds
end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end


function auroc(preds,labels)

  roc_points, thresholds = metrics.roc.points(remap_labels_for_auroc(preds), remap_labels_for_auroc(labels))

  area = metrics.roc.area(roc_points)
  return area
end




function save_model_info(model,file_path)
  local fd = io.open(file_path, 'w')
  fd:write(string.format("fine grained argument: %s \n", tostring(model.fine_grained)))
  fd:write(string.format("num params: %s \n", model.params:size(1)))
  fd:write(string.format("word vector dim: %i \n", model.emb_dim))
  fd:write(string.format("Tree-LSTM memory dim: %i \n", model.mem_dim))
  fd:write(string.format("regularization strength:  %f \n", model.reg))
  fd:write(string.format("minibatch size:  %i \n", model.batch_size))
  fd:write(string.format("learning rate: %f \n", model.learning_rate))
  fd:write(string.format("word vector learning rate: %f \n", model.emb_learning_rate))
  fd:write(string.format("dropout: %s \n", tostring(model.dropout)))

  fd:close()
end



-- read command line arguments
local args = lapp [[
Training script for sentiment classification on the SST dataset.
  -m,--model  (default constituency) Model architecture: [constituency, lstm, bilstm]
  -l,--layers (default 1)            Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)          LSTM memory dimension
  -e,--epochs (default 10)           Number of training epochs
  -b,--binary                        Train and evaluate on binary sub-task
]]

local model_name, model_class, model_structure
if args.model == 'constituency' then
  model_name = 'Constituency Tree LSTM'


  model_class = treelstm.TreeLSTMSentiment
elseif args.model == 'dependency' then
  model_name = 'Dependency Tree LSTM'
  model_class = treelstm.TreeLSTMSentiment
elseif args.model == 'lstm' then
  model_name = 'LSTM'
  model_class = treelstm.LSTMSentiment
elseif args.model == 'bilstm' then
  model_name = 'Bidirectional LSTM'
  model_class = treelstm.LSTMSentiment
end
model_structure = args.model
header(model_name .. ' for Sentiment Classification')

-- binary or fine-grained subtask
local fine_grained = not args.binary

-- directory containing dataset files
local data_dir = 'data/arg_mining_new/'

-- load vocab
local vocab = treelstm.Vocab(data_dir .. 'vocab-cased.txt')


local tests_dir = data_dir .. 'tests/'
lfs.mkdir(tests_dir)
local actual_test_dir = tests_dir .. 'treelstm_tesi_we_50_bs_10/'
lfs.mkdir(actual_test_dir)


unconsistence_test_idx = {}

--for each topic

for topic = 16, 38 do
--for topic = 9
-- local topic = 32


  local best_epoch = 1
  local topic_test_directory = actual_test_dir ..  string.format("topic_%i/", topic)
  lfs.mkdir(topic_test_directory)

  -- load embeddings
  print('loading word embeddings')
  local emb_dir = 'data/glove/'
  local emb_prefix = emb_dir .. 'glove.6B'
  local emb_vocab, emb_vecs = treelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.50d.th')
  local emb_dim = emb_vecs:size(2)


  -- use only vectors in vocabulary (not necessary, but gives faster training)
  local num_unk = 0
  local vecs = torch.Tensor(vocab.size, emb_dim)

  for i = 1, vocab.size do
    local w = string.gsub(vocab:token(i), '\\', '') -- remove escape characters
    if emb_vocab:contains(w) then
      vecs[i] = emb_vecs[emb_vocab:index(w)]
    else
      num_unk = num_unk + 1
      vecs[i]:uniform(-0.05, 0.05)
    end
  end
  print('unk count = ' .. num_unk)
  emb_vocab = nil
  emb_vecs = nil
  collectgarbage()

  -- load datasets
  print('loading datasets')
  local train_dir = data_dir .. string.format("topic_%i_train/", topic)
  local dev_dir = data_dir .. 'validation/'
  local test_dir = data_dir .. string.format("topic_%i_test/", topic)
  local dependency = (args.model == 'dependency')
  local train_dataset = treelstm.read_sentiment_dataset(train_dir, vocab, fine_grained, dependency,topic_test_directory .. 'train_unconsistences.txt')
  local dev_dataset = treelstm.read_sentiment_dataset(dev_dir, vocab, fine_grained, dependency,topic_test_directory .. 'val_unconsistences.txt')
  local test_dataset = treelstm.read_sentiment_dataset(test_dir, vocab, fine_grained, dependency,topic_test_directory .. 'test_unconsistences.txt')

  local trainset_pos, trainset_neg = get_neg_pos_from_set(train_dataset)
  print(#trainset_pos , #trainset_neg)

  printf('num train = %d\n', train_dataset.size)
  printf('num dev   = %d\n', dev_dataset.size)
  printf('num test  = %d\n', test_dataset.size)

  -- initialize model
  local model = model_class{
    emb_vecs = vecs,
    structure = model_structure,
    fine_grained = fine_grained,
    num_layers = args.layers,
    mem_dim = args.dim,
  }



  save_model_info(model, actual_test_dir .. 'model_info.txt')

  -- number of epochs to train
  local num_epochs = args.epochs

  -- print information
  header('model configuration')
  printf('max epochs = %d\n', num_epochs)

  model:print_config()

  print(string.format('training topic: %i',topic))

  -- train
  local train_start = sys.clock()
  local best_dev_score = -1.0
  local best_dev_model = model
  header('Training model')
  for i = 1, num_epochs do
    local start = sys.clock()
    printf('-- epoch %d\n', i)

    model_path = actual_test_dir .. string.format('model_topic_%i_epoch_%i.th',topic,i)

    if file_exists(model_path) then
      model = model_class.load(model_path)
    else
     -- model:train_with_balanced_minibatch(trainset_pos, trainset_neg,train_dataset)
      model:train(train_dataset)
      print('writing model to ' .. model_path)
      model:save(model_path)
    end


    printf('-- finished epoch in %.2fs\n', sys.clock() - start)

    -- uncomment to compute train scores
    --[[
    local train_predictions = model:predict_dataset(train_dataset)
    local train_score = accuracy(train_predictions, train_dataset.labels)
    printf('-- train score: %.4f\n', train_score)
    --]]


    train_pred_path = topic_test_directory .. string.format('topic_%i_train_epoch_%i_predictions.pred',topic,i)
    train_label_path = topic_test_directory .. string.format('topic_%i_train_epoch_%i_labels.lbl',topic,i)

    if file_exists(model_path) and not file_exists(train_pred_path) then
    printf('predict test_set\n')
    local train_predictions, train_labels = model:predict_dataset(train_dataset)

      torch.save( train_pred_path,train_predictions  )
      torch.save( train_label_path, train_labels)
    end



    test_pred_path = topic_test_directory .. string.format('topic_%i_test_epoch_%i_predictions.pred',topic,i)
    test_label_path = topic_test_directory .. string.format('topic_%i_test_epoch_%i_labels.lbl',topic,i)

    if file_exists(model_path) and not file_exists(test_pred_path) then
    printf('predict test_set\n')
    local test_predictions, test_labels = model:predict_dataset(test_dataset)


     -- print('save predictions of test')
      --model:save_predictions(test_dataset,topic, test_pred_path)
      torch.save( test_pred_path,test_predictions  )
      torch.save( test_label_path, test_labels)
    end

    validation_pred_path = topic_test_directory .. string.format('topic_%i_validation_epoch_%i_predictions.pred',topic,i)
    validation_label_path = topic_test_directory .. string.format('topic_%i_validation_epoch_%i_labels.lbl',topic,i)


    if file_exists(model_path) and not file_exists(validation_pred_path) then
    printf('predict validation_set\n')
    local dev_predictions, dev_labels = model:predict_dataset(dev_dataset)


      torch.save( validation_pred_path,dev_predictions  )
      torch.save( validation_label_path , dev_labels)
      --print('save predictions of validation')
      --local dev_predictions = model:save_predictions(dev_dataset,topic, validation_pred_path)

      -- print('computing validation score')
      -- local dev_score = auroc(dev_predictions, dev_dataset.labels)
      -- dev_predictions = {}
      -- printf('-- dev score: %.4f\n', dev_score)
      --
      -- if dev_score > best_dev_score then
      --   best_dev_score = dev_score
      --   best_epoch = i
      --   best_dev_model = model_class{
      --     emb_vecs = vecs,
      --     structure = model_structure,
      --     fine_grained = fine_grained,
      --     num_layers = args.layers,
      --     mem_dim = args.dim,
      --   }
      --   best_dev_model.params:copy(model.params)
      --   best_dev_model.emb.weight:copy(model.emb.weight)
      -- end
    end
    --
    -- print(dev_predictions)
    -- print( dev_dataset.labels)
    -- local dev_predictions = model:predict_dataset(dev_dataset)

  end
  printf('finished training in %.2fs\n', sys.clock() - train_start)

  -- -- evaluate
  -- header('Evaluating on test set')
  -- printf('-- using model with dev score = %.4f\n', best_dev_score)
  -- local test_predictions = best_dev_model:predict_dataset(test_dataset)
  --
  -- best_model_pred_path = topic_test_directory .. string.format('topic_%i_best_test_epoch_%i_predictions.txt',topic,best_epoch)
  --
  -- best_dev_model:save_predictions(test_dataset,topic,best_model_pred_path)
  --
  -- printf('-- test score: %.4f\n', auroc(test_predictions, test_dataset.labels))

  -- create predictions and models directories if necessary

  -- if lfs.attributes(treelstm.predictions_dir) == nil then
  --   lfs.mkdir(treelstm.predictions_dir)
  -- end
  --
  -- if lfs.attributes(treelstm.models_dir) == nil then
  --   lfs.mkdir(treelstm.models_dir)
  -- end

  -- treelstm.predictions_dir = topic_test_directory .. 'predictions_default/'
  -- treelstm.models_dir = topic_test_directory .. 'models/'
  --
  -- lfs.mkdir(treelstm.predictions_dir)
  -- lfs.mkdir(treelstm.models_dir)
  --
  -- -- get paths
  -- local file_idx = 0
  -- local subtask = fine_grained and '5class' or '2class'
  -- local predictions_save_path, model_save_path
  -- while true do
  --   predictions_save_path = string.format(
  --     treelstm.predictions_dir .. '/sent-%s.%s.%dl.%dd.%d.pred', args.model, subtask, args.layers, args.dim, file_idx)
  --   model_save_path = string.format(
  --     treelstm.models_dir .. '/sent-%s.%s.%dl.%dd.%d.th', args.model, subtask, args.layers, args.dim, file_idx)
  --   if lfs.attributes(predictions_save_path) == nil and lfs.attributes(model_save_path) == nil then
  --     break
  --   end
  --   file_idx = file_idx + 1
  -- end
  --
  -- -- write predictions to disk
  -- local predictions_file = torch.DiskFile(predictions_save_path, 'w')
  -- print('writing predictions to ' .. predictions_save_path)
  -- for i = 1, test_predictions:size(1) do
  --   predictions_file:writeInt(test_predictions[i])
  -- end
  -- predictions_file:close()
  --
  -- -- write model to disk
  -- print('writing model to ' .. model_save_path)
  -- best_dev_model:save(model_save_path)

  -- to load a saved model
  -- local loaded = model_class.load(model_save_path)
end
