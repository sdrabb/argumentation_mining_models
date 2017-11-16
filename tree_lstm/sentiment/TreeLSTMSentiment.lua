--[[

  Sentiment classification using a Binary Tree-LSTM.

--]]

-----actual----
-- mem_dim           =  80,
-- learning_rate     =  0.0001,
-- emb_learning_rate =  0.002,
-- batch_size        =  50,
-- reg               = 1e-4,
-- structure         = 'constituency',
-- fine_grained      = false,
-- dropout           = false,

-----------------default-parameter
-- self.mem_dim           = config.mem_dim           or 80
-- self.learning_rate     = config.learning_rate     or 0.0001
-- self.emb_learning_rate = config.emb_learning_rate or 0.002
-- self.batch_size        = config.batch_size        or 50
-- self.reg               = config.reg               or 1e-4
-- self.structure         = config.structure         or 'constituency'
-- self.fine_grained      = (config.fine_grained == nil) and true or config.fine_grained
-- self.dropout           = (config.dropout == nil) and true or config.dropout

local tnt = require 'torchnet'


local TreeLSTMSentiment = torch.class('treelstm.TreeLSTMSentiment')

function TreeLSTMSentiment:__init(config)
  self.mem_dim           =  250
  self.learning_rate     =  0.05
  self.emb_learning_rate =  0.1
  --self.batch_size        =  25
  self.batch_size        =  10
  self.reg               =  1e-4
  self.structure         = 'constituency'
  self.fine_grained      = (config.fine_grained == nil) and true or config.fine_grained
  self.dropout           = false
  self.threads            = 8



  torch.setnumthreads(self.threads)
  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)

  self.emb.weight:copy(config.emb_vecs)

  self.in_zeros = torch.zeros(self.emb_dim)
  self.num_classes = self.fine_grained and 5 or 2

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- negative log likelihood optimization objective
  local weights = torch.Tensor(2)
  weights[1] = 1
  weights[2] = 32


  --self.criterion = nn.ClassNLLCriterion(weights)
  self.criterion = nn.ClassNLLCriterion()

  local treelstm_config = {
    in_dim  = self.emb_dim,
    mem_dim = self.mem_dim,
    output_module_fn = function() return self:new_sentiment_module() end,
    criterion = self.criterion,
  }

  if self.structure == 'dependency' then
    self.treelstm = treelstm.ChildSumTreeLSTM(treelstm_config)
  elseif self.structure == 'constituency' then
    self.treelstm = treelstm.BinaryTreeLSTM(treelstm_config)
  else
    error('invalid parse tree type: ' .. self.structure)
  end

  self.params, self.grad_params = self.treelstm:getParameters()
end

function TreeLSTMSentiment:new_sentiment_module()
  local sentiment_module = nn.Sequential()
  if self.dropout then
    sentiment_module:add(nn.Dropout())
  end
  sentiment_module
    :add(nn.Linear(self.mem_dim, self.num_classes))
    --posso mettere una sigmoide per buttare valori tra 0 e 1
    -- :add(nn.LogSoftMax())
    :add(nn.LogSoftMax())


  return sentiment_module
end

function TreeLSTMSentiment:cross_validation(dataset)

  local unconsistence_train_idx = {}
  self.treelstm:training()
  local indices = torch.randperm(dataset.size)
  --local indices = torch.range(1,dataset.size)
  local zeros = torch.zeros(self.mem_dim)


  -- this matrix records the current confusion across classes
  classes = {'1','2'}
  confusion = optim.ConfusionMatrix(classes)

  -- -- log results to files
  --trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
  -- testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

  local global_loss = 0

  local meter = tnt.AUCMeter()  -- initialize meter
  meter:reset()  -- reset meter


 local outputs_tensor = torch.Tensor(dataset.size)
 local labels_tensor = torch.Tensor(dataset.size)
 local count = 1

  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local sent = dataset.sents[idx]
        local tree = dataset.trees[idx]

        --le foglie di alcuni alberi parsati con nltk parser non coincidono con i tokens di queste per questo al momento vengono saltati

	   local inputs = self.emb:forward(sent)



           local _, tree_loss = self.treelstm:forward(tree, inputs)
           loss = loss + tree_loss

	   local outputs = torch.exp(tree.output)

           local input_grad = self.treelstm:backward(tree, inputs, {zeros, zeros})
           self.emb:backward(sent, input_grad)

	   outputs_tensor[count] = outputs[2]
	   labels_tensor[count] = tree.gold_label-1
           count = count + 1


	   confusion:add(outputs, tree.gold_label)


      end

      loss = loss

      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    local _, minibatch_loss = optim.adagrad(feval, self.params, self.optim_state)


    global_loss = global_loss + minibatch_loss[1]

    self.emb:updateParameters(self.emb_learning_rate)
  end

  meter:add(outputs_tensor,labels_tensor )

  xlua.progress(dataset.size, dataset.size)
  print (string.format('global loss: %f',(global_loss/(dataset.size)) ))
  print('AUC:' .. meter:value())
        -- print confusion matrix
         print(confusion)
         --trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
         confusion:zero()
  meter:reset()
end

function TreeLSTMSentiment:train(dataset)
  local unconsistence_train_idx = {}
  self.treelstm:training()
  local indices = torch.randperm(dataset.size)
  --local indices = torch.range(1,dataset.size)
  local zeros = torch.zeros(self.mem_dim)


  -- this matrix records the current confusion across classes
  local classes = {'1','2'}
  local confusion = optim.ConfusionMatrix(classes)

  -- -- log results to files
  --trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
  -- testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

  local global_loss = 0

  local meter = tnt.AUCMeter()  -- initialize meter
  meter:reset()  -- reset meter


 local outputs_tensor = torch.Tensor(dataset.size)
 local labels_tensor = torch.Tensor(dataset.size)
 local count = 1

  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local sent = dataset.sents[idx]
        local tree = dataset.trees[idx]

        --le foglie di alcuni alberi parsati con nltk parser non coincidono con i tokens di queste per questo al momento vengono saltati

	   local inputs = self.emb:forward(sent)



           local _, tree_loss = self.treelstm:forward(tree, inputs)
           loss = loss + tree_loss

	   --local outputs = torch.exp(tree.output)

           local input_grad = self.treelstm:backward(tree, inputs, {zeros, zeros})
           self.emb:backward(sent, input_grad)

	   --outputs_tensor[count] = outputs[2]
	   --labels_tensor[count] = tree.gold_label-1
           --count = count + 1


	   --confusion:add(outputs, tree.gold_label)


      end

      loss = loss

      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    local _, minibatch_loss = optim.adagrad(feval, self.params, self.optim_state)


    global_loss = global_loss + minibatch_loss[1]

    self.emb:updateParameters(self.emb_learning_rate)
  end

  --meter:add(outputs_tensor,labels_tensor )

  xlua.progress(dataset.size, dataset.size)
  print (string.format('global loss: %f',(global_loss/(dataset.size)) ))
  --print('AUC:' .. meter:value())
        -- print confusion matrix
    --     print(confusion)
      --      confusion:zero()
  --meter:reset()
end


function built_balance_indexes(positive_subset,negative_subset,batch_size)
   local indices = {}

  -- print(#positive_subset)
  -- print(#negative_subset)

   for i = 1, batch_size do
      if i <= 15   then
	 table.insert(indices, positive_subset[ math.random( #positive_subset ) ])
      else
	 table.insert(indices , negative_subset[ math.random( #negative_subset ) ])
      end
   end
   return indices
end


function TreeLSTMSentiment:train_with_balanced_minibatch(positive_subset,negative_subset,dataset)

      local unconsistence_train_idx = {}
      self.treelstm:training()
      --local indices = torch.range(1,dataset.size)
      local zeros = torch.zeros(self.mem_dim)

      -- this matrix records the current confusion across classes
      classes = {'1','2'}
	 confusion = optim.ConfusionMatrix(classes)
      local global_loss = 0
      local n_of_sampling = 1000
      local batch_size = self.batch_size

      local meter = tnt.AUCMeter()  -- initialize meter
      meter:reset()  -- reset meter

      local outputs_tensor = torch.Tensor(n_of_sampling)
      local labels_tensor = torch.Tensor(n_of_sampling)
      local count = 1


      for i = 1, n_of_sampling, batch_size do
      local indices = built_balance_indexes(positive_subset,negative_subset,batch_size)


      xlua.progress(i, n_of_sampling)
      local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()

      local loss = 0



      for j = 1, batch_size do
        local idx = indices[ j ]
        local sent = dataset.sents[idx]
        local tree = dataset.trees[idx]
        --le foglie di alcuni alberi parsati con nltk parser non coincidono con i tokens di queste per questo al momento vengono saltati

	   local inputs = self.emb:forward(sent)


           local _, tree_loss = self.treelstm:forward(tree, inputs)
           local outputs = torch.exp(tree.output)


	   loss = loss + tree_loss
           local input_grad = self.treelstm:backward(tree, inputs, {zeros, zeros})
           self.emb:backward(sent, input_grad)




	   outputs_tensor[count] = outputs[2]
	   labels_tensor[count] = tree.gold_label-1
	   count = count + 1

	   -- meter:add(outputs[2], (tree.gold_label-1))
	   confusion:add(outputs, tree.gold_label)

      end


      loss = loss / batch_size

      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2


      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end



    local _, minibatch_loss = optim.adagrad(feval, self.params, self.optim_state)
    global_loss = global_loss + minibatch_loss[1]

    self.emb:updateParameters(self.emb_learning_rate)
  end
   meter:add(outputs_tensor,labels_tensor )
  xlua.progress( n_of_sampling,  n_of_sampling)
  print (string.format('global loss: %f',(global_loss/(dataset.size/self.batch_size)) ))
  print('AUC:' .. meter:value())
  meter:reset()
        -- print confusion matrix
         print(confusion)
         --trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
         confusion:zero()

end


function TreeLSTMSentiment:predict_with_scores(tree, sent)
  self.treelstm:evaluate()
  local prediction
  -- print (sent)
  local inputs = self.emb:forward(sent)

  self.treelstm:forward(tree, inputs)
  local output = tree.output

  if self.fine_grained then
    prediction = output
  else
    prediction = torch.exp(output[2])
  end
  self.treelstm:clean(tree)
  return prediction
end


function TreeLSTMSentiment:predict(tree, sent)
  self.treelstm:evaluate()
  local prediction
  local inputs = self.emb:forward(sent)
  self.treelstm:forward(tree, inputs)
  local output = tree.output
  if self.fine_grained then
    prediction = argmax(output)
  else
    prediction = (output[1] > output[3]) and 1 or 3
  end
  self.treelstm:clean(tree)
  return prediction
end

function TreeLSTMSentiment:predict_dataset(dataset)


  local predictions = torch.Tensor(dataset.size)
  local labels_tensor = torch.Tensor(dataset.size)
  local meter = tnt.AUCMeter()  -- initialize meter
  meter:reset()  -- reset meter
  -- this matrix records the current confusion across classes
  local classes = {1,2}
  local confusion = optim.ConfusionMatrix(#classes,classes)


  for i = 1, dataset.size do

       xlua.progress(i, dataset.size)
       predictions[i] = self:predict_with_scores(dataset.trees[i], dataset.sents[i])
       labels_tensor[i] = dataset.trees[i].gold_label-1

       --print (dataset.trees[i].gold_label)
       --print (predictions[i])
       --confusion:add(predictions[i] , dataset.trees[i].gold_label)

  end

    meter:add(predictions,labels_tensor )
    print('AUC:' .. meter:value())
    --print(confusion)
    --confusion:zero()


  return predictions,labels_tensor
end


function TreeLSTMSentiment:save_predictions(dataset,topic,predictions_dst_dir)
  local num_skipped_sents = 0
  local predictions = torch.Tensor(dataset.size)
  local stringed_prediction = ''
  local fd = io.open(predictions_dst_dir , 'w')

  for i = 1, dataset.size do

      --  xlua.progress(i, dataset.size)
       predictions[i] = self:predict_with_scores(dataset.trees[i], dataset.sents[i])
       stringed_prediction = stringed_prediction .. string.format("%i %i %f \n", dataset.trees[i].dataset_idx, remap_label(dataset.trees[i].gold_label), predictions[i])

  end
  fd:write(stringed_prediction)
  fd:close()

  return predictions
end

function argmax(v)
  local idx = 1
  local max = v[1]
  for i = 2, v:size(1) do
    if v[i] > max then
      max = v[i]
      idx = i
    end
  end
  return idx
end

function TreeLSTMSentiment:print_config()
  local num_params = self.params:size(1)
  local num_sentiment_params = self:new_sentiment_module():getParameters():size(1)
  printf('%-25s = %s\n',   'fine grained sentiment', tostring(self.fine_grained))
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sentiment_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'Tree-LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %s\n',   'dropout', tostring(self.dropout))
end

function TreeLSTMSentiment:save(path)
  local config = {
    batch_size        = self.batch_size,
    dropout           = self.dropout,
    emb_learning_rate = self.emb_learning_rate,
    emb_vecs          = self.emb.weight:float(),
    fine_grained      = self.fine_grained,
    learning_rate     = self.learning_rate,
    mem_dim           = self.mem_dim,
    reg               = self.reg,
    structure         = self.structure,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function TreeLSTMSentiment.load(path)
  local state = torch.load(path)
  local model = treelstm.TreeLSTMSentiment.new(state.config)
  model.params:copy(state.params)
  return model
end

function save_unconsistence_idx(unconsistences,dst_file_path)
  local fd = io.open(dst_file_path, 'w')
  for i=1,#unconsistences do
    fd:write(string.format("%i \n", unconsistences[i]))
  end
  fd:close()
end

function remap_label(label)
  local remapped_label = nil

  if label == 1 then
    remapped_label = 0
  else
    remapped_label = 1
  end

  return remapped_label
end
