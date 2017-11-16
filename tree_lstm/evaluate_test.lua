local tnt = require 'torchnet'
require('optim')

function evaluate(test_name, n_epoch, n_topic)
   local test_path = "data/arg_mining/tests/".. test_name 
   local output_path = "log/" .. test_name

   local auroc_test_values = torch.DoubleTensor(n_topic)
   

   for topic = 0, n_topic-1 do
	local logger_auroc = optim.Logger(output_path .. string.format('/AUROC_topic_%i.log',topic))
        logger_auroc:setNames{'Validation auroc', 'Test auroc'} 

        local best_val_value = -1
	local best_epoch = -1

   	for epoch = 1, n_epoch do
	   
	   local predictions_test_path = test_path .. string.format("/topic_%i/topic_%i_test_epoch_%i_predictions.pred",topic,topic,epoch)
	   local labels_test_path = test_path .. string.format("/topic_%i/topic_%i_test_epoch_%i_labels.lbl",topic,topic,epoch)
	   
	   local predictions_test = torch.load(predictions_test_path)
           local labels_test = torch.load(labels_test_path)
	   
	   local auroc_test = evaluate_auroc(labels_test,predictions_test)

	   local predictions_validation_path = test_path .. string.format("/topic_%i/topic_%i_validation_epoch_%i_predictions.pred",topic,topic,epoch)
	   local labels_validation_path = test_path .. string.format("/topic_%i/topic_%i_validation_epoch_%i_labels.lbl",topic,topic,epoch)

	   local predictions_validation = torch.load(predictions_validation_path)
	   local labels_validation = torch.load(labels_validation_path)

	   local auroc_validation = evaluate_auroc(labels_validation,predictions_validation)
	   
	   logger_auroc:add{auroc_validation, auroc_test}
	   
	   if (best_val_value < auroc_validation) then
		   best_val_value = auroc_validation
		   best_epoch = epoch
		   auroc_test_values[topic+1] = auroc_test
	   end
   	end
	print(string.format("epoch %i choosed for topic %i",best_epoch,topic))
	print(auroc_test_values[topic+1]) 
	
	--logger_auroc:display(false)
        --logger_auroc:plot()

   end
   print(auroc_test_values)
   print(torch.mean(auroc_test_values))

end

function evaluate_auroc(labels,predictions)
    
    local meter = tnt.AUCMeter()  -- initialize meter
    
    meter:add(predictions, labels)

    local evaluation_value = meter:value()

    meter:reset()
    --print('AUC:' .. evaluation_value)
    return evaluation_value
    
end

function main()
   evaluate("test_1", 10,5)

end

main()
