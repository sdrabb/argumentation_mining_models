local tnt = require 'torchnet'
require('optim')

function evaluate(test_path, n_epoch)
   local logger_auroc = optim.Logger('AUROC.log')
   logger_auroc:setNames{'Validation auroc', 'Test auroc'} 


   for topic = 0,15 do
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


   	end
   end
   logger_auroc:display(false)
   logger_auroc:plot()
end

function evaluate_auroc(labels,predictions)
    
    local meter = tnt.AUCMeter()  -- initialize meter
    meter:reset()  -- reset meter
    meter:add(predictions, labels_tensor)

    local evaluation_value = meter:value()
    print('AUC:' .. evaluation_value)
    return evaluation_value
    
end

function main()
   evaluate("/data/arg_mining/tests/test_1", 10)

end

main()
