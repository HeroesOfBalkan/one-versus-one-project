using Match

function predict_classes(data_predictions, class1 = 0, class2 = 1)
    data_predictions_class = zeros(length(data_predictions))

    for index in eachindex(data_predictions)
        data_predictions_class[index] = @match data_predictions[index] < 0.5 begin
            true => class1
            false => class2
        end
    end
    
    return data_predictions_class
end



function test_validity(data_predictions, data)
    roc_test = ROC.roc(data_predictions, data.voce, true)
    auc_test = AUC(roc_test)
    println("Povrsina ispod krive u procentima je: $auc_test")

    if (auc_test > 0.9)
        println("Klasifikator je jako dobar")
    elseif (auc_test > 0.8)
        println("Klasifikator je veoma dobar")
    elseif (auc_test > 0.7)
        println("Klasifikator je dosta dobar")
    elseif (auc_test > 0.5)
        println("Klasifikator je relativno dobar")
    else
        println("Klasifikator je los")
    end

    return (roc_test, auc_test)
end