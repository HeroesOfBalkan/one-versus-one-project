using Match

function predict_classes(data_predictions, class1 = 0, class2 = 1)
    data_predictions_class = zeros(length(data_test_prediction0))

    for i in eachindex(data_predictions)
        if (data_predictions[i] < 0.5)
            data_predictions_class[i] = 0 # class1
        else
            data_predictions_class[i] = 1 # class2
        end
    end
    
    return data_predictions_class
end



function test_validity(data_predictions, data, class_no = -1)
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

    # @match auc_test begin
    #     0.9:1.0 => println("Klasifikator je jako dobar")
    #     0.8:0.9 => println("Klasifikator je veoma dobar")
    #     0.7:0.8 => println("Klasifikator je dosta dobar")
    #     0.5:0.7 => println("Klasifikator je relativno dobar")
    #     0.0:0.5 => println("Klasifikator je los")
    #     _ => println("Greska")
    # end

    return (roc_test, auc_test)

    # if class_no == -1
    #     plot(roc_test, label="ROC kriva klasifikatora")
    # else
    #     plot(roc_test, label="ROC kriva klasifikatora #$(class_no)")
    # end
end