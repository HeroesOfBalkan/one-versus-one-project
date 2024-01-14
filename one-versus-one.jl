using StatsModels
using GLM
using DataFrames
using CSV
using Lathe.preprocess: TrainTestSplit
using Plots
using Statistics
using StatsBase
using MLBase
using ROC

using Pipe
using Match

include("preps.jl")

# 1. Priprema i provera podataka
# Ucitavanje podataka za 100 voca
# Klase: 
#   0 - Sljiva
#   1 - Jabuka
#   2 - Banana
try
    global data = DataFrame(CSV.File("voce-pravi.csv"))
catch
    error("Greska pri otvaranju fajla: `voce-pravi.csv`.")
    exit(-1)
    println("Greska pri otvaranju fajla: `voce-pravi.csv`.")
end

# Podela na skup za obuku i testiranja
data_train, data_test = TrainTestSplit(data, 0.80)
input_plot = scatter(data_train.boja, data_train.velicina, title = "Voce, pre One Vs One", ylabel = "Velicina", xlabel = "Boja")
scatter!(input_plot, data_test.boja, data_test.velicina)



# 2. One versus One regresija

# Formiramo 3 klasifikacije (n * (n - 1) / 2)
    # n - br. klasa
    # Klasifikacija 0: Sljiva vs Jabuka
    # Klasifikacija 1: Sljiva vs Banana
    # Klasifikacija 2: Jabuka vs Banana
fm0 = @formula(voce != 2 ~ boja + velicina)
fm1 = @formula(voce != 1 ~ boja + velicina)
fm2 = @formula(voce != 0 ~ boja + velicina)

classification0 = glm(fm0, data_train, Binomial(), ProbitLink()) # Sljiva vs Jabuka
classification1 = glm(fm1, data_train, Binomial(), ProbitLink()) # Sljiva vs Banana
classification2 = glm(fm2, data_train, Binomial(), ProbitLink()) # Jabuka vs Banana

# Testranje podataka OvO regresijom
data_test_prediction0 = predict(classification0, data_test)
data_test_prediction1 = predict(classification1, data_test)
data_test_prediction2 = predict(classification2, data_test)

println("Predvidjeni podaci za klasu 0: $(round.(data_test_prediction0; digits = 2))")
println("\nPredvidjeni podaci za klasu 1: $(round.(data_test_prediction1; digits = 2))")
println("\nPredvidjeni podaci za klasu 2: $(round.(data_test_prediction2; digits = 2))")



# Racunanje matrica predvidjanja
data_test_prediction_class0 = predict_classes(data_test_prediction0, 0, 1)
data_test_prediction_class1 = predict_classes(data_test_prediction1, 0, 2)
data_test_prediction_class2 = predict_classes(data_test_prediction2, 1, 2)

println("Predvidjena voca #0: $(data_test_prediction_class0)")
println("\nPredvidjena voca #1: $(data_test_prediction_class1)")
println("\nPredvidjena voca #2: $(data_test_prediction_class2)")
println("\nVoca: $(data_test.voce)")


# Grafikoni
(roc_test0, auc_test0) = test_validity(data_test_prediction0, data_test, 0)
# sleep(5) # Pauza 5 sekundi da se vidi grafik
(roc_test1, auc_test1) = test_validity(data_test_prediction1, data_test, 1)
# sleep(5) # Pauza 5 sekundi da se vidi grafik
(roc_test2, auc_test2) = test_validity(data_test_prediction2, data_test, 2)

# plot(roc_test0, label="ROC kriva klasifikatora #0")
# plot(roc_test1, label="ROC kriva klasifikatora #1")
# plot(roc_test2, label="ROC kriva klasifikatora #2")

# @pipe data_test_prediction1 |> println.(_)
@pipe (roc_test0 .+ roc_test1 .+ roc_test2) ./ 3.0 |> plot(_, label = "ROC prosecna kriva klasifikatora")