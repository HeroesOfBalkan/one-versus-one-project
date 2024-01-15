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

const NUMBER_OF_CLASSES = 3
const NUMBER_OF_CLASSIFICATIONS = NUMBER_OF_CLASSES * (NUMBER_OF_CLASSES - 1) / 2.0

# Klase:
@enum FruitType begin
    Unknown = 0 # Nepoznata - 0
    Plum = 1    # Sljiva - 1
    Apple = 2   # Jabuka - 2
    Banana = 3  # Banana - 3
end

# Priprema i provera podataka
# Ucitavanje podataka za 100+ voca
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
    # Klasifikacija 1: Sljiva vs Jabuka
    # Klasifikacija 2: Sljiva vs Banana
    # Klasifikacija 3: Jabuka vs Banana
fm1 = @formula(voce != 3 ~ boja + velicina)
fm2 = @formula(voce != 2 ~ boja + velicina)
fm3 = @formula(voce != 1 ~ boja + velicina)

classification1 = glm(fm1, data_train, Binomial(), ProbitLink()) # Sljiva vs Jabuka
classification2 = glm(fm2, data_train, Binomial(), ProbitLink()) # Sljiva vs Banana
classification3 = glm(fm3, data_train, Binomial(), ProbitLink()) # Jabuka vs Banana

# Predvidjanje podataka OvO regresijom
data_test_prediction1 = predict(classification1, data_test)
data_test_prediction2 = predict(classification2, data_test)
data_test_prediction3 = predict(classification3, data_test)

println("Predvidjeni podaci za klasu 1: $(round.(data_test_prediction1; digits = 2))")
println("\nPredvidjeni podaci za klasu 2: $(round.(data_test_prediction2; digits = 2))")
println("\nPredvidjeni podaci za klasu 3: $(round.(data_test_prediction3; digits = 2))")



# Racunanje niza predvidjenih klasa
data_test_prediction_class1 = predict_classes(data_test_prediction1, 1, 2)
data_test_prediction_class2 = predict_classes(data_test_prediction2, 1, 3)
data_test_prediction_class3 = predict_classes(data_test_prediction3, 2, 3)


predictions = zeros(length(data_test_prediction1))
const GUARANTEE_MAJORITY = ceil(UInt64, NUMBER_OF_CLASSIFICATIONS / 2.0)
for index in eachindex(predictions)
    # Glasovi klasa
    plums::UInt64 = 0
    apples::UInt64 = 0
    bananas::UInt64 = 0

    # Predvidnjanja iz testa 1
    if data_test_prediction1[index] == 1
        plums += 1
    elseif data_test_prediction1[index] == 2
        apples += 1
    end

    # Predvidnjanja iz testa 2
    if data_test_prediction2[index] == 1
        plums += 1
    elseif data_test_prediction2[index] == 3
        bananas += 1
    end
    
    # Predvidnjanja iz testa 3
    if data_test_prediction3[index] == 2
        apples += 1
    elseif data_test_prediction3[index] == 3
        bananas += 1
    end
    
    (majority_votes_amount, majority_class) = findmax([plums, apples, bananas])

    if majority_votes_amount < GUARANTEE_MAJORITY
        continue
    end

    predictions[index] = majority_class
end

println("Predvidjena voca: $(predictions)")
println("\nVoca: $(data_test.voce)")


# Grafikoni
(roc_test, auc_test) = test_validity(predictions, data_test)
# @pipe (roc_test1 .+ roc_test2 .+ roc_test3) ./ 3.0 |> plot(_, label = "ROC prosecna kriva klasifikatora")
plot(roc_test, label = "ROC prosecna kriva modela")