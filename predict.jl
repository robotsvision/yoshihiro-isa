using Flux
using BSON
using NNlib

# загрузка модели
BSON.@load "model.bson" model

# предположим, у нас есть новые данные для тестирования
new_data = Float32[1, 0]  # новый вектор входных данных

# делаем предсказание
predicted_output = model(new_data)

println("Predicted output: ", predicted_output)