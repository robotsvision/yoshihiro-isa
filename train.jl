using Flux
using Flux: gradient, params, mse, update!  # Явное указание на использование этих функций


# Определение и обучение модели
inputs = [Float32[1, 1], Float32[1, 0], Float32[0, 1], Float32[0, 0]]
outputs = [Float32[0, 1, 0, 1], Float32[1, 0, 0, 1], Float32[0, 1, 1, 0], Float32[0, 0, 0, 0]]

# Преобразование в обучающий набор
train_data = [(inputs[i], outputs[i]) for i in 1:lastindex(inputs)]

# Создание модели
model = Chain(
  Dense(2, 10, relu),
  Dense(10, 4),
  softmax
)

# Функция потерь и оптимизатор
loss(x, y) = Flux.crossentropy(model(x), y)
optimizer = ADAM()

# Обучение модели
for epoch in 1:100
  for (x, y) in train_data
    gs = gradient(() -> loss(x, y), params(model))
    update!(optimizer, params(model), gs)
  end
end

using BSON

# Сохранение модели
BSON.@save "model.bson" model
