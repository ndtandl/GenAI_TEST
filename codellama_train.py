# import torch
# import deepspeed
# import transformers
# import pandas as pd

# # Tải xuống mô hình
# model = transformers.AutoModel.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

# # Tạo cấu hình DeepSpeed
# config = deepspeed.config.DeepSpeedConfig()
# config.fp16_enabled = True
# config.zero_optimization_enabled = True

# # Chuẩn bị dữ liệu
# def load_data(filename):
#   data = pd.read_csv(filename)
#   train_data = data.iloc[:75]
#   test_data = data.iloc[75:]
#   train_data["code"] = train_data["code"].apply(lambda x: x.strip())
#   train_data["target"] = train_data["target"].apply(lambda x: x.strip())
#   train_data = train_data.to_numpy()
#   test_data = test_data.to_numpy()
#   return train_data, test_data

# train_data, test_data = load_data("./data/codellama_train.csv")

# # Huấn luyện mô hình
# def train_function(config, model, train_data, test_data):
#   # Huấn luyện mô hình
#   model.train()
#   for epoch in range(100):
#     losses = []
#     for batch in train_data:
#       loss = model(batch)
#       losses.append(loss)
#     print(f"Epoch {epoch} loss: {torch.mean(losses)}")

#   # Đánh giá mô hình
#   model.eval()
#   for batch in test_data:
#     loss = model(batch)
#     print(f"Loss on test data: {loss}")

# deepspeed.launch(
#     config,
#     init_process_group=True,
#     training_function=train_function,
#     train_data=train_data,
#     test_data=test_data,
# )


# import torch
# import deepspeed
# import transformers

# # Tải xuống dữ liệu huấn luyện
# data_path = "path/to/data"

# # Khởi tạo mô hình
# model = transformers.AutoModel.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

# # Khởi tạo DeepSpeed
# ds_config = deepspeed.config.DeepSpeedConfig()
# ds_config.zero_optimization_level = 2
# ds_config.fp16_enabled = True
# ds_config.gpu_batch_size = 4

# # Tạo optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# # Khởi tạo trainer
# trainer = deepspeed.training.DeepSpeedTrainer(model, optimizer, ds_config)

# # Huấn luyện mô hình
# trainer.train(data_path, epochs=10)

# Tải mô hình Codellama
model = Codellama.from_pretrained("codellama")

# Tạo trình huấn luyện DeepSpeed
trainer = deepspeed.initialize(model,
                             model_parallel_size=8,
                             local_rank=0,
                             host="localhost",
                             port=8788)

# Chuẩn bị dữ liệu huấn luyện từ file CSV
train_data = read_csv_data("train.csv")

# Huấn luyện mô hình
trainer.fit(train_data, epochs=10)