import torch
from torchvision import transforms

import data_setup, engine, model_builder, train, utils

torch.manual_seed(42)

BATCH_SIZE = 32
NUM_EPOCHES = 5
LEARNING_RATE = 0.001
model_name = "script_model.pth"
model_save_dir = "models/"
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

transform = transforms.Compose([transforms.Resize((64,64)),
                    transforms.ToTensor()])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir,
                                                                              test_dir,
                                                                              transform,
                                                                              BATCH_SIZE)

model_0 = model_builder.TinyVGG(3, 10, len(class_names))

optimizer = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

results = engine.train(model_0,
                       train_dataloader,
                       test_dataloader,
                       optimizer,
                       loss_fn,
                       NUM_EPOCHES)

utils.save_model(model_0,
                 model_save_dir,
                 model_name)
