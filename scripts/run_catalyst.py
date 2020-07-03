import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from catalyst import dl
from catalyst.utils import metrics, meters
import numpy as np
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback, EarlyStoppingCallback
from catalyst.data import ImageReader, ScalarReader, ReaderCompose


callbacks = [
    EarlyStoppingCallback(patience=10, metric='loss', minimize=True, min_delta=0),
    AccuracyCallback(num_classes=2),
    AUCCallback(num_classes=2, input_key="targets_one_hot"),

]

num_samples, num_features = int(1e4), int(1e1)
model = torch.nn.Linear(num_features, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)


X = torch.rand(num_samples, num_features)
y = np.ones(num_samples)
y[:int((num_samples/2))] = 0
np.random.shuffle(y)
y = torch.from_numpy(y).long()
y = y.reshape(num_samples)
a = torch.rand(num_samples,1)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

class CustomRunner(dl.Runner):


    def predict_batch(self, batch):
    #    # model inference step
        return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

    def _handle_batch(self, batch):
        # model train/valid step
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)




        # and network output to state `output`
        # we recommend to use key-value storage to make it Callbacks-friendly
        self.state.output = {"logits": y_hat}
        y_onehot = torch.FloatTensor(x.shape[0], 2)
        y_onehot.zero_()
        y_onehot[np.arange(x.shape[0]),y] = 1
        self.state.input = {"features": x, "targets": y, "targets_one_hot": y_onehot}

        #print(self.state.input)
        #print(self.state.output)

        #accuracy01 = metrics.accuracy(y_hat, y)
        #auc = meters.aucmeter(y_hat,y)
        self.state.batch_metrics.update(
            {"loss": loss}
        )

        if self.state.is_train_loader:
            print('in train')
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()
        else:
            print('in val')

runner = CustomRunner(val=0)
# model training
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    callbacks=callbacks,
    logdir="./logs",
    num_epochs=6,
    verbose=True,
    load_best_on_end=True,
)
# model inference
#for prediction in runner.predict_loader(loader=loaders["valid"]):
#    assert prediction.detach().cpu().numpy().shape[-1] == 10
# model tracing
#traced_model = runner.trace(loader=loaders["valid"])