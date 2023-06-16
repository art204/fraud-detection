import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch import nn


def train_epochs(model, optimizer, criterion, metric, train_loader,
          num_epochs, scheduler=None, device='cpu', verbose=True):

    train_losses = []
    train_metrics = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, running_metric = 0, 0
        pbar = tqdm(train_loader, desc=f'Training {epoch}/{num_epochs}') \
            if verbose else train_loader

        for i, (X_batch, y_batch) in enumerate(pbar, 1):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric_value = metric(predictions, y_batch)
                if type(metric_value) == torch.Tensor:
                    metric_value = metric_value.item()
                running_loss += loss.item() * X_batch.shape[0]
                running_metric += metric_value * X_batch.shape[0]

        if scheduler is not None:
            scheduler.step()

        train_losses += [running_loss / len(train_loader.dataset)]
        train_metrics += [running_metric / len(train_loader.dataset)]

    return train_metrics[-1]

def train(model, optimizer, scheduler, X_train, y_train):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_set = TensorDataset(torch.from_numpy(X_train.values).to(torch.float32), torch.tensor(y_train))
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
    num_epochs = 3
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    metric = lambda y_pred, y_true: (y_true == y_pred.argmax(dim=1)).sum() / y_true.shape[0]
    train_epochs(model, optimizer, criterion, metric, train_loader, num_epochs, scheduler, device)

def predict(model, x):
    with torch.no_grad():
        model.eval()
        pred = model(torch.from_numpy(x.values).to(torch.float32))

    softmax = nn.Softmax(dim=1)
    y_pred_softmax = softmax(pred)
    y_pred = y_pred_softmax.argmax(dim=1)
    return y_pred.numpy()
