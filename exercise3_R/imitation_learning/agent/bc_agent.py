import torch
from agent.networks import CNN


class BCAgent:
    
    def __init__(self, device='cpu', history_length=1, lr=1e-4, n_classes=4):
        # TODO: Define network, loss function, optimizer
        self.device = torch.device(device)

        self.net = CNN(history_length=history_length, n_classes=n_classes)
        self.net.to(self.device)

        self.lossfn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        X_batch = X_batch.float().to(self.device)
        y_batch = y_batch.long().to(self.device)

        # TODO: forward + backward + optimize
        pred = self.net(X_batch)
        loss = self.lossfn(pred, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, X):
        # TODO: forward pass
        X = X.float().to(self.device)
        outputs = self.net(X)
        # print(torch.nn.functional.softmax(outputs))
        output = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        return output

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))
