from matplotlib import pyplot as plt
import numpy as np
import torch

class Optimization:
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        # Set model to training mode
        self.model.train()
        # Forward pass
        y_pred = self.model(x.float())
        y_pred = y_pred.view(-1, 13)
        # Select max value from output
        # Compute Loss
        loss = self.loss_fn(y_pred, y)
        # Backward pass
        self.optimizer.zero_grad()
        # Loss requires grad
        # loss.requires_grad = True
        loss.backward()
        self.optimizer.step()

        # Return the loss
        return loss.item()

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs+1):
            batch_losses = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)#.to_float()
                y_batch = y_batch.to(self.device)#.to_float()
                train_loss = self.train_step(x_batch, y_batch)
                batch_losses.append(train_loss)
                
            train_loss = np.mean(batch_losses)
            self.train_losses.append(train_loss)

            # Validation
            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    # Set model to evaluation mode
                    self.model.eval()
                    # Forward pass
                    y_pred = self.model(x_val.float())
                    y_pred = y_pred.view(-1, 13)
                    # Compute Loss
                    val_loss = self.loss_fn(y_pred, y_val)
                    batch_val_losses.append(val_loss.item())
                val_loss = np.mean(batch_val_losses)
                self.val_losses.append(val_loss)
            
            # Printing progress
            #if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
                y_test = y_test.to(self.device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to('cpu').detach().numpy())
                predictions = [1 if x > 0.5 else 0 for x in predictions]
                values.append(y_test.to('cpu').detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.val_losses, label='Validation loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        plt.show()