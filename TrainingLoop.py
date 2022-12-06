from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import random

def random_blur(p=0.5):
    if random.uniform(0, 1) < p:
        return transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    return transforms.RandomHorizontalFlip(p=0)

class Optimization:
    def __init__(self, model, loss_fn, optimizer, device, name):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracy = [] # accuracy
        self.val_accuracy = [] # accuracy
        self.name = name
    
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
        # calculate accuracy
        prediction = torch.argmax(y_pred, dim=1)
        correct = (prediction == y).float()
        accuracy = correct.sum() / len(correct)
        
        
        

        # Return the loss
        return loss.item(), accuracy

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs+1):
            batch_losses = []
            batch_accuracy = []

            for x_batch, y_batch in train_loader:
                preprocess = transforms.Compose([
                                            transforms.RandomHorizontalFlip(p=1/4),
                                            transforms.RandomPerspective(p=1/4),
                                            random_blur(p=1/4),
                                            # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                                            # transforms.ToTensor(),
                                            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])
                x_batch = preprocess(x_batch)
                x_batch = x_batch.to(self.device)#.to_float()
                y_batch = y_batch.to(self.device)#.to_float()
                train_loss, accuracy = self.train_step(x_batch, y_batch)
                batch_losses.append(train_loss)
                batch_accuracy.append(accuracy.cpu())
                # print("Batch loss: ", train_loss)
                
            train_loss = np.mean(batch_losses)
            train_accuracy = np.mean(batch_accuracy)
            self.train_losses.append(train_loss)
            self.train_accuracy.append(train_accuracy)

            # Validation
            best_val_loss = 100000
            with torch.no_grad():
                batch_val_losses = []
                batch_val_accuracy = []
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
                    # compute accuracy
                    prediction = torch.argmax(y_pred, dim=1)
                    correct = (prediction == y_val).float()
                    accuracy = correct.sum() / len(correct)
                    batch_val_accuracy.append(accuracy.cpu())
                    
                val_loss = np.mean(batch_val_losses)
                self.val_losses.append(val_loss)
                val_accuracy = np.mean(batch_val_accuracy)
                self.val_accuracy.append(val_accuracy)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), f'models/{self.name}/model.pt')
                    print("Model saved")
                
                if epoch >= 4:
                    if best_val_loss in self.val_losses[-4:]:
                        print("Early stopping")
                        break
            
            # Printing progress
            #if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    def evaluate(self, test_loader):
        with torch.no_grad():
            predictions = []
            props = []
            values = []
            i=0
            for x_test, y_test in test_loader:
                i+=1
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                yhat = self.model(x_test.float())
                predictions.append(np.argmax(yhat.to('cpu').detach().numpy(), axis=1))
                values.append(y_test.to('cpu').detach().numpy())
                props.append(yhat.to('cpu').detach().numpy())
                if i % 100 == 0:
                    print(i)
                

        return predictions, values, props

    def plot_losses(self):
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.val_losses, label='Validation loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        # Save figure
        plt.savefig(f'models/{self.name}/losses.png')
        # clear plot
        plt.clf()
        
    def plot_accuracy(self):
        plt.plot(self.train_accuracy, label='Training accuracy')
        plt.plot(self.val_accuracy, label='Validation accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        # Save figure
        plt.savefig(f'models/{self.name}/accuracy.png')
        # clear plot
        plt.clf()