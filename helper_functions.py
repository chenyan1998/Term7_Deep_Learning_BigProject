import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def cal_acc(outputs,labels):
    pred_labels=torch.max(outputs,1)[1]
    equality = torch.eq(pred_labels,labels).float()
    accuracy = torch.mean(equality)
    return accuracy


def validation(model, validation_loader, criterion):
    validation_loss = 0
    accuracy = 0
    
    for images, labels in validation_loader:
                
        outputs = model(images)
        validation_loss += criterion(outputs,torch.flatten(labels.long())).item()
        
        accuracy += cal_acc(outputs,labels)

    val_loss=validation_loss/len(validation_loader)
    val_acc=accuracy/len(validation_loader)
    return val_loss,val_acc

def plot_loss(model):
    plt.plot(model.history['train_loss'], label='Loss (training data)')
    plt.plot(model.history['validation_loss'], label='Loss (validation data)')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc="upper left")
    plt.show()
    
    return None

def plot_accuracy(model):
    plt.plot(model.history['train_accuracy'], label='Accuracy (training data)')
    plt.plot(model.history['validation_accuracy'], label='Accuracy (validation data)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc="upper left")
    plt.show()
    return None