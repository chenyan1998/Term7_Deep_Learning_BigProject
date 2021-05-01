import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

import torch
import torch.nn as nn

def cal_acc(outputs,labels):
    pred_labels=torch.max(outputs,1)[1]
    equality = torch.eq(pred_labels,labels).float()
    accuracy = torch.mean(equality)
    return accuracy


def validation(model, validation_loader, criterion):
    '''
    The function take in the model, dataloader, lossfunction to calculate the loss and accuray, 
    it returns list of predicted labels and original labels as well  
    
    Param:
    model: Model to make prediction
    validation_loader: DataLoader that loads the data you want to make prediction on
    criterion: Loss function to calculate loss
    
    Return:
    val_loss: Loss value 
    val_acc: Accuracy value
    output_list: List of predicted labels
    labels_list: List of original labels
    '''
    validation_loss = 0
    accuracy = 0
    output_list=[]
    labels_list=[]
    for images, labels in validation_loader:
        
        labels_list.append(labels)
        outputs = model(images)
        output_list.append(torch.max(outputs,1)[1])
        validation_loss += criterion(outputs,torch.flatten(labels.long())).item()
        
        accuracy += cal_acc(outputs,labels)

    val_loss=validation_loss/len(validation_loader)
    val_acc=accuracy/len(validation_loader)
    return val_loss,val_acc,output_list,labels_list

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



def evaluate(y_test, y_pred):
    result={}
      # Calculate AUC
    result['AUC']=roc_auc_score(y_test,y_pred)
    #   print("AUC is: ", roc_auc_score(y_test,y_pred) )
      # recall and precision
    result['report']=classification_report(y_test, y_pred)
    #   print(classification_report(y_test, y_pred))
      # confusion matrix
    result['ConMatrix']=confusion_matrix(y_test, y_pred)
    #   print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

      # calculate points for ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
      # Plot ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_score(y_test, y_pred))
    ax.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate or (1 - Specifity)')
    ax.set_ylabel('True Positive Rate or (Sensitivity)')
    ax.set_title('Receiver Operating Characteristic')


    return result,fig