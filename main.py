import numpy as np
import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
from torch.utils.data import DataLoader, Dataset #type: ignore
import os
import pandas as pd #type: ignore
import torchvision.transforms as transforms #type: ignore
from PIL import Image #type: ignore
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score #type: ignore
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights # type: ignore
from torchvision.models import densenet169, DenseNet169_Weights # type: ignore
from sklearn.metrics import f1_score, roc_curve, auc # type: ignore
import matplotlib.pyplot as plt # type: ignore
"Declare Constants"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
HEIGHT = 224
WIDTH = 224
EPOCHS = 30
LEARNING_RATE = 0.003  
NUM_CLASSES = 7
FACTOR = 0.5
PATIENCE = 2           
MIN_LR = 0.000125
WEIGHT_DECAY = 0.0015

"Define Dataset Class"

class CustomDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self.data.iloc[:, 1:].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        class_label = torch.argmax(label)

        if self.transform:
            image = self.transform(image)

        return image, class_label

"Define Transformations"
class Transformations:
    def __init__(self, height, width, mean, std, train = True):
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std
        self.train = train
        self.transform = self.get_transforms()

    def get_transforms(self):
        if self.train:
            return transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0), ratio=(3/4, 4/3)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        
    def __call__(self, img):
        return self.transform(img)
    
"Custom Mobile Net Model"
class CustomMobileNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomMobileNet, self).__init__()
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        
        # Get the input features from the last conv layer
        last_channel = self.model.classifier[0].in_features
        
        # Replace classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(last_channel, 1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        # Forward pass through the entire model
        return self.model(x)
      
"Main Function"
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(current_dir, 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
    test_csv = os.path.join(current_dir, 'ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv')
    val_csv = os.path.join(current_dir, 'ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv')

    train_img_dir = os.path.join(current_dir, 'ISIC2018_Task3_Training_Input')
    test_img_dir = os.path.join(current_dir, 'ISIC2018_Task3_Test_Input')
    val_img_dir = os.path.join(current_dir, 'ISIC2018_Task3_Validation_Input')

    train_transform = Transformations(HEIGHT, WIDTH, MEAN, STD, train=True)
    test_transform = Transformations(HEIGHT, WIDTH, MEAN, STD, train=False)
    val_transform = Transformations(HEIGHT, WIDTH, MEAN, STD, train=False)

    train_dataset = CustomDataset(train_csv, train_img_dir, transform=train_transform)
    test_dataset = CustomDataset(test_csv, test_img_dir, transform=test_transform)
    val_dataset = CustomDataset(val_csv, val_img_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CustomMobileNet(num_classes=NUM_CLASSES)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, momentum=0.85, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=PATIENCE, min_lr=MIN_LR)

    precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average='macro').to(device)
    recall_metric = MulticlassRecall(num_classes=NUM_CLASSES, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(device)

    best_accuracy = 0.0
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Learning Rate: {current_lr}")
  
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct +=(labels==predicted).sum().item()
      
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)

        #scheduler.step()
   
        epoch_precision = precision_metric.compute().cpu().numpy()
        epoch_recall = recall_metric.compute().cpu().numpy()
        epoch_f1 = f1_metric.compute().cpu().numpy()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00* correct / total

        print(f"   - Training Loss: {epoch_loss:.3f}, Training Accuracy: {epoch_acc:.2f}%, Precision: {epoch_precision:.3f}, Recall: {epoch_recall:.3f}, F1-Score: {epoch_f1:.3f}\n")

        model.eval()  
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        with torch.no_grad(): 
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                val_total += labels.size(0)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                val_correct += (predicted == labels).sum().item()

                precision_metric.update(predicted, labels)
                recall_metric.update(predicted, labels)
                f1_metric.update(predicted, labels)

        val_epoch_precision = precision_metric.compute().cpu().numpy()
        val_epoch_recall = recall_metric.compute().cpu().numpy()
        val_epoch_f1 = f1_metric.compute().cpu().numpy()
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100.0 * val_correct / val_total

        print(f"   - Validation Loss: {val_epoch_loss:.3f}, Validation Accuracy: {val_epoch_acc:.2f}%, Precision: {val_epoch_precision:.3f}, Recall: {val_epoch_recall:.3f}, F1-Score: {val_epoch_f1:.3f}\n")
        scheduler.step(val_epoch_loss)

        train_accuracies.append(epoch_acc)
        val_accuracies.append(val_epoch_acc)
        train_losses.append(epoch_loss)
        val_losses.append(val_epoch_loss)

        if val_epoch_acc > best_accuracy:
            best_accuracy = val_epoch_acc
            torch.save(model.state_dict(), 'best_model_MobileNet.pth')
            print(f"Model saved with accuracy: {val_epoch_acc:.2f}%\n")

    model.load_state_dict(torch.load('best_model_MobileNet.pth'))
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    test_true = []
    test_pred_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            test_total += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
            test_correct += (predicted == labels).sum().item()

            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)

            test_true.extend(labels.cpu().numpy())
            test_pred_probs.extend(probs.cpu().numpy())  # Store probabilities instead of predictions
    
    test_epoch_precision = precision_metric.compute().cpu().numpy()
    test_epoch_recall = recall_metric.compute().cpu().numpy()
    test_epoch_f1 = f1_metric.compute().cpu().numpy()
    test_epoch_loss = test_loss / len(test_loader)
    test_epoch_acc = 100.0 * test_correct / test_total

    print(f"   - Test Loss: {test_epoch_loss:.3f}, Test Accuracy: {test_epoch_acc:.2f}%, Precision: {test_epoch_precision:.3f}, Recall: {test_epoch_recall:.3f}, F1-Score: {test_epoch_f1:.3f}\n")


"Run Main Function"
if __name__ == '__main__':
    main()