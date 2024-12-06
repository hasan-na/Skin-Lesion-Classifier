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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights # type: ignore

"Declare Constants"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
HEIGHT = 224
WIDTH = 224
EPOCHS = 30
LEARNING_RATE = 0.01
NUM_CLASSES = 7
FACTOR = 0.5
PATIENCE = 3
MIN_LR = 0.001
WEIGHT_DECAY = 0.00008

"Define Dataset Class"

class CustomDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name)
        label = torch.tensor(int(self.data.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)

        return image, label

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
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.4, 1.0), ratio=(3/4, 4/3)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomVerticalFlip(),
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

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES),
  )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=PATIENCE, min_lr=MIN_LR)
    precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average='macro').to(device)
    recall_metric = MulticlassRecall(num_classes=NUM_CLASSES, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(device)

    best_accuracy = 0.0

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

   
        epoch_precision = precision_metric.compute().item()
        epoch_recall = recall_metric.compute().item()
        epoch_f1 = f1_metric.compute().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00* correct / total

        print(f"   -  Training Accuracy: {epoch_acc:.3f}%, Training Loss: {epoch_loss:.3f}, Training Precision: {epoch_precision:.3f}, Training Recall: {epoch_recall:.3f}, Training F1-Score: {epoch_f1:.3f}\n")

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

        val_epoch_precision = precision_metric.compute().item()
        val_epoch_recall = recall_metric.compute().item()
        val_epoch_f1 = f1_metric.compute().item()
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100.0 * val_correct / val_total
        print(f"   -  Validation Accuracy: {val_epoch_acc:.2f}%, Validation Loss: {val_epoch_loss:.3f}, Validation Precision: {val_epoch_precision:.3f}, Validation Recall: {val_epoch_recall:.3f}, Validation F1-Score: {val_epoch_f1:.3f}\n")
    
        scheduler.step(val_epoch_loss)

        if val_epoch_acc > best_accuracy:
            best_accuracy = val_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved with accuracy: {val_epoch_acc:.2f}%\n")

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            test_total += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
            test_correct += (predicted == labels).sum().item()

            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)
    
    test_epoch_precision = precision_metric.compute().item()
    test_epoch_recall = recall_metric.compute().item()
    test_epoch_f1 = f1_metric.compute().item()
    test_epoch_loss = val_loss / len(val_loader)
    test_epoch_acc = 100.0 * val_correct / val_total
    print(f"   - Test Accuracy: {test_epoch_acc:.2f}%, Test Loss: {test_epoch_loss:.3f}, Test Precision: {test_epoch_precision:.3f}, Test Recall: {test_epoch_recall:.3f}, Test F1-Score: {test_epoch_f1:.3f}\n")

"Run Main Function"
if __name__ == '__main__':
    main()