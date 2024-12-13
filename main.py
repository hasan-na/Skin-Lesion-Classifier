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
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights # type: ignore
from torchvision.models import resnet50, ResNet50_Weights # type: ignore
from torchvision.models import densenet121, DenseNet121_Weights # type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns   #type: ignore
from torchvision.utils import make_grid #type: ignore
from torchviz import make_dot #type: ignore
from torchinfo import summary #type: ignore
from sklearn.metrics import roc_curve, auc


"Declare Constants"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
HEIGHT = 224
WIDTH = 224
EPOCHS = 50
LEARNING_RATE = 0.0001  
NUM_CLASSES = 7
FACTOR = 0.5
PATIENCE = 2           
#WEIGHT_DECAY = 0.00001

def plot_roc_curves(model, train_loader, val_loader, test_loader, device, save_dir='./plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    def get_roc_metrics(loader):
        model.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(NUM_CLASSES):
            labels_binary = (all_labels == i).astype(int)
            fpr[i], tpr[i], _ = roc_curve(labels_binary, all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        return fpr, tpr, roc_auc

    # Get metrics for each dataset
    train_fpr, train_tpr, train_auc = get_roc_metrics(train_loader)
    val_fpr, val_tpr, val_auc = get_roc_metrics(val_loader)
    test_fpr, test_tpr, test_auc = get_roc_metrics(test_loader)
    
    # Plot ROC curves
    plt.figure(figsize=(15, 5))
    class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    splits = ['Training', 'Validation', 'Testing']
    
    for idx, (split, fpr, tpr, roc_auc) in enumerate(zip(
        splits,
        [train_fpr, val_fpr, test_fpr],
        [train_tpr, val_tpr, test_tpr],
        [train_auc, val_auc, test_auc]
    )):
        plt.subplot(1, 3, idx+1)
        for i in range(NUM_CLASSES):
            plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{split} ROC Curves')
        plt.legend(loc="lower right", fontsize='small')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curves.png')
    plt.close()

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
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.25),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
                transforms.RandomErasing(p=0.2)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        
    def __call__(self, img):
        return self.transform(img)
    
    
"Custom Efficient Net Model"
class CustomDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet, self).__init__()
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        
        # Replace original classifier
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()  # Remove original classifier
        
        # Custom classifier with correct dimensions
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Get features from DenseNet
        features = self.model(x)
        # Pass through our classifier
        return self.classifier(features)
      
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

    model = CustomDenseNet(num_classes=NUM_CLASSES)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)  # Use AdamW
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=PATIENCE)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.002, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.2, div_factor=10, final_div_factor=100)

    precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average=None).to(device)
    recall_metric = MulticlassRecall(num_classes=NUM_CLASSES, average=None).to(device)
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average=None).to(device)

    best_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_recalls = []
    val_recalls = []

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
            scheduler.step()

            running_loss += loss.item()
            correct +=(labels==predicted).sum().item()
      
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)
   
        epoch_precision_per_class = precision_metric.compute().cpu().numpy()
        epoch_recall_per_class = recall_metric.compute().cpu().numpy()
        epoch_f1_per_class = f1_metric.compute().cpu().numpy()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00* correct / total
        train_losses.append(epoch_loss)
        train_recalls.append(np.mean(epoch_recall_per_class))

        print(f"   -  Training Accuracy: {epoch_acc:.3f}%, Training Loss: {epoch_loss:.3f}")
        for idx, (precision, recall, f1) in enumerate(zip(epoch_precision_per_class, epoch_recall_per_class, epoch_f1_per_class)):
            print(f"      Class {idx}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")

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

        val_precision_per_class = precision_metric.compute().cpu().numpy()
        val_recall_per_class = recall_metric.compute().cpu().numpy()
        val_f1_per_class = f1_metric.compute().cpu().numpy()
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100.0 * val_correct / val_total

        val_losses.append(val_epoch_loss)
        val_recalls.append(np.mean(val_recall_per_class))

        print(f"   -  Validation Accuracy: {val_epoch_acc:.3f}%, Validation Loss: {val_epoch_loss:.3f}")
        for idx, (precision, recall, f1) in enumerate(zip(val_precision_per_class, val_recall_per_class, val_f1_per_class)):
            print(f"      Class {idx}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")
    
        #scheduler.step(val_epoch_loss)

        if val_epoch_acc > best_accuracy:
            best_accuracy = val_epoch_acc
            torch.save(model.state_dict(), 'best_model_DenseNet2.pth')
            print(f"Model saved with accuracy: {val_epoch_acc:.2f}%\n")

    model.load_state_dict(torch.load('best_model_DenseNet2.pth'))
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
    
    test_precision_per_class = precision_metric.compute().cpu().numpy()
    test_recall_per_class = recall_metric.compute().cpu().numpy()
    test_f1_per_class = f1_metric.compute().cpu().numpy()
    test_epoch_loss = test_loss / len(test_loader)
    test_epoch_acc = 100.0 * test_correct / test_total

    print(f"   -  Test Accuracy: {test_epoch_acc:.3f}%, Test Loss: {test_epoch_loss:.3f}")
    for idx, (precision, recall, f1) in enumerate(zip(test_precision_per_class, test_recall_per_class, test_f1_per_class)):
            print(f"      Class {idx}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")
    
    plot_roc_curves(model, train_loader, val_loader, test_loader, device)

"Run Main Function"
if __name__ == '__main__':
    main()