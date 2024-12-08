'''
Citation for Dataset:
HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161

MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368

[1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).
'''
import sys
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
import matplotlib.pyplot as plt #type: ignore
from imblearn.over_sampling import SMOTE #type: ignore
from sklearn.utils.class_weight import compute_class_weight #type: ignore

"Declare Constants"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
SMOTE_BATCH_SIZE = 200
HEIGHT = 224
WIDTH = 224
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 7
FACTOR = 0.5
PATIENCE = 3
MIN_LR = 0.001
WEIGHT_DECAY = 0.0001

"Dataset for SMOTE Data"
'''
class CustomDatasetSMOTE(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = image.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((image * 255).astype(np.uint8))
            image = self.transform(image)

        return image, label
'''

"Use Smote to Balance Dataset"
'''

def apply_smote(dataset):
    train_data = []
    train_labels = []
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        train_data.append(np.array(img.flatten()))
        train_labels.append(np.argmax(label))

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    class_counts = np.bincount(train_labels)
    majority_count = class_counts.max()
    
    sampling_strategy = {
        label: min(int(majority_count * 0.75), count * 2)
        for label, count in enumerate(class_counts)
        if count < majority_count
    }

    k_neighbors = min(1, SMOTE_BATCH_SIZE // (NUM_CLASSES * 2))
    
    print("Original class distribution:", class_counts)
    print("Sampling strategy:", sampling_strategy)

    smote = SMOTE(sampling_strategy = sampling_strategy, random_state=42, k_neighbors=k_neighbors)
    num_batches = len(train_data) // SMOTE_BATCH_SIZE
    smote_data, smote_labels = [], []

    for i in range(num_batches):
        try:
            batch_data = train_data[i * SMOTE_BATCH_SIZE: (i + 1) * SMOTE_BATCH_SIZE]
            batch_labels = train_labels[i * SMOTE_BATCH_SIZE: (i + 1) * SMOTE_BATCH_SIZE]

            batch_class_counts = np.bincount(batch_labels)
            print(f"Batch {i} class distribution:", batch_class_counts)

            min_samples = min(batch_class_counts[batch_class_counts > 0])
            if min_samples <= k_neighbors:
                print(f"Skipping batch {i}: insufficient samples (min={min_samples}, k={k_neighbors})")
                continue

            batch_smote_data, batch_smote_labels = smote.fit_resample(batch_data, batch_labels)
            smote_data.append(batch_smote_data)
            smote_labels.append(batch_smote_labels)

        except ValueError:
            print("No SMOTE data generated for batch:", i)
            continue
    
    if smote_data and smote_labels:
        smote_data = np.concatenate(smote_data, axis=0)
        smote_labels = np.concatenate(smote_labels, axis=0)
        print("Class distribution after SMOTE:", np.bincount(smote_labels))
    else:
        raise ValueError("No data after SMOTE resampling.")
    
    return smote_data, smote_labels
'''

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
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.0), ratio=(3/4, 4/3)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2)),
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

'''
Custom EfficientNet Model
'''
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        for param in list(self.model.parameters())[:-4]:
            param.requires_grad = False
            
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
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

    #smote_data, smote_labels = apply_smote(train_dataset)

    #smote_images = torch.tensor(smote_data, dtype=torch.float32).view(-1, 3, HEIGHT, WIDTH)
    #smote_labels = torch.tensor(smote_labels, dtype=torch.long)

    #balanced_train_dataset = CustomDatasetSMOTE(smote_images, smote_labels, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for images, labels in train_loader:
        print("Batch labels:", labels)
        break 
    
    model = CustomEfficientNet(NUM_CLASSES)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=PATIENCE, min_lr=MIN_LR)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3)

    precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average=None).to(device)
    recall_metric = MulticlassRecall(num_classes=NUM_CLASSES, average=None).to(device)
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average=None).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

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

   
        epoch_precision_per_class = precision_metric.compute().cpu().numpy()
        epoch_recall_per_class = recall_metric.compute().cpu().numpy()
        epoch_f1_per_class = f1_metric.compute().cpu().numpy()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00* correct / total

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

        print(f"   -  Validation Accuracy: {val_epoch_acc:.3f}%, Validation Loss: {val_epoch_loss:.3f}")
        for idx, (precision, recall, f1) in enumerate(zip(val_precision_per_class, val_recall_per_class, val_f1_per_class)):
            print(f"      Class {idx}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")
    
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
    
    test_precision_per_class = precision_metric.compute().cpu().numpy()
    test_recall_per_class = recall_metric.compute().cpu().numpy()
    test_f1_per_class = f1_metric.compute().cpu().numpy()
    test_epoch_loss = test_loss / len(test_loader)
    test_epoch_acc = 100.0 * test_correct / test_total

    print(f"   -  Test Accuracy: {test_epoch_acc:.3f}%, Test Loss: {test_epoch_loss:.3f}")
    for idx, (precision, recall, f1) in enumerate(zip(test_precision_per_class, test_recall_per_class, test_f1_per_class)):
            print(f"      Class {idx}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")

"Run Main Function"
if __name__ == '__main__':
    main()