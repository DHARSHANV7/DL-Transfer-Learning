

# NAME:Dharshan V
# REG NO:212224240035

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset

An organization has a dataset of labeled images belonging to multiple categories, and accurate classification of these images is important for automation and decision-making. However, training a deep neural network from scratch requires a large amount of data and computational resources.

To address this, the organization plans to use transfer learning with the VGG19 architecture, which is a pre-trained convolutional neural network that has already learned rich feature representations from large-scale image datasets. By reusing this model, the system can efficiently extract important visual features from images.

The model will be fine-tuned using the given dataset so that it adapts to the specific classification task. This reduces training time and improves performance, especially when the dataset is limited.

After training, the model will be used to classify new, unseen images and evaluate its accuracy. The objective is to achieve high classification performance while minimizing computational cost and training effort.

## Neural Network Model


<img width="818" height="740" alt="image" src="https://github.com/user-attachments/assets/85b8d7de-e3f5-4fc3-99e0-09fa1092dcb9" />



## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transform

### STEP 2: 

Load training and testing datasets using ImageFolder


### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 


Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 
Define loss function (CrossEntropyLoss) and optimizer (Adam). Train the model and plot the loss curve.


### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.



## PROGRAM

### Name:Dharshan V

### Register Number:212224240035

```python
# Load Pretrained Model and Modify for Transfer Learning

model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer

criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

# Train the model

def train_model(model, train_loader, test_loader, num_epochs=100):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # ----- Training -----
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(test_loader)
        val_losses.append(epoch_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Validation Loss: {epoch_val_loss:.4f}')

    # ----- Plot -----
    print("Name:  Dharshan V")
    print("Register Number: 212224240035")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return train_losses, val_losses


```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot
<img width="902" height="687" alt="Screenshot 2026-05-14 094155" src="https://github.com/user-attachments/assets/1f9905c1-52f4-48d2-a253-3b878242dda7" />


## Confusion Matrix
<img width="1017" height="701" alt="Screenshot 2026-05-14 094209" src="https://github.com/user-attachments/assets/68aa420c-05e6-440d-a23b-6e6fc288ea40" />


## Classification Report
<img width="560" height="242" alt="Screenshot 2026-05-14 094223" src="https://github.com/user-attachments/assets/d2dfbda8-2f3d-406f-b310-e8cb5c565c70" />


### New Sample Data Prediction

<img width="449" height="404" alt="image" src="https://github.com/user-attachments/assets/a76b86d7-0950-4e4a-9e66-09236274d418" />


<img width="468" height="401" alt="image" src="https://github.com/user-attachments/assets/57ab3d02-9ca8-4030-aaba-5f9f0bd57443" />


## RESULT

Hence a Neural Network transfer model is developed using transfer learning
