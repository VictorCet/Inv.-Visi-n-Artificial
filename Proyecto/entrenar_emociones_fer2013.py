
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import io

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 128), nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

class FERDataset(Dataset):
    def __init__(self, split="train"):
        dataset = load_dataset("Jeneral/fer-2013", split=split)
        self.images = dataset["img_bytes"]
        self.labels = dataset["labels"]
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.images[idx]))
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

def entrenar_modelo():
    train_dataset = FERDataset("train")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = EmotionCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "modelo_emociones.pt")
    print("Modelo entrenado y guardado como modelo_emociones.pt")

if __name__ == "__main__":
    entrenar_modelo()
