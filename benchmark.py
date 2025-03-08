import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models.attack_model import AttackModel
from advtorchattacks import FGSM, PGD, CW, DeepFool, MIFGSM, AutoAttack

# Load model
model = AttackModel(model_name="resnet18", pretrained=False).model

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.FakeData(transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# Get sample data
images, labels = next(iter(test_loader))

# List of attacks
attacks = {
    "FGSM": FGSM(model, epsilon=0.03),
    "PGD": PGD(model, epsilon=0.03, alpha=0.01, steps=10),
    "CW": CW(model, steps=10, c=1.0),
    "DeepFool": DeepFool(model, steps=10),
    "MI-FGSM": MIFGSM(model, epsilon=0.03, alpha=0.01, steps=10, decay=1.0),
    "AutoAttack": AutoAttack(model, epsilon=0.03, steps=10)
}

# Run attacks & calculate success rate
for name, attack in attacks.items():
    adv_images = attack(images, labels)
    preds = model(adv_images).argmax(dim=1)
    success_rate = (preds != labels).float().mean().item() * 100
    print(f"{name} Attack Success Rate: {success_rate:.2f}%")
