import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from advtorchattacks.models.attack_model import AttackModel
from advtorchattacks.advtorchattacks.fgsm import FGSM
from advtorchattacks.advtorchattacks.pgd import PGD
from advtorchattacks.advtorchattacks.cw import CW
from advtorchattacks.advtorchattacks.deepfool import DeepFool
from advtorchattacks.advtorchattacks.mifgsm import MIFGSM
from advtorchattacks.advtorchattacks.autoattack import AutoAttack

# Load model
model = AttackModel(pretrained=False).model

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.FakeData(transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Get sample data
images, labels = next(iter(dataloader))

# List of attacks
attacks = {
    "FGSM": FGSM(model, epsilon=0.03),
    "PGD": PGD(model, epsilon=0.03, alpha=0.01, steps=40),
    "CW": CW(model, steps=100, c=1.0),
    "DeepFool": DeepFool(model, steps=50),
    "MI-FGSM": MIFGSM(model, epsilon=0.03, alpha=0.01, steps=10, decay=1.0),
    "AutoAttack": AutoAttack(model, epsilon=0.03, steps=40)
}

# Apply each attack and store results
adv_images = {}
for name, attack in attacks.items():
    adv_images[name] = attack(images, labels)
    print(f"{name} attack applied!")

# Print results
print("\n--- Adversarial Labels ---")
for name, adv_img in adv_images.items():
    preds = model(adv_img).argmax(dim=1).tolist()
    print(f"{name}: {preds}")

# Save adversarial images
for name, adv_img in adv_images.items():
    torch.save(adv_img, f"examples/{name}_adv_images.pt")

print("\nAll adversarial images saved!")
