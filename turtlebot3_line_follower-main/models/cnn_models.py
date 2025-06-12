# models/cnn_model.py
from torchvision.models import mobilenet_v2

class MobileNetLineFollower(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v2(pretrained=True)
        self.features = base.features
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        return self.classifier(x)