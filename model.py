import torch.nn as nn
import torchvision.models as models

class FineGrainedClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FineGrainedClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x