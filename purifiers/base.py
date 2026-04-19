import torch.nn as nn


class BasePurifier(nn.Module):
    name = "base"
    supports_sdedit_attack = False

    def purify(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.purify(x)


class IdentityPurifier(BasePurifier):
    name = "none"

    def purify(self, x):
        return x


class PurifiedClassifier(nn.Module):
    def __init__(self, purifier, classifier):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier

    def purify(self, x):
        return self.purifier(x)

    def forward(self, x):
        return self.classifier(self.purifier(x))
