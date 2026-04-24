import torch

from attacks.base import BaseAttack


class SDEditDiffusionPGDAttack(BaseAttack):
    name = "diff_pgd"

    def __init__(self, purifier, classifier, pgd_conf, device, timestep=None):
        if not hasattr(purifier, "purify") and not hasattr(purifier, "sdedit"):
            raise ValueError("The selected purifier does not expose a purification method for diff_pgd.")
        self.purifier = purifier
        self.classifier = classifier
        self.pgd_conf = pgd_conf
        self.device = device
        self.timestep = timestep
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    def _purify(self, x):
        if hasattr(self.purifier, "sdedit"):
            return self.purifier.sdedit(x, timestep=self.timestep)
        return self.purifier.purify(x)

    def run(self, x, y):
        delta = torch.zeros_like(x)
        eps = self.pgd_conf["eps"]
        alpha = self.pgd_conf["alpha"]
        steps = self.pgd_conf["iter"]

        for _ in range(steps):
            x_diff = self._purify(x + delta).detach()
            x_diff.requires_grad_()
            with torch.enable_grad():
                loss = self.loss_fn(self.classifier(x_diff), y)
                loss.backward()
                grad_sign = x_diff.grad.data.sign()
            delta += grad_sign * alpha
            delta = torch.clamp(delta, -eps, eps)

        return torch.clamp(x + delta, 0, 1).detach()
