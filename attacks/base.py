class BaseAttack:
    name = "base"

    def run(self, x, y):
        raise NotImplementedError

    def __call__(self, x, y):
        return self.run(x, y)


class IdentityAttack(BaseAttack):
    name = "identity"

    def run(self, x, y):
        del y
        return x
