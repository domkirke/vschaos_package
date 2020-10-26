import numpy as np


class SerieDeformation:
    def __init__(self):
        self.full_serie = None
        self.original_cage = None
        self.deformed_cage = None

    def add_original(self, z, n_cage):
        cage = self.get_cage(z.shape[-1], n_cage)
        self.full_serie = z.copy()

        self.original_cage = z.dot(cage)
        self.deformed_cage = self.original_cage.copy()

        return self.original_cage

    def deform(self, index, cage):
        if index is not None:
            self.deformed_cage[index, :len(cage)] = cage
        else:
            self.deformed_cage[:len(cage)] = cage

        return self.mesh_deform(self.full_serie, self.original_cage, self.deformed_cage)

    def get_deformed_serie(self):
        return self.mesh_deform(self.full_serie, self.original_cage, self.deformed_cage)

    def get_cage(self, n_serie, n_cage):
        """
        Given to ints n_serie and n_cage, output a matrix "cage" that verifies:
        resampled_y = latent_series x cage

        """
        if n_serie <= n_cage:
            eye = np.eye(n_serie)
            cage = np.concatenate([eye, np.zeros([n_serie, n_cage - n_serie])], -1)

        else:
            hop = n_serie // (n_cage - 1)
            cage = np.zeros([n_serie, n_cage])
            window = np.hanning(2 * hop)
            window /= np.sum(window)

            for i in range(n_cage):
                if i and i != n_cage - 1:
                    cage[(i - 1) * hop:(i + 1) * hop, i] = window
                elif i == 0:
                    cage[:(i + 1) * hop, i] = window[hop:] * 2
                else:
                    cage[(i - 1) * hop:, i] = window[:hop] * 2

        return cage

    def mesh_deform(self, x, oldcage, newcage):
        """
        Given a latent serie x, and two different cages (original and deformed),
        apply a linear transformation to x
        """

        deform = newcage - oldcage
        if x.shape[-1] <= deform.shape[0]:
            deform = deform[:, :x.shape[-1]]
        else:
            if len(deform.shape) == 1:
                deform = np.interp(np.linspace(0, 1, x.shape[-1]), np.linspace(0, 1, deform.shape[-1]), deform)
            else:
                deform = np.asarray(
                    [np.interp(np.linspace(0, 1, x.shape[-1]), np.linspace(0, 1, deform.shape[-1]), d) for d in deform])

        y = x + deform
        return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn;

    seaborn.set()

    sd = SerieDeformation()

    x = np.zeros((4, 1024))
    cage = sd.add_original(x, 65)
    plt.plot(cage.T)
    plt.show()

    deform = np.random.randn(65)
    y = sd.deform(0, deform)

    plt.plot(y.T)
    plt.show()

