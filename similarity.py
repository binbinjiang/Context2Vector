import numpy as np
import scipy
from scipy.spatial.distance import cdist

class Dist2prob():
    def __init__(self, v_input):
        self.v_input = v_input
    
    def forward(self, input_center_vec, neighbor_vocs, rho=0, sigma=1):
        P = self._Similarity(
            dist=self._DistanceSquared(input_center_vec, neighbor_vocs),
            rho=rho,
            sigma_array=sigma,
            gamma=self._CalGamma(self.v_input),
            v=self.v_input)
        # print(P.type)
        # print(P)
        # print(P.shape)
        P = P/(P.sum())
        return P

    def _Similarity(self, dist, rho, sigma_array, gamma, v=100):
        # if torch.is_tensor(rho):
        #     dist_rho = (dist - rho) / sigma_array
        # else:
        #     dist_rho = dist
        dist_rho = dist
        dist_rho[dist_rho < 0] = 0
        # print(dist_rho)
        Pij = np.power(
            gamma * np.power(
                (1 + dist_rho / v),
                -1 * (v + 1) / 2
                ) * np.sqrt(2 * 3.14),
                2
            )

        # print(dist)
        # P = Pij + Pij.t() - torch.mul(Pij, Pij.t())
        # print('pij', Pij)
        return Pij
    
    def _DistanceSquared(self, x, y):
        # print(x.shape,y.shape)
        # print(np.power(cdist(x, y, p=2),2).shape)
        return np.power(cdist(x, y, p=2),2)

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out


if __name__ == "__main__":
    n, p = 5, 300
    v_input = 10

    input_center_vec = np.random.randn(n, p)
    neighbor_vocs = np.random.randn(n, p)

    MyLoss = Dist2prob(v_input=v_input)

    loss = MyLoss.forward(input_center_vec=input_center_vec, neighbor_vocs=neighbor_vocs)

    print(loss)

    input_center_vec = np.expand_dims(input_center_vec[0,:],axis=0)
    loss = MyLoss.forward(input_center_vec=input_center_vec, neighbor_vocs=neighbor_vocs)

    print(loss)