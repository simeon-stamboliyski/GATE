import numpy as np
from numpy.fft import fftn, ifftn

def gauss2d(L, s, ord_=0, beffect=False):
    if beffect:
        L = np.concatenate((L, np.flip(L, axis=1)), axis=1)
        L = np.concatenate((L, np.flip(L, axis=0)), axis=0)

    if isinstance(s, np.ndarray) and s.shape == L.shape:
        w = s
        FFL = fftn(L)
        FFL = FFL * w
        G = ifftn(FFL)
        if np.isrealobj(L):
            G = G.real
        if beffect:
            G = G[:L.shape[0]//2, :L.shape[1]//2]
        return G, w, G

    ny, nx = L.shape  # rows, cols

    if hasattr(s, "__len__") and len(s) > 1:
        sx, sy = s[0], s[1]
    else:
        sx = sy = s

    if ord_ is None:
        ord_ = 0
    if isinstance(ord_, int):
        ord_ = [ord_, ord_]
    if len(ord_) == 1:
        ord_ = [ord_[0], ord_[0]]
    dx, dy = ord_[0], ord_[1]

    FFL = fftn(L)

    wx = np.fft.fftfreq(nx) * 2 * np.pi * sx
    wy = np.fft.fftfreq(ny) * 2 * np.pi * sy

    if dx != 0:
        wx = (1j * wx) ** dx * np.exp(-wx ** 2)
    else:
        wx = np.exp(-wx ** 2)

    if dy != 0:
        wy = (1j * wy) ** dy * np.exp(-wy ** 2)
    else:
        wy = np.exp(-wy ** 2)

    # kron(wy, wx) so w shape matches (ny, nx)
    w = np.outer(wy, wx)

    FFL = FFL * w

    G = ifftn(FFL)
    ffx = G.copy()

    if np.isrealobj(L):
        G = G.real

    if beffect:
        G = G[:ny//2, :nx//2]

    return G, w, ffx

Image = np.array([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 9]
], dtype=float)

s = [1, 1]       # Gaussian scale in x and y directions
ord_ = [0, 0]    # No derivative, just smoothing
beffect = False  # No boundary effect padding

G, w, ffx = gauss2d(Image, s, ord_, beffect)

print("Smoothed Image G:\n", G)
print("Frequency Filter w shape:", w.shape)
print("FFT filtered output ffx (complex):\n", ffx)