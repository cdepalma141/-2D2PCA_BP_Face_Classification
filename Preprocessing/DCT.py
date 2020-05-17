# Class for impelmenting a 2D DCT based on
# https://stackoverflow.com/questions/40104377/issiue-with-implementation-of-2d-discrete-cosine-transform-in-python

import numpy as np
from scipy.fftpack import dct, idct


class DCT2D():

    def __init__(self, img):
        self.img = img

    def dct2(self, img):
        # implement 2D DCT
        return dct(dct(img.T, norm='ortho').T, norm='ortho')

    def idct2(self, img):
        # implement 2D IDCT
        return idct(idct(img.T, norm='ortho').T, norm='ortho')

    def dct_compression(self):
        img = np.float32(self.img)
        comp = self.dct2(img)/255. # Scale to force low values to 0
        comp = np.uint8(self.idct2(comp)*255)
        return comp
