# Ref from https://github.com/yuxiangsun/RTFNet/blob/master/util/augmentation.py


import numpy as np
from PIL import Image


class RandomFlip():
    def __init__(self, prob=0.5):
        #super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, rgb,tdisp, label):
        if np.random.rand() < self.prob:
            rgb = rgb[:,::-1]
            tdisp = tdisp[:,::-1]
            label = label[:,::-1]
        return rgb,tdisp, label


class RandomCrop():
    def __init__(self, crop_rate=0.1, prob=1.0):
        #super(RandomCrop, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, rgb,tdisp, label):
        if np.random.rand() < self.prob:
            w, h, c = rgb.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            rgb = rgb[w1:w2, h1:h2]
            tdisp = tdisp[w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]

        return rgb,tdisp, label


class RandomCropOut():
    def __init__(self, crop_rate=0.2, prob=1.0):
        #super(RandomCropOut, self).__init__()
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self,rgb,tdisp, label):
        if np.random.rand() < self.prob:
            w, h, c = rgb.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = int(h1 + h*self.crop_rate)
            w2 = int(w1 + w*self.crop_rate)

            rgb[w1:w2, h1:h2] = 0
            tdisp[w1:w2, h1:h2] = 0
            label[w1:w2, h1:h2] = 0

        return rgb,tdisp, label


class RandomBrightness():
    def __init__(self, bright_range=0.15, prob=0.9):
        #super(RandomBrightness, self).__init__()
        self.bright_range = bright_range
        self.prob = prob

    def __call__(self, rgb,tdisp, label):
        if np.random.rand() < self.prob:
            bright_factor = np.random.uniform(1-self.bright_range, 1+self.bright_range)
            rgb = (rgb * bright_factor).astype(rgb.dtype)
            tdisp = (tdisp * bright_factor).astype(tdisp.dtype)

        return rgb,tdisp, label


class RandomNoise():
    def __init__(self, noise_range=5, prob=0.9):
        #super(RandomNoise, self).__init__()
        self.noise_range = noise_range
        self.prob = prob

    def __call__(self, rgb,tdisp, label
):
        if np.random.rand() < self.prob:
            w, h, c = rgb.shape

            noise = np.random.randint(
                -self.noise_range,
                self.noise_range,
                (w,h,c)
            )

            rgb = (rgb + noise).clip(0,255).astype(rgb.dtype)
            tdisp = (tdisp + noise).clip(0,255).astype(tdisp.dtype)

        return rgb,tdisp, label

        


