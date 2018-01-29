import numbers
import random
import torchvision.transforms as trans

class ScaleJittering(object):

    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, img):
        w, h = img.size
        origin = min(w, h)
        ratios = [1, 0.875, 0.75, 0.66]
        cw = int(origin * ratios[random.randint(0, 3)])
        ch = int(origin * ratios[random.randint(0, 3)])
        t = trans.Compose([
            trans.RandomCrop(size=(cw, ch)),
            trans.Resize(self.size)
        ])
        return t(img)

'''
TSN paper: 
    fix the size of input image as 256×340, and the width and height of cropped region
    are randomly selected from {256, 224, 192, 168}. Finally, these cropped regions
    will be resized to 224 × 224 for network training
'''