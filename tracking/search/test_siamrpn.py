import cv2
import numpy as np

from models import models
from tracker.lighttrack import Lighttrack
import torch

if __name__ == '__main__':
    model = models.LightTrackM_Supernet().cuda()
    tracker = Lighttrack()

    z = cv2.imread(r'E:\Dataset\OTB2015\Basketball\img\0001.jpg')
    x = cv2.imread(r'E:\Dataset\OTB2015\Basketball\img\0002.jpg')
    state = tracker.init(z, np.array([165, 125.5]), np.array([26, 43]), model)
    tracker.track(state, x)
