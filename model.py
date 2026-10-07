from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image

ROOT=Path(__file__).parent
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(32*7*7,64),nn.ReLU(),nn.Linear(64,10))
    def forward(self,x):return self.layers(x)

def prepare_drawing(image):
    image=image.convert('L')
    box=image.getbbox()
    if box is None:raise ValueError('Draw a digit before clicking Predict.')
    crop=image.crop(box)
    crop.thumbnail((20,20),Image.Resampling.LANCZOS)
    centered=Image.new('L',(28,28),0)
    centered.paste(crop,((28-crop.width)//2,(28-crop.height)//2))
    pixels=np.asarray(centered,dtype=np.float32)
    # Center the ink mass, as in MNIST preprocessing.
    total=pixels.sum()
    if total>0:
        ys,xs=np.indices(pixels.shape)
        dx=int(round(13.5-(pixels*xs).sum()/total));dy=int(round(13.5-(pixels*ys).sum()/total))
        shifted=Image.new('L',(28,28),0);shifted.paste(centered,(dx,dy));centered=shifted
    return np.array(centered,dtype=np.uint8)

class Recognizer:
    def __init__(self):
        torch.set_num_threads(2)
        self.model=DigitCNN()
        self.model.load_state_dict(torch.load(ROOT/'digit_cnn.pt',map_location='cpu',weights_only=True))
        self.model.eval()
    def predict(self,pixels):
        array=np.asarray(pixels)
        if array.shape!=(28,28) or not np.isfinite(array).all():raise ValueError('Expected a finite 28 by 28 image.')
        if array.min()<0 or array.max()>255:raise ValueError('Pixels must be between 0 and 255.')
        if array.max()==0:raise ValueError('Draw or load a digit first.')
        x=torch.from_numpy(array.astype(np.float32)/255).view(1,1,28,28)
        with torch.no_grad():scores=torch.softmax(self.model(x),dim=1)[0].numpy()
        return int(scores.argmax()),scores
