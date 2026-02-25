import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import torch
from torchvision import transforms

def price_to_image(df):
    fig, ax = plt.subplots()
    ax.plot(df['Close'])
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).convert('L')
    plt.close(fig)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    return transform(img).unsqueeze(0)
