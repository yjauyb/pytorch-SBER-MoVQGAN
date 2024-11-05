from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from model import MOVQ

def show_images(batch, return_image=False):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    if return_image:
        return reshaped.numpy()
    img = Image.fromarray(reshaped.numpy())
    img.show()

def prepare_image(img):
    """ Transform and normalize PIL Image to tensor. """
    transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(1., 1.), ratio=(1., 1.), interpolation=transforms.InterpolationMode.BICUBIC),
        ])
    pil_image = transform(img)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    return torch.from_numpy(np.transpose(arr, [2, 0, 1]))


def main():
    # img_path = '/mnt/work/paper_reading/code_reading/MOVQGAN/test.jpeg'
    img_path = '/mnt/work/paper_reading/code_reading/MOVQGAN/test2.jpg'
    img =  prepare_image(Image.open(img_path))
    orig_img = show_images(img[None, ...], True)
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')
    model = MOVQ().to(device) 
    
    state_dict = torch.load("/mnt/work/paper_reading/code_reading/MOVQGAN/movqgan_67M.ckpt")
    model.load_state_dict(state_dict)
    with torch.no_grad():
        model.eval()
        out = model(img.to(device)[None, ...])[0]
    
    construct_img = show_images(out, True)
    mse = torch.nn.functional.mse_loss(out, img.to('cuda').unsqueeze(0))
    l1 = torch.nn.functional.l1_loss(out, img.to('cuda').unsqueeze(0))
    print('mse =', np.round(mse.item(),4), 'l1 =', np.round(l1.item(),4))
    # absolute difference between input and output
    abs_diff = show_images(torch.abs(out - img.to('cuda').unsqueeze(0)), True)
    
    combined_img = np.hstack([orig_img, construct_img, abs_diff])
    combined_img = Image.fromarray(combined_img)
    combined_img.show()

if __name__ == "__main__":
    main()
