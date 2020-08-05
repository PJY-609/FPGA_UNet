import numpy as np
from skimage import io

def batch_standarization(batch):
    batch = (batch - batch.mean()) / batch.std()
    return batch


def load_images(fps):
    images = []
    for fp in fps:
        image = io.imread(fp)
        images.append(image)
    
    images = np.stack(images, axis=0)
    return images


def load_masks(fps, num_classes):
    masks = []
    for fp in fps:
        mask = io.imread(fp, as_gray=True)
        mask = (mask == 255)
        w, h = mask.shape
        mask_oh = np.zeros((w, h, num_classes))
        for c in range(num_classes):
            mask_oh[:, :, c] = (mask == c)
        masks.append(mask_oh)
    
    masks = np.stack(masks, axis=0).astype(np.bool)
    return masks
