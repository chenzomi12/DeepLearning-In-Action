from skimage import segmentation, util, color, feature
from skimage.feature import local_binary_pattern as LBP
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline



def _generate_segments(im_orig, scale=1.0, sigma=0.8, min_size=50):
    """
        merge Felzenswalb method mask as 4th channel of original image
    """
    scale=300
    sigma=0.90
    min_size=50
    # felzenszwalb method
    im_mask = segmentation.felzenszwalb(
        util.img_as_float(im_orig), scale=scale, sigma=sigma, min_size=min_size)

    # merge mask channel to the image as a 4th channel
    im_mask_ = np.zeros(im_orig.shape[:2])[:, :, np.newaxis]  # (424, 640, 1)
    plt.imshow(im_mask)
#     img.set_cmap('hot')
    plt.axis('off')
    im_orig = np.append(im_orig, im_mask_, axis=2)  # shape(424, 640, 4)
    im_orig[:, :, 3] = im_mask

    print(im_orig.shape)
    return im_orig

img = io.imread("person.jpg")
img2=_generate_segments(img)
img2*=255/np.max(img2)
# print(img2)
# plt.imshow(img)
plt.imshow(img2)



img = io.imread("person.jpg")
imgshow = img[:,:,2]
plt.imshow(imgshow)
imgshow = imgshow.ravel()
print(imgshow.shape)



img = io.imread("person.jpg")
img0 = img[:,:,0]
img1 = img[:,:,1]
img2 = img[:,:,2]
METHOD = 'uniform'
radius = 1
n_points = 8 * radius
imgshow_ = LBP(imgshow, n_points, radius, METHOD)
print(imgshow_)
plt.imshow(imgshow_)
