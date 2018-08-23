# -*- coding: utf-8 -*-
import skimage.io
import skimage.transform
import skimage.util
from skimage import segmentation, util, color, feature
import numpy as np
import numpy


def _generate_segments(im_orig, scale, sigma, min_size):
    """
        merge Felzenswalb method mask as 4th channel of original image
    """

    # felzenszwalb method
    im_mask = segmentation.felzenszwalb(
        util.img_as_float(im_orig), scale=scale, sigma=sigma, min_size=min_size)

    # merge mask channel to the image as a 4th channel
    im_mask_ = np.zeros(im_orig.shape[:2])[:, :, np.newaxis]  # (424, 640, 1)
    im_orig = np.append(im_orig, im_mask_, axis=2)  # shape(424, 640, 4)
    im_orig[:, :, 3] = im_mask

    return im_orig


def _sim_colour(r1, r2):
    """
        calculate the sum of histogram intersection of colour
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    """
        calculate the sum of histogram intersection of texture
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    """
        calculate the size similarity over the image
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    sim_colour = _sim_colour(r1, r2)
    sim_texture = _sim_texture(r1, r2)
    sim_size = _sim_size(r1, r2, imsize)
    sim_fill = _sim_fill(r1, r2, imsize)

    return (sim_colour + sim_texture + sim_size + sim_fill)


def _calc_texture_gradient(img):
    """
        calculate texture gradient for entire image

        The original Selective Search algorithm proposed Gaussian derivative
        for 8 orientations, but we use LBP instead.
    """

    im_texture = np.zeros(img.shape[:3])  # (424, 640, 4)

    for colour_channel in (0, 1, 2):
        im_texture[:, :, colour_channel] = feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)

    return im_texture


def _calc_colour_hist(img):
    """
        calculate colour histogram for each region

        the size of output histogram will be BINS * COLOUR_CHANNELS(3)

        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]

        extract HSV
    """

    BINS = 25
    hist = numpy.array([])

    for colour_channel in (0, 1, 2):
        c = img[:, colour_channel]

        # calculate histogram for each colour and join to the result
        hist = numpy.concatenate(
            [hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1 normalize
    hist = hist / len(img)
    return hist


def _calc_texture_hist(img):
    """
        calculate texture histogram for each region

        calculate the histogram of gradient for each colours
        the size of output histogram will be
            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10
    hist = np.array([])

    # extracting colour channel
    for colour_channel in (0, 1, 2):
        fd = img[:, colour_channel]

        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = np.concatenate(
            [hist] + [np.histogram(fd, BINS, (0.0, 1.0))[0]])

    # L1 Normalize
    hist = hist / len(img)
    return hist


def _extract_regions(img):
    """
        segment smallest regions by the algorithm of Felzenswalb and Huttenlocher
    """
    R = {}

    # 记录每一个由FFelzenswalb算法分割出来的区域
    # pass 1: count pixel positions accroding segmentation image
    for y, i in enumerate(img):  # iter rows
        for x, (r, g, b, l) in enumerate(i):  # iter cols
            # initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}
            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    # pass 2: calculate texture gradient and hsv
    tex_grad = _calc_texture_gradient(img)
    hsv = color.rgb2hsv(img[:, :, :3])

    # pass 3: calculate colour histogram of each region
    for k, v in list(R.items()):
        masked = [img[:, :, 3] == k]  # true / false

        # colour histogram
        # mask the pixels in color
        masked_pixels = hsv[:, :, :][masked]   # shape( color_size, 3)
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)

        # texture histogram
        # mask the pixels in texture
        masked_texture = tex_grad[:, :][masked]  # shape( color_size, 3)
        R[k]["hist_t"] = _calc_texture_hist(masked_texture)

    return R


def _extract_neighbours(regions):
    """
        regions: dict
    """

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = list(regions.items())
    neighbours = []

    for idx, a in enumerate(R[:-1]):
        for b in R[idx + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours


def _merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def selective_search(im_orig, scale=1.0, sigma=0.8, min_size=50):
    '''Selective Search

    Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, right, bottom),
                    'labels': [...]
                },
                ...
            ]
    '''
    assert im_orig.shape[2] == 3, "输入应该是彩色图片"

    # load image and get smallest regions
    # region label is stored in the 4th value of each pixel [r,g,b,(region)]
    img = _generate_segments(im_orig, scale, sigma, min_size)

    if img is None:
        print("ERROR in felzenszwalb")
        return None, {}

    R = _extract_regions(img)

    imsize = img.shape[0] * img.shape[1]

    # extract neighbouring information
    neighbours = _extract_neighbours(R)

    # calculate initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    print(list(S.items()))
    # exit()
    # hierarchal search
    while S != {}:

        # get highest similarity
        # i, j = sorted(list(S.items()), cmp=lambda a, b: cmp(a[1], b[1]))[-1][0]
        highest = sorted(list(S.items()), key=lambda a: a[1])[-1]
        i, j = highest[0]

        # merge corresponding regions
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        # mark similarities for regions to be removed
        key_to_delete = []
        for k, v in S.items():
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]

        # calculate similarity set with the new region
        for k in filter(lambda a: a != (i, j), key_to_delete):
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    regions = []
    for k, r in list(R.items()):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions
