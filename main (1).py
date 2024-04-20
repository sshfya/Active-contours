import argparse
import math
import numpy as np
import skimage.io
from scipy.signal import convolve
import matplotlib.pyplot as plt
import skimage.io as skio

def display_image_in_actual_size(img, dpi = 80):
    height = img.shape[0]
    width = img.shape[1]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(img, cmap='gray')

#     plt.show()
    
    return fig, ax

def display_snake(img, init_snake, result_snake):
    fig, ax = display_image_in_actual_size(img)
    ax.plot(init_snake[:, 0], init_snake[:, 1], '-r', lw=2)
    ax.plot(result_snake[:, 0], result_snake[:, 1], '-b', lw=2)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()
    
def save_mask(fname, snake, img):
    plt.ioff()
    fig, ax = display_image_in_actual_size(img)
    ax.fill(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    fig.savefig(fname, pad_inches=0, bbox_inches='tight', dpi='figure')
    plt.close(fig)
    
    mask = skio.imread(fname)
    blue = ((mask[:,:,2] == 255) & (mask[:,:,1] < 255) & (mask[:,:,0] < 255)) * 255
    skio.imsave(fname, blue.astype(np.uint8))
    plt.ion()

def conv(img, ker):
    h, w = ker.shape
    h -= 1
    w -= 1
    return convolve(np.pad(img, ((h // 2, h - h // 2), (w // 2, w - w // 2)), mode='edge'), ker, mode='valid')

def potential(img, w_line, w_edge, kappa):
    sigma = 1
    rad = math.ceil(3 * sigma)
    ker = np.zeros([rad * 2 + 1, rad * 2 + 1])
    for i in range(0, rad * 2 + 1):
        for j in range(0, rad * 2 + 1):
            ker[i][j] = math.exp((-(rad - i)**2 -(rad - j)**2) / (2 * sigma**2))
    ker /= ker.sum()
    P_line = - conv(img, ker)

    for i in range(0, rad * 2 + 1):
        for j in range(0, rad * 2 + 1):
            ker[i][j] = (i - rad) * math.exp((-(rad - i)**2 -(rad - j)**2) / (2 * sigma**2))
    ker /= ker.sum()
    P_edge = - (conv(img, ker)**2 + conv(img, ker.T)**2)
    P = w_line * P_line + w_edge * P_edge

    P_pad = np.pad(P, 1, mode='edge')
    height, width = img.shape
    Grd_x = P_pad[1:height+1, 2: width+2] - P_pad[1:height+1, 0: width]
    Grd_y = P_pad[2:height+2, 1: width+1] - P_pad[0:height, 1: width+1]
    return -kappa * Grd_x / np.abs(Grd_x).mean(), -kappa * Grd_y / np.abs(Grd_y).mean()

def bil_int(snake, img):
    size_img = img.shape[1]
    img = img.reshape(-1)
    p = np.array([img[np.floor(snake).astype(int)[:, 1] * size_img + np.floor(snake).astype(int)[:, 0]],
                  img[np.ceil(snake).astype(int)[:, 1] * size_img + np.floor(snake).astype(int)[:, 0]],
                  img[np.floor(snake).astype(int)[:, 1] * size_img + np.ceil(snake).astype(int)[:, 0]],
                  img[np.ceil(snake).astype(int)[:, 1] * size_img + np.ceil(snake).astype(int)[:, 0]]])
    return (((np.ceil(snake).astype(int)[:, 1] - snake[:, 1]) * 
            ((np.ceil(snake).astype(int)[:, 0] - snake[:, 0]) * p[0] + (snake[:, 0] - np.floor(snake).astype(int)[:, 0]) * p[2]) /
              (np.ceil(snake).astype(int)[:, 0] - np.floor(snake).astype(int)[:, 0]) +
            (snake[:, 1] - np.floor(snake).astype(int)[:, 1]) *
            ((np.ceil(snake).astype(int)[:, 0] - snake[:, 0]) * p[1] + (snake[:, 0] - np.floor(snake).astype(int)[:, 0]) * p[3]) /
              (np.ceil(snake).astype(int)[:, 0] - np.floor(snake).astype(int)[:, 0])) / 
            (np.ceil(snake).astype(int)[:, 1] - np.floor(snake).astype(int)[:, 1]))

def F(snake, G_x, G_y):
    eps = 1e-10
    snake[snake != 0] -= eps
    snake[snake == 0] += 2 * eps

    p_y = bil_int(snake, G_y)
    p_x = bil_int(snake, G_x)

    return np.hstack((p_x.reshape(-1, 1), p_y.reshape(-1, 1)))


def matrix(snake, alpha, beta, tau):
    A = np.zeros([len(snake), len(snake)])
    s = np.zeros(len(snake))
    s[0] = 2 * alpha + 6 * beta
    s[1] = - alpha - 4 * beta
    s[2] = beta
    s[len(snake) - 2] = beta
    s[len(snake) - 1] = - alpha - 4 * beta
    for i in range (0, len(snake)):
        A[i] = np.roll(s, i)
    I = np.eye(len(snake))
    return np.linalg.inv(I + tau * A)


def resample(snake):
    segment = ((snake - np.roll(snake, -1, axis=0))**2).sum(axis=1)**0.5
    h = segment[1:].mean()
    new_snake = np.zeros_like(snake)
    seg_len = 0
    new_snake[0, :] = snake[0, :]
    j = 0
    for i in range(1, len(snake) - 1):
        len_h = h * i
        while len_h > seg_len:
            seg_len = seg_len + segment[j + 1]
            j += 1
        pbefore = seg_len - len_h
        pafter = len_h - seg_len + segment[j]
        new_snake[i] = (snake[j] * pafter + snake[j - 1] * pbefore) / segment[j]
    new_snake[-1] = snake[-1]
    return new_snake

    
def give_snake(img, snake, alpha, beta, tau, w_line, w_edge, kappa):
    G_x, G_y = potential(img, w_line, w_edge, kappa) 
    matr = matrix(snake, alpha, beta, tau)
    max_itr = 4000
    pr = 1e-2
    for i in range(max_itr):
        new_snake = np.dot(matr, snake + tau * F(snake, G_x, G_y))
        new_snake[new_snake < 0] = 0
        new_snake[:, 1][new_snake[:, 1] > img.shape[0] - 1] = img.shape[0] - 1
        new_snake[:, 0][new_snake[:, 0] > img.shape[1] - 1] = img.shape[1] - 1
        if (np.abs(new_snake - snake)).mean() < pr:
            break
        snake = new_snake
        if i % 5 == 0:
            snake = resample(snake)

    return snake


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image')
    parser.add_argument('initial_snake')
    parser.add_argument('output_image')
    parser.add_argument('alpha')
    parser.add_argument('beta')
    parser.add_argument('tau')
    parser.add_argument('w_line')
    parser.add_argument('w_edge')
    parser.add_argument('kappa')
    args = parser.parse_args()
    
    img = skimage.io.imread(args.input_image)
    init = np.loadtxt(args.initial_snake)
    img = img.astype(float) / 255
    if len(img.shape) == 3:
        img = img[:, :, 0]
    res = give_snake(img, init, float(args.alpha), float(args.beta), float(args.tau), float(args.w_line), float(args.w_edge), float(args.kappa))
    save_mask(args.output_image, res, img)

    ref = skimage.io.imread(f'{args.input_image.split(".")[0]}_mask.png')
    ref = ref.astype(float) / 255
    if len(ref.shape) == 3:
        ref = ref[:, :, 0]
        
    res = skimage.io.imread(args.output_image)
    res = res.astype(float) / 255
    if len(res.shape) == 3:
        res = res[:, :, 0]
    
