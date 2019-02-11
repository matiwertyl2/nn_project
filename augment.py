import torchvision.transforms.functional as TF
import torch
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

def draw_line(p1, p2):
    x = np.array([p1[0], p2[0]] ) 
    y = np.array([p1[1], p2[1]])
    plt.plot(x, y, marker = 'o')

def rotate(corners, angle, xcenter, ycenter):
    alpha = np.deg2rad(angle)
    s, c = np.sin(alpha), np.cos(alpha)
    A = np.array([[c, -s], [s, c]])
    return np.array([np.dot(A, np.array(corner) - np.array([xcenter, ycenter])) + np.array([xcenter, ycenter]) 
                        for corner in corners])


def get_corners(coords, width, height):
    xmin = coords[0] * width
    xmax = coords[1] * width
    ymin = coords[2] * height
    ymax = coords[3] * height
    return np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])

def get_new_rect(corners, width, height):
    xmin = np.min(corners[:, 0])
    xmax = np.max(corners[:, 0])
    ymin = np.min(corners[:, 1])
    ymax = np.max(corners[:, 1])
    return np.array([xmin/width, xmax/width, ymin/height, ymax/height])

    
def rotate_image(input, angle, show=False):
    img = input[0]
    data = input[1]
    rect_coords = data[1:]
    width, height = img.size

    img = TF.rotate(img, angle)
    corners = get_corners(rect_coords, width, height)
    corners = rotate(corners, -angle, width/2, height/2)
    new_rect = get_new_rect(corners, width, height)
   
    if show:
        draw = ImageDraw.Draw(img)
        draw.rectangle(((new_rect[0]*width, new_rect[2]*height), (new_rect[1]*width, new_rect[3]*height)))
        for i in range(4):
            draw_line(corners[i], corners[i-1])
        plt.imshow(img)
        plt.show()
    return (img, np.insert(new_rect, 0,  data[0]))

def translate_image(input, translate, show=False):
    img = input[0]
    plt.imshow(img)
    data = input[1]
    rect_coords = data[1:]
    width, height = img.size

    new_rect =  np.zeros(4)
    print rect_coords[0], rect_coords[0] + float(translate[0]) / width
    new_rect[0] = max(0., rect_coords[0] + float(translate[0]) / width)
    new_rect[1] = min(1., rect_coords[1] + float(translate[0]) / width)
    new_rect[2] = max(0., rect_coords[2] + float(translate[1]) / height)
    new_rect[3] = min(1., rect_coords[3] + float(translate[1]) / height)
    print new_rect
    img = TF.affine(img, 0, translate, 1, 0)

    if show:
        draw = ImageDraw.Draw(img)
        draw.rectangle(((new_rect[0]*width, new_rect[2]*height), (new_rect[1]*width, new_rect[3]*height)))
        plt.imshow(img)
        plt.show()

    return (img, np.insert(new_rect, 0,  data[0]))

def random_affine(input, angle, translate, show=False):
    alpha = np.random.uniform(-angle, angle)
    t = np.random.uniform(-translate, translate, 2)
    rotated = rotate_image(input, alpha)
    translated = translate_image(rotated, tuple(t))

    if show:
        img = translated[0]
        coords = translated[1][1:]
        width, height = img.size
        draw = ImageDraw.Draw(img)
        draw.rectangle(((coords[0]*width, coords[2]*height), (coords[1]*width, coords[3]*height)))
        plt.imshow(img)
        plt.show()

    return translated

#number = 1
#img = Image.open('data/raw/' + str(number) + '.jpg')
#rect_coords = [map(float, line.split(',')) for line in open('metadata.csv').read().splitlines()][number]
#img, data = rotate_image((img, rect_coords), 20, show=True)
#translate_image((img, rect_coords), (50., 50.), show=True)

#random_affine((img, rect_coords), 30, 30, show=True)