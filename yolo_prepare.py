import numpy as np  
import math

img_height = 240
img_width = 240
S = 24 ### S musi dzielic img_width i im_height


def make_grid_cell_data(data):
    ## data format: [xmin, xmax, ymin, ymax] z przedzialow 0, 1
    x_center = (data[0]  +  data[1]) / 2 * img_width
    y_center = (data[2] + data[3]) / 2 * img_height

    box_width = (data[1] - data[0]) * img_width 
    box_height = (data[3] - data[2]) * img_height

    box_width_relative = math.sqrt(box_width / img_width) 
    box_height_relative = math.sqrt(box_height / img_height)

    grid_x_index = int(math.floor(x_center / S))
    grid_y_index = int(math.floor(y_center / S))

    grid_x = grid_x_index * S
    grid_y = grid_y_index * S

    x_center_relative = (x_center - grid_x ) / S
    y_center_relative = (y_center - grid_y ) / S

    grid_data = np.zeros((img_width / S, img_height / S, 5))
    grid_data[grid_y_index, grid_x_index] = [x_center_relative, y_center_relative, box_width_relative, box_height_relative, 1]

    return grid_data

dogs_metadata = [map(float, line.split(',')) for line in open('metadata.csv').read().splitlines()]
yolo_metadata = np.zeros((len(dogs_metadata), img_width / S, img_height / S, 5))

for i, dog_metadata in enumerate(dogs_metadata):
    yolo_metadata[i, :, :, :] = make_grid_cell_data(dog_metadata[1:])



### yolo_metadata format : dla kazdego obrazka mamy tablice S x S x 5 
### gdzie metadata[i, j] to wartosci dla cella w i-tym wierszu i j-tej kolumnie

np.save("yolo_metadata", yolo_metadata)