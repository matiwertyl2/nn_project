import numpy as np 

def iou(boxes, centers):
    # box [xmin, xmax, ymin, ymax]
    # center tak samo
    leftX = np.maximum(boxes[:, 0].reshape((-1, 1)), centers.T[0, :])
    rightX = np.minimum(boxes[:, 1].reshape((-1, 1)), centers.T[1, :])
    topY = np.maximum(boxes[:, 2].reshape((-1, 1)), centers.T[2, :])
    bottomY = np.minimum(boxes[:, 3].reshape((-1, 1)), centers.T[3, :])

    interX = np.maximum(0, rightX - leftX)
    interY = np.maximum(0, bottomY - topY)

    interS = interX * interY 

    boxesS = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])

    centersS = (centers[:, 1] - centers[:, 0]) * (centers[:, 3] - centers[:, 2]) 

    unionS = (boxesS.reshape((-1, 1)) + centersS.T) - interS

    return interS / unionS



def k_means(max_iter, idle_value, K, boxes):
    prev_error = np.inf
    centers = boxes[np.random.choice(boxes.shape[0], K, replace=False)]
    for i in range(max_iter):
        D = 1 - iou(boxes, centers)
        assigned_centers = np.argmin(D, axis=1)
        error = D[np.arange(0, boxes.shape[0]), assigned_centers].sum()
        print "iter ", i, "error ", error
        if np.abs(error - prev_error) < idle_value:
            return centers
        prev_error = error
        for k in range(K):
            assigned_boxes = np.argwhere(assigned_centers == k).flatten()
            centers[k] = boxes[assigned_boxes].mean(axis=0)
    return centers



dogs_metadata = np.array([map(float, line.split(',')) for line in open('metadata.csv').read().splitlines()])[:, 1:]

k_means(1000, 1e-6, 100, dogs_metadata)
