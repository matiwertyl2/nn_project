import numpy as np 

def iou(boxes, centers):
    # box [width, height]

    interX = np.minimum(boxes[:, 0].reshape((-1, 1)), centers.T[0, :])
    interY = np.minimum(boxes[:, 1].reshape((-1, 1)), centers.T[1, :])

    interS = interX * interY 

    boxesS = (boxes[:, 1] * boxes[:, 0])

    centersS = (centers[:, 1] * centers[:, 0])

    unionS = (boxesS.reshape((-1, 1)) + centersS.T) - interS

    print np.min(unionS)
    print np.min(boxesS), np.argmin(boxesS)
    return interS / (unionS + 0.00000001)



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
            if len(assigned_boxes) > 0:
                centers[k] = boxes[assigned_boxes].mean(axis=0)
    return centers



dogs_metadata = np.array([map(float, line.split(',')) for line in open('metadata.csv').read().splitlines()])[:, 1:]
boxes = np.zeros((dogs_metadata.shape[0], 2))
boxes[:, 0] = dogs_metadata[:, 1] - dogs_metadata[:, 0]
boxes[:, 1] = dogs_metadata[:, 3] - dogs_metadata[:, 2]

print k_means(1000, 1e-6, 5, boxes)
