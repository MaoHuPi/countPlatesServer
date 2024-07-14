import base64
import cv2
import numpy as np
from numpy import array as nparray
from numpy import min as npmin
from numba import jit
# https://emptygamer.itch.io/sushi-cashier

# f = open('test.text', 'r+')
# c = f.read()
# f.close()
# imgBytes = base64.b64decode(c)
# array = np.frombuffer(imgBytes, np.uint8)
# image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
# cv2.imshow('image', image)
# cv2.imwrite('image/frame.png', image)
# cv2.waitKey(-1)

@jit
def classifyEdge(rLen, cLen, rRng, cRng, edge, classify, link, lastClass):
    classCount = 0
    for r in rRng:
        for c in cRng:
            if edge[r][c] == 255:
                classify[r][c] = lastClass
            else:
                neighbor = [
                    # (classify[r+1][c+1] if edge[r+1][c+1] != 255 else lastClass) if r+1 <  rLen and c+1 <  cLen else lastClass, 
                    (classify[r+1][c  ] if edge[r+1][c  ] != 255 else lastClass) if r+1 <  rLen                 else lastClass, 
                    # (classify[r+1][c-1] if edge[r+1][c-1] != 255 else lastClass) if r+1 <  rLen and c-1 >= 0    else lastClass, 
                    (classify[r  ][c+1] if edge[r  ][c+1] != 255 else lastClass) if c+1 <  cLen                 else lastClass, 
                    (classify[r  ][c-1] if edge[r  ][c-1] != 255 else lastClass) if c-1 >= 0                    else lastClass, 
                    # (classify[r-1][c+1] if edge[r-1][c+1] != 255 else lastClass) if r-1 >= 0    and c+1 <  cLen else lastClass, 
                    (classify[r-1][c  ] if edge[r-1][c  ] != 255 else lastClass) if r-1 >= 0                    else lastClass, 
                    # (classify[r-1][c-1] if edge[r-1][c-1] != 255 else lastClass) if r-1 >= 0    and c-1 >= 0    else lastClass
                ]
                value = npmin(nparray(neighbor))
                for n in neighbor:
                    if n != lastClass:
                        if value != lastClass and n != value:
                            link[n] = value
                if value == lastClass:
                    classCount += 1
                    value = classCount
                # classify[r][c] = value
                classify[r][c] = link[value]

def countAmount(frame):
    ''' load image '''
    frame = frame[100:950]
    classify = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint16)
    image = frame

    ''' get edge '''
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image[abs(image - 51) >= 10] = 0
    image[abs(image - 51) < 10] = 255
    edge = image

    ''' classify plate '''
    [rLen, cLen] = classify.shape
    [rRng, cRng] = [np.array(list(range(rLen))), np.array(list(range(cLen)))]
    # classLen = 2**16
    classLen = 600
    lastClass = classLen-1
    link = np.arange(classLen)
    classify[:, :] = lastClass
    
    classifyEdge(rLen, cLen, rRng, cRng, edge, classify, link, lastClass)

    classify[classify == lastClass] = 0
    classes = np.unique(classify)
    for i in np.flip(np.copy(classes)):
        if link[i] != i:
            classify[classify == i] = link[i]
            classes = np.delete(classes, np.argwhere(classes == i))
    # classes = np.unique(classify)
    for i in np.copy(classes):
        if np.argwhere(classify == i).shape[0] < 20:
            classify[classify == i] = 1
            classes = np.delete(classes, np.argwhere(classes == i))
    # classes = np.unique(classify)
    # reClassCounter = 2
    # for i in classes:
    #     if i >= 2:
    #         classify[classify == i] = reClassCounter
    #         reClassCounter += 1
    # classes = np.unique(classify)
    # print(classes)

    ''' detect plate type and count the amount '''
    sampleNum = 5
    red = [34, 33, 136]
    white = [227, 224, 219]
    black = [54, 54, 60]
    def colorDistance(c1, c2):
        return np.sqrt(np.sum(np.square(c1 - c2)))
    amount = 0
    for i in classes:
        # if i >= 2:
        if i > 1:
            average = [0, 0, 0]
            points = np.argwhere(classify == i)
            sampleIndex = np.random.randint(0, points.shape[0], sampleNum)
            for s in sampleIndex:
                average += frame[points[s][0], points[s][1]]
            average = average / sampleNum
            distances = [
                colorDistance(average, red), 
                colorDistance(average, white), 
                colorDistance(average, black)
            ]
            minDistance = np.min(distances)
            plateType = 'red' if distances[0] == minDistance else 'white' if distances[1] == minDistance else 'black'
            # print(plateType, 'rgb(' + ', '.join(average.astype(np.uint8).astype(np.str)) + ')')
            amount += {'red': 1, 'white': 3, 'black': 5}[plateType]
    # for i in classes:
    #     image = np.zeros(classify.shape, dtype=np.uint8)
    #     image[classify == i] = 255
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #     image[image == 255] = frame[image == 255]
    #     cv2.imshow('image', image)
    #     cv2.waitKey(-1)
    return amount

frame = cv2.imread('image/frame_8064.png')
amount = countAmount(frame)
print(amount)