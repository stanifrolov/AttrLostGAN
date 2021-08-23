import numpy as np

# adapted from original MNIST Dialog code at http://cvlab.postech.ac.kr/research/attmem/

np.random.seed(123)

# possible properties
colors = ['blue', 'red', 'green', 'violet', 'brown']
bgcolors = ['white', 'cyan', 'salmon', 'yellow', 'silver']
styles = ['flat', 'stroke']
properties = ['number', 'color', 'bgcolor', 'style']


def generateGridImg(size=3):
    img = []
    for i in range(size):
        img.append([])
        for j in range(size):
            cell = {}
            cell['number'] = np.random.randint(10)
            cell['color'] = colors[np.random.randint(len(colors))]
            cell['bgcolor'] = bgcolors[np.random.randint(len(bgcolors))]
            cell['style'] = styles[np.random.randint(len(styles))]
            img[i].append(cell)
    return img


def printGridImg(img):
    arr=[]
    for i in range(len(img)):
        for k in img[0][0]:
            for j in range(len(img[i])):
                arr.append((k, img[i][j][k]))
                print( k, img[i][j][k], '\t'),
            print
        print
    return np.vstack(arr)


def initTargetMap(size=3):
    targetMap = []
    for i in range(size):
        targetMap.append([])
        for j in range(size):
            targetMap[i].append(True)
    return targetMap


def getChecklist(gridImg, targetMap):
    checklist = {}
    for k in properties:
        checklist[k] = set()
        for i in range(len(gridImg)):
            for j in range(len(gridImg[i])):
                if targetMap[i][j]:
                    checklist[k].add(gridImg[i][j][k])

    return checklist


def updateChecklist(checklist, gridImg, targetMap, reinit=False):
    for k in checklist:
        if not reinit and len(checklist[k]) == 0:
            continue

        checklist[k] = set()
        for i in range(len(gridImg)):
            for j in range(len(gridImg[i])):
                if targetMap[i][j]:
                    checklist[k].add(gridImg[i][j][k])


def noChecklist(checklist):
    for k in checklist:
        if len(checklist[k]) != 0:
            return False
    return True


def getTargets(gridImg, targetMap, prop):
    count = 0
    targetIndices = []
    for i in range(len(gridImg)):
        for j in range(len(gridImg[i])):
            if targetMap[i][j]:
                targetIndices.append((i, j))
    return targetIndices


def countTargets(gridImg, targetMap, prop, val):
    count = 0
    targetIndices = []
    for i in range(len(gridImg)):
        for j in range(len(gridImg[i])):
            if targetMap[i][j] and gridImg[i][j][prop] == val:
                count += 1
                targetIndices.append((i, j))
    return count, targetIndices


def moveTarget(gridImg, targetMap, index, direction):
    new_index = list(index)

    if direction == 0:
        new_index[0] -= 1
    elif direction == 1:
        new_index[0] += 1
    elif direction == 2:
        new_index[1] -= 1
    elif direction == 3:
        new_index[1] += 1

    targetMap[index[0]][index[1]] = False
    targetMap[new_index[0]][new_index[1]] = True

    return tuple(new_index)


def selectSubTargetMap(gridImg, targetMap, prop, val):
    count = 0
    targetIndices = []
    for i in range(len(gridImg)):
        for j in range(len(gridImg[i])):
            if targetMap[i][j] and gridImg[i][j][prop] != val:
                targetMap[i][j] = False


def reinitTargetMap(targetMap):
    for i in range(len(targetMap)):
        for j in range(len(targetMap[i])):
            targetMap[i][j] = True
