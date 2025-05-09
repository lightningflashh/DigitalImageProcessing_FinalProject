import cv2
import numpy as np
L = 256

def Erosion(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    imgout = cv2.erode(imgin, w)
    return imgout

def Dilation(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgout = cv2.dilate(imgin, w)
    return imgout

def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temp = cv2.erode(imgin, w)
    imgout = imgin - temp
    return imgout

def Contour(imgin):
    # Contour chỉ dùng cho ảnh nhị phân
    # Ảnh nhị phân là ảnh có 2 màu trắng và đen
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    n = len(contour)
    for i in range(0, n-1):
        x1 = contour[i,0,0]
        y1 = contour[i,0,1]
        x2 = contour[i+1,0,0]
        y2 = contour[i+1,0,1]
        cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)
    x1 = contour[n-1,0,0]
    y1 = contour[n-1,0,1]
    x2 = contour[0,0,0]
    y2 = contour[0,0,1]
    cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return imgout

def ConvexHull(imgin):
    # Contour chỉ dùng cho ảnh nhị phân
    # Ảnh nhị phân là ảnh có 2 màu trắng và đen
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)

    # Bước 1: Tìm contour
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    # Bước 2: Tìm Convex Hull
    p = cv2.convexHull(contour, returnPoints = False)
    n = len(p)
    for i in range(0, n-1):
        vi_tri_1 = p[i][0]
        vi_tri_2 = p[i+1][0]
        x1 = contour[vi_tri_1,0,0]
        y1 = contour[vi_tri_1,0,1]
        x2 = contour[vi_tri_2,0,0]
        y2 = contour[vi_tri_2,0,1]
        cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)
    vi_tri_1 = p[n-1][0]
    vi_tri_2 = p[0][0]
    x1 = contour[vi_tri_1,0,0]
    y1 = contour[vi_tri_1,0,1]
    x2 = contour[vi_tri_2,0,0]
    y2 = contour[vi_tri_2,0,1]
    cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return imgout

def DefectDetect(imgin):
    # Contour chỉ dùng cho ảnh nhị phân
    # Ảnh nhị phân là ảnh có 2 màu trắng và đen
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)

    # Bước 1: Tìm contour
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    # Bước 2: Tìm Convex Hull
    p = cv2.convexHull(contour, returnPoints = False)
    n = len(p)
    for i in range(0, n-1):
        vi_tri_1 = p[i][0]
        vi_tri_2 = p[i+1][0]
        x1 = contour[vi_tri_1,0,0]
        y1 = contour[vi_tri_1,0,1]
        x2 = contour[vi_tri_2,0,0]
        y2 = contour[vi_tri_2,0,1]
        cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)
    vi_tri_1 = p[n-1][0]
    vi_tri_2 = p[0][0]
    x1 = contour[vi_tri_1,0,0]
    y1 = contour[vi_tri_1,0,1]
    x2 = contour[vi_tri_2,0,0]
    y2 = contour[vi_tri_2,0,1]
    cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Bước 3: Phát hiện các điểm khuyết
    defects = cv2.convexityDefects(contour, p)
    nguong_do_sau = np.max(defects[:,:,3]) // 2
    n = len(defects)
    for i in range(0, n):
        do_sau = defects[i,0,3]
        if do_sau > nguong_do_sau:
            vi_tri = defects[i,0,2]
            x = contour[vi_tri,0,0]
            y = contour[vi_tri,0,1]
            cv2.circle(imgout, (x, y), 5, (0, 255, 0), -1)
    return imgout

def HoleFill(imgin):
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    cv2.floodFill(imgout, None, (261, 148), (0,0,255))
    return imgout

def ConnectedComponent(imgin):
    nguong = 200
    _, temp = cv2.threshold(imgin, nguong, L - 1, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, 7)
    n, label = cv2.connectedComponents(temp)
    a = np.zeros(n, np.int32)
    M, N = label.shape
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            if r > 0:
                a[r] += 1
    s = 'Co %d thanh phan lien thong' % (n - 1)
    cv2.putText(temp, s, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i in range(1, n):
        s = '%2d %3d' % (i, a[i])
        cv2.putText(temp, s, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return temp

def RemoveSmallRice(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81,81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    ret, temp = cv2.threshold(temp, 100, L-1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = cv2.medianBlur(temp, 3)
    dem, label = cv2.connectedComponents(temp)
    text = 'Co %d hat gao' % (dem-1) 
    a = np.zeros(dem, int)
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] = a[r] + 1
            if r > 0:
                label[x,y] = label[x,y] + color

    for r in range(0, dem):
        print('%4d %10d' % (r, a[r]))

    max = a[1]
    rmax = 1
    for r in range(2, dem):
        if a[r] > max:
            max = a[r]
            rmax = r

    xoa = np.array([], int)
    for r in range(1, dem):
        if a[r] < 0.5*max:
            xoa = np.append(xoa, r)

    for x in range(0, M):
        for y in range(0, N):
            r = label[x,y]
            if r > 0:
                r = r - color
                if r in xoa:
                    label[x,y] = 0
    label = label.astype(np.uint8)
    cv2.putText(label,text,(1,25),cv2.FONT_HERSHEY_SIMPLEX,1.0, (255,255,255),2)
    return label
