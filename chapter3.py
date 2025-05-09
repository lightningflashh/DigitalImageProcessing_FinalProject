import cv2
import numpy as np
L = 256
def Negative(imgin):
    # M : height, N width
    M,N = imgin.shape
    # Tao ra anh imgout co kich thuoc bnag imgin va co mau den
    imgout = np.zeros((M,N),np.uint8)
    for x in range(0,M):
        for y in  range(0,N):
            r = imgin[x,y]
            s = L -1 -r
            imgout[x,y]= np.uint8(s)
    return imgout

def Logarit(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    c = (L-1)/np.log(L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r ==0:
                r=1
            s = c*np.log(1.0 + r)
            imgout[x, y] = np.uint8(s)
    return imgout

def LogaritColor(imgin):
    # C : channel: 3 cho anh mau
    M, N, C = imgin.shape
    imgout = np.zeros((M, N, C), np.uint8)
    c = (L-1)/np.log(L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y,2]
            g = imgin[x,y,1]
            b = imgin[x,y,2]
            
            if r == 0:
                r=1
            if g ==0:
                g=1
            if b == 0:
                b=1
            r = c*np.log(1.0 + r)
            g = c*np.log(1.0 + g)
            b = c*np.log(1.0 + b)
            imgout[x, y,2] = np.uint8(r)
            imgout[x, y,1] = np.uint8(g)
            imgout[x, y,0] = np.uint8(b)
    return imgout 

def Power(imgin):
    # C: Channel: 3 cho ảnh màu
    M, N = imgin.shape
    imgout = np.zeros((M,N,), np.uint8)
    gamma = 5.0
    c = np.power(L-1,1-gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            if r == 0:
                r = 1
            s = c*np.power(r, gamma)
            imgout[x,y] = np.uint8(s)
    return imgout

def PiecewiseLine(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    rmin, rmax, _, _ = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L -1
    for x in range(0,M):
        for y in range(0,N):
            r = imgin[x,y]
            #Doan 1 :
            if r < r1 :
                s = 1.0*s1/r1*r
            #Doan 2 : 
            elif r < r2:
                s = 1.0*(s2-s1)/(r2-r1)*(r-r1) + s1
            #Doan 3:
            else:
                s = 1.0*(L-1-s1)/(L-1-r1)*(r-r2) + s2
            imgout[x,y] = np.uint8(s)
    return imgout

def Histogram(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,L,3), np.uint8) + np.uint8(255)
    h = np.zeros(L, np.int32)
    for x in range(0,M):
        for y in range(0,N):
            r = imgin[x,y]
            h[r] = h[r] + 1
    p = 1.0*h/(M*N)
    scale = 3000
    for r in range(0,L):
        cv2.line(imgout, (r,M-1), (r, M-1 -np.int32(scale*p[r])), (255,0,0))
    return imgout

def HisEqual(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    h = np.zeros(L, np.int32)
    for x in range(0,M):
        for y in range(0,N):
            r = imgin[x,y]
            h[r] = h[r] + 1
    p = 1.0*h/(M*N)
    s = np.zeros(L, np.float32)
    
    for k in range(0,L):
        for j in range(0,k+1):
            s[k] = s[k] + p[j]
            
    for x in range(0,M):
        for y in range(0,N):
            r = imgin[x,y]
            imgout[x,y] = np.uint8((L-1)*s[r])
    return imgout

def HisEqualColor(imgin):
        # Ảnh BGR của openCV
    # Ảnh của pillow là RGB
    img_b = imgin[:,:,0]
    img_g = imgin[:,:,1]
    img_r = imgin[:,:,2]

    img_b = cv2.equalizeHist(img_b)
    img_g = cv2.equalizeHist(img_g)
    img_r = cv2.equalizeHist(img_r)

    imgout = imgin.copy()
    imgout[:,:,0] = img_b
    imgout[:,:,1] = img_g
    imgout[:,:,2] = img_r
    return imgout


def LocalHist(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 3
    n = 3
    a = m // 2
    b = n // 2
    for x in range(a, M - a):
        for y in range(b, N - b):
            w = imgin[x-a:x+a+1, y-b:y+b+1]
            w = cv2.equalizeHist(w)
            imgout[x, y] = w[a, b]
    return imgout
    
    
def HistStat(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 3
    n = 3
    a = m // 2
    b = n // 2
    mG, sigmaG = cv2.meanStdDev(imgin)
    C = 22.8
    k0 = 0.0
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1
    for x in range(a, M-a):
        for y in range(b, N-b):
            w = imgin[x-a:x+a+1, y-b:y+b+1]
            msxy, sigmasxy = cv2.meanStdDev(w)
            if (k0*mG <= msxy <= k1*mG)and (k2*sigmaG <= sigmasxy <= k3*sigmaG):
                imgout[x,y] = np.uint8(C*imgin[x,y])
            else:
                imgout[x,y]= imgin[x,y]
    return imgout
    
def MySmoothBox(imgin):
    m = 21
    n = 21
    w = np.zeros((m,n), np.float32) + np.float32(1.0/(m*n))
    imgout = cv2.filter2D(imgin, cv2.CV_8UC1, w)
    return imgout

def MySmoothGauss(imgin):
    m = 43
    n = 43
    sigma = 7.0
    a = m // 2
    b = n // 2
    w = np.zeros((m,n), np.float32)

    for s in range(-a, a+1):
        for t in range(-b, b+1):
            w[s+a,t+b] = np.exp(-(s*s + t*t)/(2*sigma*sigma))
    K = 0

    for s in range(0, m):
        for t in range(0,n):
            K = K + w[s,t]
    w = w/K
    imgout = cv2.filter2D(imgin, cv2.CV_8UC1, w)
    return imgout

def Hubble(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    temp = cv2.boxFilter(imgin,cv2.CV_8UC1, (15,15))
    threshold = 65
    for x in range(0,M):
        for y in range(0,N):
            r = temp[x,y]
            if r > threshold:
                s = 255
            else:
                s = 0
            imgout[x,y] = np.uint8(s)
    return imgout

def MyMedianFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 5
    n = 5
    a = m // 2
    b = n // 2
    for x in range(a, M - a):
        for y in range(b, N - b):
            w = imgin[x-a:x+a+1, y-b:y+b+1]
            # Chuyển ư thành mảng một chiều
            w = w.reshape(m*n)
            # Sắp tăng theo mảng một chiều
            w.sort()
            # Thay phần tử đang xét bằng phần tử ở giữa mảng 1 chiều
            imgout[x, y] = w[m*n//2]
    return imgout

def MyMedianFilter2(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 3
    n = 3
    a = m // 2
    b = n // 2
    w = np.zeros((m,n), np.uint8)
    for x in range(0,M):
        for y in range(0,N):
            for s in range(-a, a+1):
                for t in range(-b,b+1):
                    if x + s < 0:
                        p = x + s + M
                    elif x + s > M - 1:
                        p = x + s - M
                    else:
                        p = x + s
                    
                    if y + t < 0:
                        q = y + t + N
                    elif y + t > N - 1:
                        q = y + t - N
                    else:
                        q = y + t
                    w[s+a, t+b] = imgin[p, q]
            v = w.reshape(m*n)
            v.sort()
            imgout[x,y] = v[m*n//2]
    return imgout

def Sharp(imgin):
    # w = np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float32)
    w = np.array([[1,1,1],[1,-8,1],[1,1,1]], np.float32)
    temp = cv2.filter2D(imgin, cv2.CV_32FC1, w)
    imgout = imgin - temp
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)
    return imgout

# %%
import numpy as np
a = np.array([[3,5,2],[8,4,1],[6,9,5]], np.int32)
b = a.reshape(9)
b.sort()
print(b)

# %%
