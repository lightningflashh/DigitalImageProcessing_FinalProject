import numpy as np
import cv2

L = 256

def Spectrum(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = 1.0 * imgin / (L - 1)  # Copy the image to the top-left corner

    # Step 3
    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y] # Invert the sign
    
    # Step 4
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)  # Compute DFT
    # Calculate spectrum
    FR = F[:, :, 0]  # Real part
    FI = F[:, :, 1]  # Imaginary part
    S = np.sqrt(FR**2 + FI**2)  # Magnitude spectrum
    S = np.clip(S, 0, L - 1)  # Clip values to [0, 255]
    imgout = S.astype(np.uint8)
    return imgout

def CreateNotchFilter(P,Q):
    H = np.ones((P,Q,2), np.float32)
    H[:,:,1] = 0.0
    u1, v1 = 44, 58
    u2, v2 = 86, 58
    u3, v3 = 40, 119
    u4, v4 = 82, 119

    u5, v5 = P-u1, Q-v1
    u6, v6 = P-u2, Q-v2
    u7, v7 = P-u3, Q-v3
    u8, v8 = P-u4, Q-v4
    D0 = 15
    for u in range(0,P):
        for v in range(0,Q):
            Duv = np.sqrt((1.0*u-u1)**2 + (1.0*v-v1**2))
            if Duv <= D0:
                H[u,v,0] = 0.0
            Duv = np.sqrt((1.0*u-u2)**2 + (1.0*v-v2)**2)
            if Duv <= D0:
                H[u, v, 0] = 0.0
            Duv = np.sqrt((1.0*u-u3)**2 + (1.0*v-v3)**2)
            if Duv <= D0:
                H[u, v, 0] = 0.0

            Duv = np.sqrt((1.0*u-u4)**2 + (1.0*v-v4)**2)
            if Duv <= D0:
                H[u, v, 0] = 0.0

            Duv = np.sqrt((1.0*u-u5)**2 + (1.0*v-v5)**2)
            if Duv <= D0:
                H[u, v, 0] = 0.0

            Duv = np.sqrt((1.0*u-u6)**2 + (1.0*v-v6)**2)
            if Duv <= D0:
                H[u, v, 0] = 0.0

            Duv = np.sqrt((1.0*u-u7)**2 + (1.0*v-v7)**2)
            if Duv <= D0:
                H[u, v, 0] = 0.0

            Duv = np.sqrt((1.0*u-u8)**2 + (1.0*v-v8)**2)
            if Duv <= D0:
                H[u, v, 0] = 0.0
    return H

def RemoveMoireSimple(imgin):
    M,N = imgin.shape
    
    #Buoc 1 va 2: tinh PxQ va tao anh fp
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = 1.0*imgin

    #Buoc 3: nhan fp voi (-1)^(x+y)
    for x in range(0,M):
        for y in range(0,N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    #Buoc 4: tinh DFT
    F= cv2.dft(fp, flags= cv2.DFT_COMPLEX_OUTPUT)

    #buoc 5: tao bo loc H la so phuc co kich thuoc PxQ, phan ao = 0
    H = CreateNotchFilter(P, Q)

    #buoc 6: G = F*H
    G = cv2.mulSpectrums(F,H,flags = cv2.DFT_ROWS)

    #buoc 7: IDFT
    g = cv2.idft(G, flags= cv2.DFT_SCALE)

    #buoc 8: lay phan thuc co kich thuoc MxN, nhan voi (-1)^(x+y)
    gR = g[:M, :N, 0]
    for x in range(0,M):
        for y in range(0,N):
            if(x+y) % 2 == 1:
                gR[x,y] = -gR[x,y]
    gR = np.clip(gR, 0, L-1)
    imgout = gR.astype(np.uint8)
    return imgout