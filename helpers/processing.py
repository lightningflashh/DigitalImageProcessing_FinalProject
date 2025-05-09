import chapter3 as c3
import chapter4 as c4
import chapter5 as c5
import chapter9 as c9
import cv2

functions_use_color = ["Logarit Color", "Hist Equal Color"]

def apply_processing(imgin, option):
    switch = {
        "Negative": c3.Negative,
        "Logarit": c3.Logarit,
        "Logarit Color": c3.LogaritColor,
        "Power": c3.Power,
        "Piecewise Line": c3.PiecewiseLine,
        "Histogram": c3.Histogram,
        "Hist Equal": lambda img: cv2.equalizeHist(img),
        "Hist Equal Color": c3.HisEqualColor,
        "Local Hist": c3.LocalHist,
        "Hist Stat": c3.HistStat,
        "Smooth Box": lambda img: cv2.boxFilter(img, cv2.CV_8UC1, (21, 21)),
        "Smooth Gauss": lambda img: cv2.GaussianBlur(img, (43, 43), 7.0),
        "Hubble": c3.Hubble,
        "Median Filter": c3.MyMedianFilter,
        "Sharp": c3.Sharp,
        "Spectrum": c4.Spectrum,
        "Remove Moire": c4.RemoveMoireSimple,
        "Create Motion": c5.CreateMotion,
        "DeMotion": c5.DeMotion,
        "DeMotion Noise": lambda img: c5.DeMotion(cv2.medianBlur(img, 7)),
        "DeMotion Weiner": c5.DeMotionWeiner,
        "Erosion": c9.Erosion,
        "Dilation": c9.Dilation,
        "Boundary": c9.Boundary,
        "Contour": c9.Contour,
        "Convex Hull": c9.ConvexHull,
        "Defect Detect": c9.DefectDetect,
        "Hole Fill": c9.HoleFill,
        "Connect Component": c9.ConnectedComponent,
        "Remove Small Rice": c9.RemoveSmallRice
    }
    return switch[option](imgin)
