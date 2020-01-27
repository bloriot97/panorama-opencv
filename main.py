import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt


def panorama(image_A, image_B):
    # Gray scale
    image_A_gray = cv2.cvtColor(image_A, cv2.COLOR_BGR2GRAY)
    image_B_gray = cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY)

    # Initialization of the sift detector
    sift = cv2.xfeatures2d.SIFT_create()

    (kp_A, desc_A) = sift.detectAndCompute(image_A_gray, None)
    (kp_B, desc_B) = sift.detectAndCompute(image_B_gray, None)

    # Just keep the position of the key points
    kp_A = np.float32([kp.pt for kp in kp_A])
    kp_B = np.float32([kp.pt for kp in kp_B])

    # Initialization of the brutforce martcher (the closest one is the match)
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(desc_A, desc_B, 2)
    matches = []

    for m in rawMatches:
        # Lowe's test ?
        # need more explanations 
        if len(m) == 2 and m[0].distance < m[1].distance * 0.7:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    pts_A = np.float32([kp_A[i] for (_, i) in matches])
    pts_B = np.float32([kp_B[i] for (i, _) in matches])

    # Compute the homography
    # Ransac methode, exlude the outliers 
    (H, status) = cv2.findHomography(pts_A, pts_B, cv2.RANSAC, 4.0)

    # Stick the two pictures together
    result = cv2.warpPerspective(image_A, H, (image_A.shape[1] + image_B.shape[1], image_B.shape[0]))
    result[0:image_B.shape[0], 0:image_B.shape[1]] = image_B

    # Show matches
    matches_image = show_matches(image_A, image_B, pts_A, pts_B)

    plot_image(result)

def show_matches(image_A, image_B, pts_A, pts_B):
    matches_image = np.concatenate((image_A,image_B),axis=1)
    width = image_A.shape[1]
    for match_A, match_B in zip(pts_A, pts_B):
        pt_A = (int(match_A[0]),  int(match_A[1]))
        pt_B = (int(match_B[0] + width),  int(match_B[1]))
        cv2.line(matches_image, pt_A, pt_B, (0, 255, 0), 1)
    return matches_image

def plot_image(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_B = cv2.imread("1.jpg")
    image_B = imutils.resize(image_B, width=600)
    image_A = cv2.imread("2_tilted.jpg")
    image_A = imutils.resize(image_A, width=600)

    panorama(image_A, image_B)
