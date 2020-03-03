import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def panorama(image_A, image_B):
    # Gray scale
    image_A_gray = cv2.cvtColor(image_A, cv2.COLOR_BGR2GRAY)
    image_B_gray = cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY)

    # Initialization of the sift detector
    sift = cv2.xfeatures2d.SIFT_create()

    (kp_A, desc_A) = sift.detectAndCompute(image_A_gray, None)
    (kp_B, desc_B) = sift.detectAndCompute(image_B_gray, None)

    img_A_kps = cv2.drawKeypoints(image_A,kp_A, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    save_image(img_A_kps, filename='image_A_kps.jpg')

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
    (H, status) = cv2.findHomography(pts_B, pts_A, cv2.RANSAC, 6.0)

    print(H)

    # Stick the two pictures together
    result = cv2.warpPerspective(image_B, H, (image_A.shape[1] + image_B.shape[1], image_B.shape[0]))
    save_image(result, filename='homo.jpg')
    result[0:image_A.shape[0], 0:image_A.shape[1]] = image_A

    mask = cv2.warpPerspective(image_B_gray, H, (image_A.shape[1] + image_B.shape[1], image_B.shape[0]))
    mask[:image_A.shape[0], image_A.shape[1]:] = 0
    _, mask = cv2.threshold(mask, thresh=0, maxval=255, type=cv2.THRESH_BINARY)

    mask = gaussian_filter(mask, sigma=7) 
    mask3 = cv2.multiply(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 1/2)  # 3 channel mask

    im_thresh_color = cv2.multiply(cv2.warpPerspective(image_B, H, (image_A.shape[1] + image_B.shape[1], image_B.shape[0])), mask3)

    # Show matches
    matches_image = show_matches(image_A, image_B, pts_A, pts_B)

    kps_img_A, kps_img_A_mask = show_key_points(image_A, pts_A)
    kps_img_B, kps_img_B_mask = show_key_points(image_B, pts_B, color=(255,0,0), marker=cv2.MARKER_DIAMOND)

    kps_img_mask = cv2.warpPerspective(kps_img_B_mask, H, (image_A.shape[1] + image_B.shape[1], image_B.shape[0]))
    kps_img_mask[0:image_A.shape[0], 0:image_A.shape[1]] += kps_img_A_mask

    kps_img = cv2.warpPerspective(kps_img_B, H, (image_A.shape[1] + image_B.shape[1], image_B.shape[0]))
    kps_img[0:image_A.shape[0], 0:image_A.shape[1]] += kps_img_A

    res_with_markers = result * (1- kps_img_mask/255) + kps_img

    save_image(res_with_markers)
    #add_markers(kps_img, kps_img)

def add_markers(img, markers):
    for c in zip(markers[:,:,0], markers[:,:,1], markers[:,:,2]):
        print(c)

def save_image(img, filename='out.jpg'):
    cv2.imwrite(filename, img)

def show_key_points(image, pts, marker=cv2.MARKER_CROSS, color=(0,0,255)):
    out_img = np.ones((image.shape[0],image.shape[1],3))
    mask = np.ones((image.shape[0],image.shape[1],3))
    #out_img[:,:, 3] = np.ones((image.shape[0],image.shape[1]))
    for pt in pts:
        out_img = cv2.drawMarker(out_img, (pt[0],pt[1]), color, markerType=marker, markerSize=15, thickness=2, line_type=cv2.LINE_AA)
        mask = cv2.drawMarker(mask, (pt[0],pt[1]), (255,255,255), markerType=marker, markerSize=15, thickness=2, line_type=cv2.LINE_AA)
    return out_img, mask

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
    image_A = cv2.imread("1.jpg")
    image_A = imutils.resize(image_A, width=600)
    image_B = cv2.imread("2_tilted.jpg")
    image_B = imutils.resize(image_B, width=600)

    panorama(image_A, image_B)
