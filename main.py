import cv2
import sys
import numpy as np

# This function tries to find the 5-target pattern that looks like this
#  0  1  2
#  3     4
# The input is a list of (x,y) locations of possible targets, where each location is
# a numpy array of length 2. The output is a list of 5 targets in the proper order.
# If 5 targets in the correct configuration is not found, it returns an empty list.
def order_targets(allTargets):
    orderedTargets = []
    nTargets = len(allTargets)
    if nTargets < 5:
        return orderedTargets

    # Find 3 targets that are in a line.
    dMin = 1e9  # distance from a point to the midpt between points 1,3
    d02 = 0     # distance between points 1,3
    for i in range(0, nTargets):
        for j in range(i+1, nTargets):
            # Get the mid point between i,j.
            midPt = (allTargets[i] + allTargets[j])/2

            # Find another target that is closest to this midpoint.
            for k in range(0, nTargets):
                if k==i or k==j:
                    continue
                d = np.linalg.norm(allTargets[k] - midPt)   # distance from midpoint
                if d < dMin:
                    dMin = d        # This is the minimum found so far; save it
                    i0 = i
                    i1 = k
                    i2 = j
                    d02 = np.linalg.norm(allTargets[i0] - allTargets[i2])

    # If the best distance from the midpoint is < 30% of the distance between
	# the two other points, then we probably have a colinear set; otherwise not.
    if dMin / d02 > 0.3:
        return orderedTargets   # return an empty list

    # We have found 3 colinear targets:  p0 -- p1 -- p2.
    # Now find the one closest to p0; call it p3.
    i3 = findClosest(allTargets, i0, excluded=[i0,i1,i2])
    if i3 is None:
        return []   #  return an empty list

    # Now find the one closest to p2; call it p4.
    i4 = findClosest(allTargets, i2, excluded=[i0,i1,i2,i3])
    if i4 is None:
        return []   #  return an empty list

    # Now, check to see where p4 is with respect to p0,p1,p2.  If the
    # signed area of the triangle p0-p2-p3 is negative, then we have
    # the correct order; ie
    #   0   1   2
    #   3		4
    # Otherwise we need to switch the order; ie
    #   2	1	0
    #   4		3

    # Signed area is the determinant of the 2x2 matrix [ p3-p0, p2-p0 ].
    p30 = allTargets[i3] - allTargets[i0]
    p20 = allTargets[i2] - allTargets[i0]
    M = np.array([[p30[0], p20[0]], [p30[1], p20[1]]])
    det = np.linalg.det(M)

    # Put the targets into the output list.
    if det < 0:
        orderedTargets.append(allTargets[i0])
        orderedTargets.append(allTargets[i1])
        orderedTargets.append(allTargets[i2])
        orderedTargets.append(allTargets[i3])
        orderedTargets.append(allTargets[i4])
    else:
        orderedTargets.append(allTargets[i2])
        orderedTargets.append(allTargets[i1])
        orderedTargets.append(allTargets[i0])
        orderedTargets.append(allTargets[i4])
        orderedTargets.append(allTargets[i3])

    return orderedTargets


# In the list of points "allPoints", find the closest point to point i0, that is not
# one of the points in the excluded list.  If none found, return None.
def findClosest(allPoints, i0, excluded):
    dMin = 1e9
    iClosest = None
    for i in range(0, len(allPoints)):
        if i in excluded:
            continue
        d = np.linalg.norm(allPoints[i] - allPoints[i0])
        if d < dMin:
            dMin = d
            iClosest = i
    return iClosest

def xyz_angles_to_rot(ax, ay, az):
    sx, sy, sz = np.sin(ax), np.sin(ay), np.sin(az)
    cx, cy, cz = np.cos(ax), np.cos(ay), np.cos(az)

    Rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    Ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    Rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))

    R = Rz @ Ry @ Rx

    return R

def fProject(x, P_M, K):
    ax, ay, az, tx, ty, tz = x
    R = xyz_angles_to_rot(ax,ay,az)
    trans = np.array([[tx, ty, tz]]).T
    Mext = np.block([R, trans])
    p = K @ Mext @ P_M
    p = p / p[2,]
    return np.reshape(p[:2,], -1, 'F')

tempList = []
coordList = []
waitNum = 0;

template = cv2.imread("circle2.PNG")
overlay_img = cv2.imread("Blip-BillBoards-1.jpg")

#create blank image the size of the overlay image
blank = np.zeros(overlay_img.shape).astype(overlay_img.dtype)

#Fill the blank image with white inside the bounds of the points
cv2.fillPoly(blank, [np.array([[356,232], [1289,108], [1293,484] ,[359,525]])], color = [255, 255, 255])

pts1 = np.array([[356,232], [1289,108], [359,525], [1293,484]])
pts1HomOrder = np.array([[356,232], [1289,108], [1293,484], [359,525]])

#use the bitwise and to reverse the image so that the new billboard is surrounded by black pixels
result = cv2.bitwise_and(overlay_img, blank)

# resize image
template2 = cv2.resize(template, (int(template.shape[1] * 90 / 100), int(template.shape[0] * 90 / 100)))
template3 = cv2.resize(template, (int(template.shape[1] * 80 / 100), int(template.shape[0] * 80 / 100)))
template4 = cv2.resize(template, (int(template.shape[1] * 70 / 100), int(template.shape[0] * 70 / 100)))
template5 = cv2.resize(template, (int(template.shape[1] * 60 / 100), int(template.shape[0] * 60 / 100)))
template6 = cv2.resize(template, (int(template.shape[1] * 50 / 100), int(template.shape[0] * 50 / 100)))

video_capture = cv2.VideoCapture("fiveCCC.mp4")
got_image, bgr_image = video_capture.read()
if not got_image:
    print("cannot read video")
    sys.exit()

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter("HW3_Max_Forsythe.avi", fourcc=fourcc, fps=30.0, frameSize=(641, 481))

while True:
    got_image, bgr_image = video_capture.read()
    if not got_image:
        break

    total_list = []

    C = cv2.matchTemplate(bgr_image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(C)
    results = np.where(C >= 0.95)
    for pt in zip(*results[::-1]):
        can_enter = True
        if not total_list:
            total_list.append(pt)
        for item in total_list:
            if abs(item[0] - pt[0]) < 8:
                if abs(item[1] - pt[1]) < 8:
                    can_enter = False
        if can_enter:
            total_list.append(pt)


    C2 = cv2.matchTemplate(bgr_image, template2, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(C)
    results = np.where(C2 >= 0.85)
    for pt in zip(*results[::-1]):
        can_enter = True
        if not total_list:
            total_list.append(pt)
        for item in total_list:
            if abs(item[0] - pt[0]) < 8:
                if abs(item[1] - pt[1]) < 8:
                    can_enter = False
        if can_enter:
            total_list.append(pt)

    C3 = cv2.matchTemplate(bgr_image, template3, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(C)
    results = np.where(C3 >= 0.85)
    for pt in zip(*results[::-1]):
        can_enter = True
        if not total_list:
            total_list.append(pt)
        for item in total_list:
            if abs(item[0] - pt[0]) < 8:
                if abs(item[1] - pt[1]) < 8:
                    can_enter = False
        if can_enter:
            total_list.append(pt)

    C4 = cv2.matchTemplate(bgr_image, template4, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(C)
    results = np.where(C4 >= 0.85)
    for pt in zip(*results[::-1]):
        can_enter = True
        if not total_list:
            total_list.append(pt)
        for item in total_list:
            if abs(item[0] - pt[0]) < 8:
                if abs(item[1] - pt[1]) < 8:
                    can_enter = False
        if can_enter:
            total_list.append(pt)

    C5 = cv2.matchTemplate(bgr_image, template5, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(C)
    results = np.where(C5 >= 0.85)
    for pt in zip(*results[::-1]):
        can_enter = True
        if not total_list:
            total_list.append(pt)
        for item in total_list:
            if abs(item[0] - pt[0]) < 8:
                if abs(item[1] - pt[1]) < 8:
                    can_enter = False
        if can_enter:
            total_list.append(pt)

    C6 = cv2.matchTemplate(bgr_image, template6, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(C)
    results = np.where(C6 >= 0.7)
    for pt in zip(*results[::-1]):
        can_enter = True
        if not total_list:
            total_list.append(pt)
        for item in total_list:
            if abs(item[0] - pt[0]) < 8:
                if abs(item[1] - pt[1]) < 8:
                    can_enter = False
        if can_enter:
            total_list.append(pt)

    tuple_array = []

    for item in total_list:
        tuple_array.append(np.array([item[0], item[1]]))
    orderedList = order_targets(tuple_array)
    print(orderedList)


    P_M = np.array([[-3.7, 0, 3.7, -3.7, 3.7],
                    [-2.275, -2.275, -2.275, 2.275, 2.275],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1]], dtype="double"
                   )

    N = P_M.shape[1]
    f = 531.0
    cx = 320.0
    cy = 240.0

    K = np.array(((f, 0, cx), (0, f, cy), (0, 0, 1)))

    pts = np.array([
        [orderedList[0][0], orderedList[0][1]],
        [orderedList[1][0], orderedList[1][1]],
        [orderedList[2][0], orderedList[2][1]],
        [orderedList[3][0], orderedList[3][1]],
        [orderedList[4][0], orderedList[4][1]]], dtype="double")

    y0 = np.reshape(pts, 2 * N)

    x = np.array([1.5, -1.0, 0.0, 0, 0, 30])
    y = fProject(x, P_M, K)

    for i in range(10):
        y = fProject(x, P_M, K)

        J = np.zeros((2 * N, 6))
        e = 1e-6
        J[:, 0] = (fProject(x + [e, 0, 0, 0, 0, 0], P_M, K) - y) / e
        J[:, 1] = (fProject(x + [0, e, 0, 0, 0, 0], P_M, K) - y) / e
        J[:, 2] = (fProject(x + [0, 0, e, 0, 0, 0], P_M, K) - y) / e
        J[:, 3] = (fProject(x + [0, 0, 0, e, 0, 0], P_M, K) - y) / e
        J[:, 4] = (fProject(x + [0, 0, 0, 0, e, 0], P_M, K) - y) / e
        J[:, 5] = (fProject(x + [0, 0, 0, 0, 0, e], P_M, K) - y) / e

        dy = y - y0

        dx = np.linalg.pinv(J) @ dy

        if np.linalg.norm(dx) < 1e-6:
            break

        x = x - dx

    threeDPoints = []
    for i in range(5):
        threeDPoints.append([P_M[0][i], P_M[1][i], P_M[2][i]])

    isPoseFound, rvec, tvec = cv2.solvePnP(objectPoints=np.array(threeDPoints), imagePoints=pts, cameraMatrix=K,
                                           distCoeffs=None)
    W = np.amax(np.array(threeDPoints), axis=0) - np.amin(np.array(threeDPoints), axis=0)  # Size of model in X,Y,Z
    L = np.linalg.norm(W)
    d = L / 5
    u0 = fProject(x, np.array([0,0,0,1]), K)    # Project the origin
    uX = fProject(x, np.array([d,0,0,1]), K)    # Project a point one unit from the origin along the X axis
    uY = fProject(x, np.array([0,d,0,1]), K)    # Project a point one unit from the origin along the Y axis
    uZ = fProject(x, np.array([0,0,d,1]), K)    # Project a point one unit from the origin along the Z axis
    # Draw a line from the origin to each of these three points, in red, green, and blue


    rvecStr = "rvec: x: " + str(round(float(rvec[0]), 3)) + " y: " + str(round(float(rvec[1]), 3)) + " z: " + str(
        round(float(rvec[2]), 3))
    tvecStr = "tvec: x: " + str(round(float(tvec[0]), 3)) + " y: " + str(round(float(tvec[1]), 3)) + " z: " + str(
        round(float(tvec[2]), 3))

    cv2.putText(bgr_image, rvecStr, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(bgr_image, tvecStr, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Compute the homagraphy matrix
    pts1_ortho = np.array([[orderedList[0][0]+10, orderedList[0][1]+10],[orderedList[2][0]+10, orderedList[2][1]+10], [orderedList[3][0]+10, orderedList[3][1]+10],[orderedList[4][0]+10, orderedList[4][1]+10]])

    H1, _ = cv2.findHomography(srcPoints=pts1, dstPoints=pts1_ortho)

    # Output dimensions equal to resolution of bgr_img
    output_width = 640
    output_height = 480

    # apply perspective warp to new billboard
    bgr_ortho = cv2.warpPerspective(result, H1, (output_width, output_height))

    # Fill the are of the billboard on the bgr_img with black pixels
    cv2.fillPoly(bgr_image, pts=[pts1_ortho], color=(0, 0, 0))

    # Adding the images so that the black space is filled with pixels from the opposite images
    final = cv2.add(bgr_image, bgr_ortho)

    frames = str(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.putText(final, text=frames, org=(40,40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0,0,255))
    cv2.line(final, tuple(np.int32(u0)), tuple(np.int32(uX)), color=(0, 0, 255), thickness=3)
    cv2.line(final, tuple(np.int32(u0)), tuple(np.int32(uY)), color=(0, 255, 0), thickness=3)
    cv2.line(final, tuple(np.int32(u0)), tuple(np.int32(uZ)), color=(255, 0, 0), thickness=3)

    cv2.imshow("video", final)
    bgr_image_output = final.copy()
    videoWriter.write(bgr_image_output)
    cv2.waitKey(300)

videoWriter.release()


