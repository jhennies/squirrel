
import numpy as np


def register_with_sift2(
        fixed_image,
        moving_image,
        verbose=False
):

    import cv2

    sift = cv2.SIFT_create(nOctaveLayers=3, sigma=1.6, contrastThreshold=0.09)

    kp_img, des_img = sift.detectAndCompute(moving_image, None)
    kp_ref, des_ref = sift.detectAndCompute(fixed_image, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_img, des_ref)

    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(moving_image, kp_img, fixed_image, kp_ref, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Compute the offset by analyzing matched keypoints
    if len(img_matches) > 0:
        # Extract location of good matches
        points1 = np.float32([kp_img[m.queryIdx].pt for m in img_matches])
        points2 = np.float32([kp_ref[m.trainIdx].pt for m in img_matches])

        # Find translation vector using mean of the differences
        offset = np.median(points2 - points1, axis=0)
        if verbose:
            print(f"Offset between images: {offset}")
    else:
        print("Not enough matches found.")
        offset = [0., 0.]

    return offset


def register_with_sift(
        fixed_image,
        moving_image,
        transform='translation',
        verbose=False
):

    import cv2 as cv
    sift = cv.SIFT_create(nOctaveLayers=3, sigma=1.6, contrastThreshold=0.09)

    mask = (fixed_image > np.quantile(fixed_image, 0.2)).astype('uint8')
    # if auto_mask > 0:
    #     mask = discErosion(mask.astype('uint8'), auto_mask)

    kp_img, des_img = sift.detectAndCompute(moving_image, mask)
    kp_ref, des_ref = sift.detectAndCompute(fixed_image, mask)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_img, des_ref, k=2)

    # bf = cv.BFMatcher(cv.NORM_L2)
    # matches = bf.knnMatch(des_img, des_ref, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.92 * n.distance:
            good.append(m)

    # good = sorted(good, key=lambda x: x.distance)
    print(f'len(good) = {len(good)}')
    # if len(good) > 20:
    #     print(f'len(good) > 100: {len(good)}')
    # good = good[:20]

    # final = good

    min_match_count = 10
    if len(good) >= min_match_count:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, matches_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5)
        matches_mask = matches_mask.ravel().tolist()

        final = np.array(good)[np.array(matches_mask) > 0]

    else:
        print("Not enough matches are found - {}/{}".format(len(good), min_match_count))
        matchesMask = None

        final = None
    print(f'len(final) = {len(final)}')

    if transform == 'translation':

        offset = np.median([
            [kp_img[x.queryIdx].pt[0] - kp_ref[x.trainIdx].pt[0],
             kp_img[x.queryIdx].pt[1] - kp_ref[x.trainIdx].pt[1]]
            for x in final
        ], axis=0)

        from squirrel.library.transformation import setup_translation_matrix
        return setup_translation_matrix(offset, ndim=2)

    if transform == 'affine':

        kp_img = np.array([[kp_img[x.queryIdx].pt[0], kp_img[x.queryIdx].pt[1]] for x in final])
        kp_ref = np.array([[kp_ref[x.trainIdx].pt[0], kp_ref[x.trainIdx].pt[1], 1.] for x in final])

        if verbose:
            print(f'kp_img.shape = {kp_img.shape}')

        affine_transform, residues, rank, s = np.linalg.lstsq(kp_img, kp_ref)

        if verbose:
            print(f'affine_transform = {affine_transform}')

        return affine_transform
