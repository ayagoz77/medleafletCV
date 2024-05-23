from ..corner_detector import CornerDetector
from ..document_detector import DocumentDetector
import cv2
import os
import csv
import numpy as np


def sort_gt(gt):
    myGtTemp = gt * gt
    sum_array = myGtTemp.sum(axis=1)
    tl_index = np.argmin(sum_array)
    tl = gt[tl_index]
    tr = gt[(tl_index + 1) % 4]
    br = gt[(tl_index + 2) % 4]
    bl = gt[(tl_index + 3) % 4]

    return np.asarray((tl, tr, br, bl))

def intersection_with_correction(a, b, img):
    img1 = np.zeros_like(img)
    cv2.fillConvexPoly(img1, np.int32(a), (255, 0, 0))

    img2 = np.zeros_like(img)
    cv2.fillConvexPoly(img2, np.int32(b), (255, 0, 0))
    min_x = min(a[0][0], a[1][0], a[2][0], a[3][0])
    min_y = min(a[0][1], a[1][1], a[2][1], a[3][1])
    max_x = max(a[0][0], a[1][0], a[2][0], a[3][0])
    max_y = max(a[0][1], a[1][1], a[2][1], a[3][1])

    dst = np.array(((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)))
    mat = cv2.getPerspectiveTransform(a.astype(np.float32), dst.astype(np.float32))
    img1 = cv2.warpPerspective(img1, mat, tuple((img.shape[0], img.shape[1])))
    img2 = cv2.warpPerspective(img2, mat, tuple((img.shape[0], img.shape[1])))

    img1 = np.sum(img1, axis=2)
    img1 = img1 / 255
    img2 = np.sum(img2, axis=2)
    img2 = img2 / 255

    inte = img1 * img2
    union = np.logical_or(img1, img2)
    iou = np.sum(inte) / np.sum(union)
    return iou


class QualCheck:
    def __init__(self) -> None:
        self.doc_det = DocumentDetector()
        self.corner_ref = CornerDetector()

    def __call__(self, path):
        l = os.listdir(path)
        ious = []
        for file in l:
            if file.endswith("jpg") or file.endswith("JPG"):
                p = os.path.join(path, file)
                corner = self.process(p, True)
                if not os.path.isfile(p + ".csv"):
                    print(p)
                    continue
                with open(p + ".csv", 'r') as csvfile:
                    spamwriter = csv.reader(csvfile, delimiter=' ',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    gt = []
                    for row in spamwriter:
                        gt.append(row)
                gt = np.array(gt).astype(np.float32)
                tgt = sort_gt(gt)
                iou = intersection_with_correction(tgt,  np.array(corner), cv2.imread(p))
                ious.append(iou)
                if iou < 0.8:
                    print(p, iou)
        return ious


    def process(self, img_path, save=False):
        img = cv2.imread(img_path)
        extracted_corners = self.doc_det(img)
        corner_address = []

        for corner in extracted_corners:
            corner_img = corner[0]
            refined_corner = np.array(self.corner_ref(corner_img))

            # Converting from local co-ordinate to global co-ordinates of the image
            refined_corner[0] += corner[1]
            refined_corner[1] += corner[2]

            # Final results
            corner_address.append(refined_corner)
        if save:
            for a in range(0, len(extracted_corners)):
                cv2.line(img, tuple(corner_address[a % 4]), tuple(corner_address[(a + 1) % 4]), (255, 0, 0), 4)
            cv2.imwrite('./results/'+img_path.split('/')[-1], img)
        return corner_address