from detectors import CornerDetector, DocumentDetector
from utils import QualityEnhancer, TextExtractor
from block_extractor import TextBlockIdentifier, TextBBoxCluster
import cv2
import pytesseract
from groq import Groq
import numpy as np
import os

def divide_text_blocks(text_blocks, n):
    """
    Divides text blocks into n parts and returns the y-coordinates
    of the dividing points.
    """

    indices = [int((len(text_blocks) -1) * (i / n)) for i in range(n + 1)]
    print(indices, len(text_blocks))
    sorted_y = sorted(np.array(text_blocks)[:, 1])
    
    l_y = [sorted_y[i] for i in indices]
    
    return l_y


class TextHandler:
    def __init__(
        self,
        doc_model_path: str = None,
        corner_model_path: str = None,
        quality_enhancer_cfg: dict = None,
        text_extractor_cfg: dict = None,
        block_ident_cfg: dict = None,
        text_cluster_cfg: dict = None,
        debug: bool = False,
    ) -> None:

        self.doc_detector = (
            DocumentDetector()
            if doc_model_path is None
            else DocumentDetector(doc_model_path)
        )
        self.corner_ref = (
            CornerDetector()
            if corner_model_path is None
            else CornerDetector(corner_model_path)
        )
        self.qual_enhancer = (
            QualityEnhancer()
            if quality_enhancer_cfg is None
            else QualityEnhancer(quality_enhancer_cfg)
        )
        self.text_block_extractor = (
            TextExtractor()
            if text_extractor_cfg is None
            else TextExtractor(text_extractor_cfg)
        )
        # self.block_identifier = (
        #     TextBlockIdentifier()
        #     if block_ident_cfg is None
        #     else TextExtractor(block_ident_cfg)
        # )
        self.text_clusterer_by_x = (
            TextBBoxCluster()
            if text_cluster_cfg is None
            else TextExtractor(text_cluster_cfg)
        )
        self.debug = debug
        if self.debug:
            for filename in os.listdir('debug_imgs'):
                file_path = os.path.join('debug_imgs', filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)


    def __call__(self, original_image: np.ndarray) -> str:
        img = original_image.copy()
        extracted_corners = self.doc_detector(img)
        corner_address = []

        for corner in extracted_corners:
            corner_img = corner[0]
            refined_corner = np.array(self.corner_ref(corner_img))

            # Converting from local co-ordinate to global co-ordinates of the image
            refined_corner[0] += corner[1]
            refined_corner[1] += corner[2]

            # Final results
            corner_address.append(refined_corner)

        pts_src = np.array(corner_address, dtype=np.float32)
        tl, tr, br, bl = pts_src
        tl_x = min([tl[0], tr[0], br[0], bl[0]])
        tl_y = min([tl[1], tr[1], br[1], bl[1]])
        br_x = max([tl[0], tr[0], br[0], bl[0]])
        br_y = max([tl[1], tr[1], br[1], bl[1]])
        width, height = int(br_x - tl_x), int(
            br_y - tl_y
        )  

        pts_dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

        result = cv2.warpPerspective(img, matrix, (width, height))
        if self.debug:
            cv2.imwrite("debug_imgs/doc_crop.png", result)

            # for a in range(0, len(extracted_corners)):
            #     cv2.line(
            #         img,
            #         tuple(corner_address[a % 4]),
            #         tuple(corner_address[(a + 1) % 4]),
            #         (255, 0, 0),
            #         4,
            #     )
            # cv2.imwrite('', img)

        enhanced_img = self.qual_enhancer(result)
        if self.debug:
            cv2.imwrite("debug_imgs/enhanced.png", enhanced_img)
        text_blocks = self.text_block_extractor(enhanced_img)
        if self.debug:
            img = enhanced_img.copy()
            for cnt in text_blocks:
                x, y, w, h = cnt
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite('debug_imgs/text_ext.png', img)
        clusters_x = self.text_clusterer_by_x(text_blocks)
        print(height)
        height = enhanced_img.shape[0]
        n_blocks = 6
        l_y = divide_text_blocks(text_blocks, n_blocks)
        print(l_y)
        clusters_x.append(width-1)
            
        text = ""
        img = enhanced_img.copy()
        print(enhanced_img.shape, width, height, clusters_x)
        c = 0
        for x in range(1, len(clusters_x)):
            x1, x2 = clusters_x[x-1], clusters_x[x]
            if x1 - 5 > 0:
                x1 -= 5
            for i in range(1, n_blocks+1):
                y1, y2 = l_y[i-1], l_y[i]
                if y1 - 2 > 0:
                    y1 -= 2
                print(x1, y1, x2, y2)
                if self.debug:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cropped = enhanced_img[y1 : y2, x1 : x2]
                cv2.imwrite('debug_imgs/'+str(c)+'.png', cropped)
                c += 1
                text += pytesseract.image_to_string(
                    cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), lang="rus"
                )
        if self.debug:
            cv2.imwrite("debug_imgs/text_blocks.png", img)
        return text
