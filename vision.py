import cv2
import numpy as np

class VisionManager:
    def __init__(self, res_scale = 1.0):
        blur = int(5 * res_scale)
        self.blur_ksize = blur if blur % 2 != 0 else blur + 1
        if self.blur_ksize < 3: self.blur_ksize = 3
        
        block = int(11 * res_scale)
        self.block_size = block if block % 2 != 0 else block + 1
        if self.block_size < 3: self.block_size = 3

    def get_aruco_ready_mask(self, frame, roi_polygon=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, aruco_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        if roi_polygon is not None:
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [np.array(roi_polygon, dtype=np.int32)], 255)
            aruco_mask[roi_mask == 0] = 255
        return aruco_mask

    def get_ink_clean_mask(self, frame, robot_mask_pts=None, roi_polygon=None):
        """
        robot_mask_pts: 接收由 Robot 算好的 8 個頂點，不再負責計算車體形狀
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        
        if roi_polygon is not None:
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [np.array(roi_polygon, dtype=np.int32)], 255)
            
        ink_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.block_size, 7
        )
        
        # 直接使用傳入的遮罩座標塗黑
        if robot_mask_pts is not None:
            cv2.fillPoly(ink_thresh, [robot_mask_pts], 0)
            
        if roi_polygon is not None:
            ink_thresh = cv2.bitwise_and(ink_thresh, roi_mask)
            
        return ink_thresh