import cv2
import numpy as np

class VisionManager:
    def __init__(self):
        pass

    def get_aruco_ready_mask(self, frame, roi_polygon=None):
        # (保持原樣不變)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, aruco_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        if roi_polygon is not None:
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [np.array(roi_polygon, dtype=np.int32)], 255)
            aruco_mask[roi_mask == 0] = 255
        return aruco_mask

    def get_ink_clean_mask(self, frame, exclude_polygon=None, roi_polygon=None):
        """
        exclude_polygon: 傳入 ArUco 標籤的四個角點 (通常順序是 左上, 右上, 右下, 左下)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if roi_polygon is not None:
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [np.array(roi_polygon, dtype=np.int32)], 255)
            
        ink_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 7
        )
        
        # --- [優化區塊] 非等比例方向性遮罩 ---
        custom_mask_pts = None # 新增一個變數來存座標
        if exclude_polygon is not None:
            poly_pts = np.array(exclude_polygon, dtype=np.float32)
            pt_TL, pt_TR, pt_BR, pt_BL = poly_pts[0], poly_pts[1], poly_pts[2], poly_pts[3]
            center = np.mean(poly_pts, axis=0)
            
            vec_forward = ((pt_TL + pt_TR) / 2.0) - ((pt_BL + pt_BR) / 2.0)
            marker_length = np.linalg.norm(vec_forward)
            dir_forward = vec_forward / (marker_length + 1e-5) 
            
            vec_right = ((pt_TR + pt_BR) / 2.0) - ((pt_TL + pt_BL) / 2.0)
            marker_width = np.linalg.norm(vec_right)
            dir_right = vec_right / (marker_width + 1e-5) 
            
            # 你可以在這裡慢慢微調這三個參數
            extend_fwd = marker_length * 2.2   
            extend_bwd = marker_length * 1.0   
            extend_side = marker_width * 1.6   
            
            mask_FL = center + (dir_forward * extend_fwd) - (dir_right * extend_side)
            mask_FR = center + (dir_forward * extend_fwd) + (dir_right * extend_side)
            mask_BR = center - (dir_forward * extend_bwd) + (dir_right * extend_side)
            mask_BL = center - (dir_forward * extend_bwd) - (dir_right * extend_side)
            
            custom_mask_pts = np.array([mask_FL, mask_FR, mask_BR, mask_BL], dtype=np.int32)
            cv2.fillPoly(ink_thresh, [custom_mask_pts], 0)
        # ---------------------------------------------
            
        if roi_polygon is not None:
            ink_thresh = cv2.bitwise_and(ink_thresh, roi_mask)
            
        # 【修改這裡】同時回傳 黑白遮罩 與 除錯用的座標點
        return ink_thresh, custom_mask_pts