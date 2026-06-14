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
            motor_fwd = marker_length * 1.3    # 黃色馬達往前凸出的距離
            body_fwd  = marker_length * 0.7    # 木板前緣的距離
            body_bwd  = marker_length * 0.9    # 木板/後輪往後延伸的距離
            
            motor_side = marker_width * 0.45   # 黃色馬達的側邊寬度 (較窄)
            wheel_side = marker_width * 1.15   # 黑色輪子的側邊寬度 (較寬)
            
            # 2. 依序推算 8 個頂點 (從左前馬達尖端開始，順時針繞一圈)
            pt1 = center + (dir_forward * motor_fwd) - (dir_right * motor_side) # 左前馬達尖端
            pt2 = center + (dir_forward * motor_fwd) + (dir_right * motor_side) # 右前馬達尖端
            pt3 = center + (dir_forward * body_fwd)  + (dir_right * motor_side) # 右前馬達根部 (內縮)
            pt4 = center + (dir_forward * body_fwd)  + (dir_right * wheel_side) # 右側木板前緣 (外擴)
            pt5 = center - (dir_forward * body_bwd)  + (dir_right * wheel_side) # 右後黑輪底緣
            pt6 = center - (dir_forward * body_bwd)  - (dir_right * wheel_side) # 左後黑輪底緣
            pt7 = center + (dir_forward * body_fwd)  - (dir_right * wheel_side) # 左側木板前緣 (外擴)
            pt8 = center + (dir_forward * body_fwd)  - (dir_right * motor_side) # 左前馬達根部 (內縮)
            
            # 3. 組合合成多邊形陣列
            custom_mask_pts = np.array([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8], dtype=np.int32)
            cv2.fillPoly(ink_thresh, [custom_mask_pts], 0)
        # ---------------------------------------------
            
        if roi_polygon is not None:
            ink_thresh = cv2.bitwise_and(ink_thresh, roi_mask)
            
        # 【修改這裡】同時回傳 黑白遮罩 與 除錯用的座標點
        return ink_thresh, custom_mask_pts