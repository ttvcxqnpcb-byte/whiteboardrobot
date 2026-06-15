import cv2
import cv2.aruco as aruco
import numpy as np

class FeatureExtractor:
    def __init__(self, res_scale=1.0):
        self.res_scale = res_scale
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def extract_robot_pose(self, aruco_mask):
        """
        [解碼車車] 從犀利黑白圖中榨取出機器人的中心點與四個角點
        """
        corners, ids, _ = self.detector.detectMarkers(aruco_mask)
        
        if ids is not None and 17 in ids:
            idx = np.where(ids == 17)[0][0]
            c = corners[idx][0]
            center = (int(np.mean(c[:, 0])), int(np.mean(c[:, 1])))
            return center, c 
        return None, None

    # 【修復】預設值改為 None，使 res_scale 的縮放能夠正確生效
    def extract_dirty_rects(self, ink_clean_mask, min_area=None):
        if min_area is None:
            min_area = int(20 * (self.res_scale ** 2))

        contours, _ = cv2.findContours(ink_clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dirty_rects = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 【保留】遵照原本的刻意設計，抓取輪廓實體邊緣上的點
            real_ink_pt = cnt[0][0]
            tx = int(real_ink_pt[0])
            ty = int(real_ink_pt[1])
            
            dirty_rects.append((x, y, w, h, tx, ty))
            
        return dirty_rects