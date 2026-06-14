import cv2
import cv2.aruco as aruco
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # 收容從 vision 移居過來的 ArUco 字典設定
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        
        # 【新增這行】為新版 OpenCV 建立專屬的 ArucoDetector 偵測器物件
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def extract_robot_pose(self, aruco_mask):
        """
        [解碼車車] 從犀利黑白圖中榨取出機器人的中心點與四個角點
        """
        # 【修改這行】改用 detector 物件來呼叫 detectMarkers
        corners, ids, _ = self.detector.detectMarkers(aruco_mask)
        
        if ids is not None and 17 in ids:
            idx = np.where(ids == 17)[0][0]
            c = corners[idx][0]
            center = (int(np.mean(c[:, 0])), int(np.mean(c[:, 1])))
            return center, c 
        return None, None

    def extract_dirty_rects(self, ink_clean_mask, min_area=40):
        contours, _ = cv2.findContours(ink_clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dirty_rects = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            real_ink_pt = cnt[0][0] 
            tx, ty = int(real_ink_pt[0]), int(real_ink_pt[1])
            
            # 把原本的 4 個值變成 6 個值回傳
            dirty_rects.append((x, y, w, h, tx, ty))
            
        return dirty_rects