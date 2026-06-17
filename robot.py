import numpy as np
import cv2
from config.robot_settings import *

class Robot:
    def __init__(self, res_scale=1.5): # 預設帶入我們系統的解析度縮放
        self.x = 0          
        self.y = 0          
        self.angle = 0.0    
        
        self.aruco_x = 0
        self.aruco_y = 0
        
        self.has_target = False
        self.target_x = None
        self.target_y = None

        self.mask_polygon = None

        self.last_rvec = None
        self.last_tvec = None

        self.proj_x = None
        self.proj_y = None
        self.proj_aruco_x = None
        self.proj_aruco_y = None
        self.proj_aruco_corners = None

        # --- 1. 建立偽造的相機內部參數 (Fake Camera Matrix) ---
        # 假設一般 Webcam 視角約 60 度，焦距 f 大約等於畫面寬度
        w = 640 * res_scale
        h = 480 * res_scale
        focal_length = w * CAMERA_FOCAL_MULTIPLIER
        self.cam_matrix = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1)) # 假設無明顯鏡頭畸變
        
        # --- 2. 定義標籤的 3D 物理角點 (Z=0 平面) ---
        s = MARKER_SIZE / 2.0
        # 順序須與 detection 出來的 corners 順序一致 (左上, 右上, 右下, 左下)
        self.obj_pts = np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0]
        ], dtype=np.float32)

    def update_state(self, center, corners):
        if center is None or corners is None:
            return
            
        self.aruco_x = int(center[0])
        self.aruco_y = int(center[1])
        
        front_x = (corners[0][0] + corners[1][0]) / 2.0
        front_y = (corners[0][1] + corners[1][1]) / 2.0
        back_x = (corners[2][0] + corners[3][0]) / 2.0
        back_y = (corners[2][1] + corners[3][1]) / 2.0
        
        dx = front_x - back_x
        dy = front_y - back_y  
        
        angle_cv = np.degrees(np.arctan2(dy, dx))
        abs_angle = angle_cv + 90
        if abs_angle > 180:
            abs_angle -= 360
        self.angle = abs_angle

        # --- 將座標基準點移到「板擦中心」 ---
        marker_length = np.hypot(dx, dy)

        self.marker_pixel_length = marker_length
        dir_x = dx / (marker_length + 1e-5)
        dir_y = dy / (marker_length + 1e-5)
        
        eraser_offset = marker_length * ERASER_OFFSET_RATIO
        self.x = int(self.aruco_x + dir_x * eraser_offset)
        self.y = int(self.aruco_y + dir_y * eraser_offset)

        # ==========================================
        #  🌟 3D 真實空間投影邏輯
        # ==========================================
        img_pts = np.array(corners, dtype=np.float32)
        success = False
        rvec = None
        tvec = None

        # 1. 每次都讓 OpenCV 吐出所有的可能解
        success_gen, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
            self.obj_pts, img_pts, self.cam_matrix, self.dist_coeffs, 
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if success_gen and len(rvecs) > 0:
            best_idx = 0
            if len(rvecs) > 1:
                # 🌟 核心進化：時間連續性 (Temporal Consistency)
                if getattr(self, 'last_rvec', None) is not None:
                    # 計算兩個解與上一幀姿態的「旋轉差異 (歐氏距離)」
                    # 物理上車子不可能瞬間翻面，所以差距最小的絕對是正確解！
                    dist0 = np.linalg.norm(rvecs[0] - self.last_rvec)
                    dist1 = np.linalg.norm(rvecs[1] - self.last_rvec)
                    
                    if dist1 < dist0:
                        best_idx = 1
                else:
                    # 如果剛開機還沒有記憶，才退回使用 Z 軸法向量當作第一眼防呆
                    z_val_0 = cv2.Rodrigues(rvecs[0])[0][2, 2]
                    z_val_1 = cv2.Rodrigues(rvecs[1])[0][2, 2]
                    if z_val_1 < z_val_0:
                        best_idx = 1
                        
            rvec = rvecs[best_idx]
            tvec = tvecs[best_idx]
            success = True

        if success:
            # 成功算出姿態後，立刻更新記憶給下一幀
            self.last_rvec = rvec
            self.last_tvec = tvec
            
            F = ROBOT_3D_FRONT
            B = -ROBOT_3D_BACK
            L = -ROBOT_3D_LEFT
            R = ROBOT_3D_RIGHT
            
            Top = -ROBOT_3D_TOP 
            Bot = ROBOT_3D_BOTTOM
            
            # 定義車體 3D 長方體的 8 個頂點
            box_3d = np.array([
                [L, F, Top], [R, F, Top], [R, B, Top], [L, B, Top],
                [L, F, Bot], [R, F, Bot], [R, B, Bot], [L, B, Bot]
            ], dtype=np.float32)
            
            # 將 8 個 3D 頂點投影回 2D 畫面
            projected_pts, _ = cv2.projectPoints(box_3d, rvec, tvec, self.cam_matrix, self.dist_coeffs)
            projected_pts = projected_pts.reshape(-1, 2).astype(np.int32)
            
            self.box_3d_pts = projected_pts

            # 取這些投影點的「凸包 (Convex Hull)」作為最終的 2D 遮罩多邊形
            hull = cv2.convexHull(projected_pts)
            self.mask_polygon = hull.reshape(-1, 2)

            # 🌟【動態欄位擴充】解算 3D 空間投影後的實際定位座標，不破壞原始 2D 觀測值
            # 1. 投影後的 ArUco 四個角點平面座標
            proj_aruco_pts, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, self.cam_matrix, self.dist_coeffs)
            self.proj_aruco_corners = proj_aruco_pts.reshape(-1, 2).astype(np.int32)
            
            # 2. 投影後的精準實體板擦中心 (底盤前緣左右角之中點)
            self.proj_x = int((self.box_3d_pts[4][0] + self.box_3d_pts[5][0]) / 2)
            self.proj_y = int((self.box_3d_pts[4][1] + self.box_3d_pts[5][1]) / 2)
            
            # 3. 投影後的精準實體車尾基準點 (底盤後緣左右角之中點)
            self.proj_aruco_x = int((self.box_3d_pts[6][0] + self.box_3d_pts[7][0]) / 2)
            self.proj_aruco_y = int((self.box_3d_pts[6][1] + self.box_3d_pts[7][1]) / 2)
        else:
            self.mask_polygon = None
            self.box_3d_pts = None
            self.last_rvec = None
            self.last_tvec = None
            
            # 姿態解算遺失時的安全性清空
            self.proj_x = None
            self.proj_y = None
            self.proj_aruco_x = None
            self.proj_aruco_y = None
            self.proj_aruco_corners = None

    def update_target(self, target_x, target_y):
        self.has_target = True
        self.target_x = target_x
        self.target_y = target_y

    def get_robot(self):
        return {
            "x" : self.x,
            "y" : self.y,
            "angle" : self.angle,
            "has_target" : self.has_target,
            "target_x" : self.target_x,
            "target_y" : self.target_y
        }
    
    def get_mask_polygon(self):
        return self.mask_polygon

    def __str__(self):
        return f"Robot Pose -> X: {self.x}, Y: {self.y}, Angle: {self.angle:.1f}°"