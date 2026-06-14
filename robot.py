import numpy as np

class Robot:
    def __init__(self):
        self.x = 0          
        self.y = 0          
        self.angle = 0.0    
        
        self.aruco_x = 0
        self.aruco_y = 0
        
        self.has_target = False
        self.target_x = None
        self.target_y = None

    def update_state(self, center, corners):
        if center is None or corners is None:
            return
            
        # 【新增】把 ArUco 的原始中心點存起來
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
        dir_x = dx / (marker_length + 1e-5)
        dir_y = dy / (marker_length + 1e-5)
        
        eraser_offset = marker_length * 0.8  
        
        # 覆寫 robot 的主座標為板擦中心
        self.x = int(self.aruco_x + dir_x * eraser_offset)
        self.y = int(self.aruco_y + dir_y * eraser_offset)

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

    def __str__(self):
        return f"Robot Pose -> X: {self.x}, Y: {self.y}, Angle: {self.angle:.1f}°"