import numpy as np
from config.robot_settings import *

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

        self.mask_polygon = None

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
        dir_x = dx / (marker_length + 1e-5)
        dir_y = dy / (marker_length + 1e-5)
        
        eraser_offset = marker_length * ERASER_OFFSET_RATIO
        
        # 覆寫 robot 的主座標為板擦中心
        self.x = int(self.aruco_x + dir_x * eraser_offset)
        self.y = int(self.aruco_y + dir_y * eraser_offset)

        poly_pts = np.array(corners, dtype=np.float32)
        pt_TL, pt_TR, pt_BR, pt_BL = poly_pts[0], poly_pts[1], poly_pts[2], poly_pts[3]
        poly_center = np.mean(poly_pts, axis=0)
        
        vec_forward = ((pt_TL + pt_TR) / 2.0) - ((pt_BL + pt_BR) / 2.0)
        marker_length_mask = np.linalg.norm(vec_forward)
        dir_forward_mask = vec_forward / (marker_length_mask + 1e-5) 
        
        vec_right = ((pt_TR + pt_BR) / 2.0) - ((pt_TL + pt_BL) / 2.0)
        marker_width = np.linalg.norm(vec_right)
        dir_right_mask = vec_right / (marker_width + 1e-5) 
        
        motor_fwd   = marker_length_mask * MASK_MOTOR_FWD   
        wheel_start = marker_length_mask * MASK_WHEEL_START  
        body_bwd    = marker_length_mask * MASK_BODY_BWD   
        
        narrow_side = marker_width * MASK_NARROW_SIDE     
        wheel_side  = marker_width * MASK_WHEEL_SIDE   
        
        pt1 = poly_center + (dir_forward_mask * motor_fwd)   - (dir_right_mask * narrow_side)
        pt2 = poly_center + (dir_forward_mask * motor_fwd)   + (dir_right_mask * narrow_side)
        pt3 = poly_center + (dir_forward_mask * wheel_start) + (dir_right_mask * narrow_side)
        pt4 = poly_center + (dir_forward_mask * wheel_start) + (dir_right_mask * wheel_side)
        pt5 = poly_center - (dir_forward_mask * body_bwd)    + (dir_right_mask * wheel_side)
        pt6 = poly_center - (dir_forward_mask * body_bwd)    - (dir_right_mask * wheel_side)
        pt7 = poly_center + (dir_forward_mask * wheel_start) - (dir_right_mask * wheel_side)
        pt8 = poly_center + (dir_forward_mask * wheel_start) - (dir_right_mask * narrow_side)
        
        self.mask_polygon = np.array([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8], dtype=np.int32)

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