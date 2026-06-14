import numpy as np

class Robot:
    def __init__(self):
        self.x = 0          
        self.y = 0          
        self.angle = 0.0    
        
        self.has_target = False
        self.target_x = None
        self.target_y = None

    def update_state(self, center, corners):
        
        if center is None or corners is None:
            return
            
        self.x = center[0]
        self.y = center[1]
        
        pt_top_left = corners[0]
        pt_top_right = corners[1]
        
        dx = pt_top_right[0] - pt_top_left[0]
        dy = pt_top_right[1] - pt_top_left[1]  
        
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        self.angle = angle_deg if angle_deg >= 0 else (angle_deg + 360)

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