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