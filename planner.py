# planner.py
import math

class CleaningPlanner:
    def __init__(self):
        self.current_target = None 
        
        self.blacklist = []           
        self.blacklist_radius = 30    

    def _calculate_distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def mark_as_visited(self, x, y):
        self.blacklist.append((x, y))
        print(f"[Planner] Blacklist added: ({x}, {y})")

    def plan_next_target(self, dirty_list, robot_x, robot_y):
        if not dirty_list or robot_x is None or robot_y is None:
            self.current_target = None
            return None

        valid_targets = []
        for dirty in dirty_list:
            cx, cy = dirty['cx'], dirty['cy']
            is_blacklisted = False
            
            for bx, by in self.blacklist:
                if self._calculate_distance(cx, cy, bx, by) < self.blacklist_radius:
                    is_blacklisted = True
                    break
            
            if not is_blacklisted:
                valid_targets.append(dirty)

        if not valid_targets and len(dirty_list) > 0:
            print("[Planner] Blacklist reset.")
            self.blacklist.clear()
            return None 

        closest_target = None
        min_distance = float('inf')

        for target in valid_targets:
            cx, cy = target['cx'], target['cy']
            dist = self._calculate_distance(robot_x, robot_y, cx, cy)
            
            if dist < min_distance:
                min_distance = dist
                closest_target = (cx, cy)

        self.current_target = closest_target
        return self.current_target

    def get_relative_movement(self, robot_x, robot_y, robot_angle, target_x, target_y):
        pixel_dist = self._calculate_distance(robot_x, robot_y, target_x, target_y)

        dx = target_x - robot_x
        dy = target_y - robot_y
        
        target_angle_rad = math.atan2(dy, dx)
        target_angle_deg = math.degrees(target_angle_rad)
        
        if target_angle_deg < 0:
            target_angle_deg += 360

        delta_angle = target_angle_deg - robot_angle

        if delta_angle > 180:
            delta_angle -= 360
        elif delta_angle < -180:
            delta_angle += 360

        return delta_angle, pixel_dist