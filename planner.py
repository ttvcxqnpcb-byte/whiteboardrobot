# planner.py
import math
from config.robot_settings import MAX_RESETS

class CleaningPlanner:
    def __init__(self, res_scale=1.0):
        self.current_target = None 
        self.blacklist = []           
        self.blacklist_radius = int(15 * res_scale)
        self.reset_count = 0   

    def _calculate_distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def mark_as_visited(self, x, y):
        self.blacklist.append((x, y))
        print(f"[Planner] Blacklist added: ({x}, {y})")

    def plan_next_target(self, dirty_list, robot_x, robot_y):
        if self.current_target is not None:
            return self.current_target

        # 如果沒有鎖定目標，才張開眼睛看視覺給的 dirty_list 來重新規劃
        if not dirty_list or robot_x is None or robot_y is None:
            self.current_target = None
            self.reset_count = 0
            return None

        # ── 以下為原本的尋找最近目標邏輯 ──
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
            if self.reset_count < MAX_RESETS:
                print(f"[Planner] Blacklist reset. (Retry: {self.reset_count + 1}/{MAX_RESETS})")
                self.blacklist.clear()
                self.reset_count += 1
                return None
            else:
                print("[Planner] 達重試上限，放棄頑固污漬！")
                return None

        closest_target = None
        min_distance = float('inf')

        for target in valid_targets:
            cx, cy = target['cx'], target['cy']
            # 注意：這裡傳進來的 robot_x, robot_y 已經是 main.py 算好的「板擦座標」
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
        
        target_angle_cv = math.degrees(math.atan2(dy, dx))
        
        target_abs_angle = target_angle_cv + 90
        if target_abs_angle > 180:
            target_abs_angle -= 360
            
        delta_angle = target_abs_angle - robot_angle
        if delta_angle > 180:
            delta_angle -= 360
        elif delta_angle < -180:
            delta_angle += 360

        return delta_angle, pixel_dist, target_abs_angle