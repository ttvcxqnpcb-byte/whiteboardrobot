# planner.py
import math

class CleaningPlanner:
    def __init__(self, res_scale=1.0):
        self.res_scale = res_scale
        self.current_target = None 
        self.task_queue = []
        self.reset_count = 0

        # 動態讀取設定檔，建立網格大小
        try:
            from config.robot_settings import ERASER_SWATH_WIDTH
            self.step_size = int(ERASER_SWATH_WIDTH * res_scale)
        except ImportError:
            self.step_size = int(40 * res_scale)

    def _calculate_distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def generate_task_queue(self, dirty_list, start_x, start_y, current_marker_length=None):
        """拍下快照，將所有矩形網格化並計算最佳走訪路徑"""
        self.current_target = None
        self.task_queue.clear()
        
        try:
            from config.robot_settings import ERASER_ARUCO_RATIO
            ratio = ERASER_ARUCO_RATIO
        except ImportError:
            ratio = 1.5

        if current_marker_length is not None and current_marker_length > 0:
            self.step_size = int(current_marker_length * ratio)
            print(f"[Planner] 根據標籤大小 ({current_marker_length:.1f}px) 動態設定網格間距為: {self.step_size}px")
        else:
            self.step_size = int(40 * self.res_scale)
        
        raw_points = []
        for dirty in dirty_list:
            x, y, w, h = dirty['x'], dirty['y'], dirty['w'], dirty['h']
            
            # 如果髒污範圍比板擦還小，直接取幾何中心
            if w <= self.step_size and h <= self.step_size:
                raw_points.append((dirty['cx'], dirty['cy']))
            else:
                # 網格化降維打擊：將大面積切碎成多個走訪點
                
                # 獨立計算 X 軸的網格點
                x_points = list(range(x + self.step_size//2, x + w, self.step_size))
                # 【防呆機制】如果寬度太窄（小於半個 step_size），強制填入幾何中心 X 座標
                if not x_points:
                    x_points = [dirty['cx']]

                # 獨立計算 Y 軸的網格點
                y_points = list(range(y + self.step_size//2, y + h, self.step_size))
                # 【防呆機制】如果高度太細（小於半個 step_size），強制填入幾何中心 Y 座標
                if not y_points:
                    y_points = [dirty['cy']]

                # 將獨立抓出來的 X 與 Y 點進行交乘組合
                for px in x_points:
                    for py in y_points:
                        raw_points.append((px, py))

        if not raw_points:
            return False

        # 貪婪演算法 (Greedy Nearest Neighbor) 路徑最佳化
        curr_x, curr_y = start_x, start_y
        while raw_points:
            best_idx = 0
            min_dist = float('inf')
            for i, pt in enumerate(raw_points):
                dist = self._calculate_distance(curr_x, curr_y, pt[0], pt[1])
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            # 拔出離現在最近的點，塞進工作排程
            next_target = raw_points.pop(best_idx)
            self.task_queue.append(next_target)
            curr_x, curr_y = next_target[0], next_target[1]

        print(f"[Planner] 任務快照已建立！共產出 {len(self.task_queue)} 個網格點，路徑已最佳化。")
        return True

    def get_current_target(self):
        """從佇列中依序領取任務"""
        if self.current_target is not None:
            return self.current_target
        
        if self.task_queue:
            self.current_target = self.task_queue.pop(0)
            return self.current_target
        
        return None

    def mark_target_reached(self):
        """呼叫此方法代表抵達目標，將當前目標清空，下次呼叫 get_current_target 就會拿新的"""
        self.current_target = None

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