# planner.py
import math
import heapq
import cv2

class CleaningPlanner:
    def __init__(self, res_scale=1.0):
        self.res_scale = res_scale
        self.current_target = None 
        self.task_queue = []
        self.reset_count = 0
        
        # 🌟 [新增] 存放不擦拭的保留區與安全距離
        self.exclude_bboxes = []
        try:
            from config.robot_settings import ERASER_SWATH_WIDTH, SAFE_MARGIN_BASE
            self.step_size = int(ERASER_SWATH_WIDTH * res_scale)
            self.safe_margin = int(SAFE_MARGIN_BASE * res_scale)
        except ImportError:
            self.step_size = int(40 * res_scale)
            self.safe_margin = int(15 * res_scale)

    def set_exclude_bboxes(self, bboxes):
        """🌟 [新增] 從外部接收保留禁區"""
        self.exclude_bboxes = bboxes if bboxes else []

    def _calculate_distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def _is_collision(self, x, y):
        """🌟 [新增] 檢查該座標是否踩入禁區 (包含安全邊界)"""
        if not self.exclude_bboxes:
            return False
        for (ex, ey, ew, eh) in self.exclude_bboxes:
            if (ex - self.safe_margin <= x <= ex + ew + self.safe_margin) and \
               (ey - self.safe_margin <= y <= ey + eh + self.safe_margin):
                return True
        return False

    def _a_star_search(self, start, goal):
        """🌟 [核心升級] A* 尋路演算法，自動繞過禁區產生中繼點"""
        def heuristic(a, b):
            return math.hypot(b[0] - a[0], b[1] - a[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        step = self.step_size
        directions = [(0, step), (0, -step), (step, 0), (-step, 0), (step, step), (step, -step), (-step, step), (-step, -step)]
        
        max_iterations = 1000 
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            current_f, current = heapq.heappop(open_set)
            
            if heuristic(current, goal) <= step * 1.5:
                path = [goal]
                while current in came_from:
                    path.append(current)          # 🌟 先把當前節點加進去
                    current = came_from[current]  # 🌟 再往前找上一個節點
                path.reverse()
                return path
                
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if self._is_collision(neighbor[0], neighbor[1]):
                    continue
                    
                tentative_g = g_score[current] + math.hypot(dx, dy)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
        print(f"⚠️ [Planner] 找不到完美繞路路線，採直線前往 {goal}")
        return [goal]

    def generate_task_queue(self, dirty_list, start_x, start_y, current_marker_length=None, ink_mask=None):
        """拍下快照，將所有矩形網格化並驗證筆跡，最後計算 A* 最佳走訪路徑"""
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
        
        # 1. 收集所有潛在的網格點
        for dirty in dirty_list:
            x, y, w, h = dirty['x'], dirty['y'], dirty['w'], dirty['h']
            temp_points = []
            
            if w <= self.step_size and h <= self.step_size:
                # 範圍太小直接取中心
                temp_points.append((dirty['cx'], dirty['cy']))
            else:
                # 網格化降維打擊
                x_points = list(range(x + self.step_size//2, x + w, self.step_size))
                if not x_points: x_points = [dirty['cx']]

                y_points = list(range(y + self.step_size//2, y + h, self.step_size))
                if not y_points: y_points = [dirty['cy']]

                for px in x_points:
                    for py in y_points:
                        temp_points.append((px, py))
            
            # 🌟 [完美合併] 2. 統一驗證網格內是否「真的有筆跡」(ink_mask)，修復小方塊被漏驗證的 Bug
            for px, py in temp_points:
                if ink_mask is not None:
                    r = self.step_size // 2
                    h_img, w_img = ink_mask.shape
                    
                    # 【重要邊界防護】確保即使網格在畫面最邊緣，也不會造成 Array Index 錯誤閃退
                    y1, y2 = max(0, py - r), min(h_img, py + r)
                    x1, x2 = max(0, px - r), min(w_img, px + r)
                    
                    roi = ink_mask[y1:y2, x1:x2]
                    
                    # 只有該網格區塊內有實質筆跡時，才正式加入待清單
                    if roi.size > 0 and cv2.countNonZero(roi) > 0:
                        raw_points.append((px, py))
                else:
                    raw_points.append((px, py))

        if not raw_points:
            return False

        # 3. 結合貪婪最佳化與 A* 避障計算最終路徑
        curr_x, curr_y = int(start_x), int(start_y)
        while raw_points:
            best_idx = 0
            min_dist = float('inf')
            for i, pt in enumerate(raw_points):
                dist = self._calculate_distance(curr_x, curr_y, pt[0], pt[1])
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            next_target = raw_points.pop(best_idx)
            
            # 使用 A* 演算法計算包含避障繞道的路徑點
            path = self._a_star_search((curr_x, curr_y), next_target)
            self.task_queue.extend(path)
            
            curr_x, curr_y = next_target[0], next_target[1]

        print(f"[Planner] 任務快照已建立！共產出 {len(self.task_queue)} 個中繼網格點 (已驗證筆跡並包含避障繞道)。")
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