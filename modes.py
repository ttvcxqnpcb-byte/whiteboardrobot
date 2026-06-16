# modes.py
import cv2
import time
from abc import ABC, abstractmethod
import numpy as np
from config.robot_settings import *


# ==========================================
#  1. 模式基底類別 (Strategy Interface)
# ==========================================
class BaseMode(ABC):
    def __init__(self, shared_context):
        self.ctx = shared_context

    @abstractmethod
    def activate(self): pass

    @abstractmethod
    def process_frame(self, frame): pass

    @abstractmethod
    def handle_key(self, key): pass


# ==========================================
#  2. Mode 0: 完整控制模式 (Full System Mode)
# ==========================================
class FullControlMode(BaseMode):
    def __init__(self, shared_context):
        super().__init__(shared_context)
        # 🌟 狀態機：IDLE (待機), CLEANING (擦拭中), RETURNING (回家中), VERIFYING (二階段複檢)
        self.state = "IDLE" 
        
        self.last_cmd = None
        self.last_send_time = 0
        self.eraser_on = False

        self.RETRY_COOLDOWN = 1.5
        self.MAX_RETRIES = 50         
        self.retry_count = 0

        self.lost_frames_count = 0
        self.MAX_LOST_FRAMES = MAX_LOST_FRAMES
        
        self.ack_start_pos = None
        self.ack_start_angle = None

        self.home_pos = None
        self.verify_timer = 0
        self.cached_roi_array = np.array(self.ctx['roi_polygon'], dtype=np.int32)
        
        self.target_start_time = 0
        self.current_target_cache = None

        self.cmd_lock_expiry = 0
        self.verify_timer = 0
        self.is_aligning_home = False

    def activate(self):
        print("\n📡 [Mode 0] 切換至全自動快照任務模式 (藍牙自動連線已啟動)")
        if self.ctx.get('bt'):
            self.ctx['bt'].enable_auto_connect()

    def process_frame(self, frame):
        aruco_mask = self.ctx['vision'].get_aruco_ready_mask(frame, roi_polygon=self.ctx['roi_polygon'])
        robot_center, robot_corners = self.ctx['extractor'].extract_robot_pose(aruco_mask)
        
        robot_mask_pts = None
        if robot_center is not None:
            self.ctx['robot'].update_state(robot_center, robot_corners)
            robot_mask_pts = self.ctx['robot'].get_mask_polygon()

        # 這裡依然每幀辨識，但不再即時干擾清潔規劃
        ink_mask = self.ctx['vision'].get_ink_clean_mask(frame, robot_mask_pts=robot_mask_pts, roi_polygon=self.ctx['roi_polygon'])
        dirty_rects = self.ctx['extractor'].extract_dirty_rects(ink_mask)
        self.ctx['whiteboard'].update_dirty_matrix(dirty_rects)

        if self.state != "IDLE":
            if robot_center is None:
                self.lost_frames_count += 1
                if self.lost_frames_count >= self.MAX_LOST_FRAMES:
                    if self.ctx.get('bt'): 
                        rescue_cmd = "B" if self.last_cmd == "F" else "S"
                        if self.last_cmd != rescue_cmd:
                            self.ctx['bt'].send_new_action(rescue_cmd)
                            self.last_cmd = rescue_cmd
                            self.last_send_time = time.time()
                            self.retry_count = 0
                    self.lost_frames_count = self.MAX_LOST_FRAMES 
            else:
                self.lost_frames_count = 0
                current_scale = self.ctx.get('res_scale', 1.0)
                
                # 視覺代償邏輯
                if self.ctx.get('bt') and not self.ctx['bt'].is_cmd_acked and self.ack_start_pos is not None:
                    dist = np.hypot(self.ctx['robot'].x - self.ack_start_pos[0], self.ctx['robot'].y - self.ack_start_pos[1])
                    delta_ang = abs(self.ctx['robot'].angle - self.ack_start_angle)
                    if delta_ang > 180: delta_ang = 360 - delta_ang
                    if dist > (VISUAL_COMP_DIST_BASE * current_scale) or delta_ang > VISUAL_COMP_ANGLE:
                        self.ctx['bt'].is_cmd_acked = True
                        self.ack_start_pos = None

                # 🌟 狀態機核心邏輯
                target = None
                if self.state == "VERIFYING":
                    # 睜開眼睛等待視線穩定 2 秒
                    if time.time() - self.verify_timer > 2.0:
                        dirty_list = self.ctx['whiteboard'].get_dirty_list()
                        if self.ctx['planner'].reset_count < MAX_RESETS:
                            has_tasks = self.ctx['planner'].generate_task_queue(dirty_list, self.ctx['robot'].x, self.ctx['robot'].y)
                            if has_tasks:
                                self.ctx['planner'].reset_count += 1
                                print(f"👀 [二階段複檢] 發現殘留髒污！啟動第 {self.ctx['planner'].reset_count} 波補刀攻堅！")
                                self.state = "CLEANING"
                            else:
                                print("🎉 [二階段複檢] 完美通過！白板已經完全乾淨，任務結束！")
                                self.state = "IDLE"
                        else:
                            print("🚨 達到最大重試次數，放棄頑固污漬，強制下班！")
                            self.state = "IDLE"
                            
                elif self.state == "RETURNING":
                    target = self.home_pos
                    
                elif self.state == "CLEANING":
                    target = self.ctx['planner'].get_current_target()
                    if target is None:
                        print("🏁 清單走訪完畢！準備退回基地進行複檢...")
                        self.state = "RETURNING"
                        target = self.home_pos

                # 🌟 目標攻堅看門狗 (Watchdog)
                if self.state == "CLEANING" and target is not None:
                    if target != self.current_target_cache:
                        self.current_target_cache = target
                        self.target_start_time = time.time()
                    elif time.time() - self.target_start_time > WATCHDOG_TIMEOUT:
                        print(f"🚨 [看門狗] 攻堅目標 {target} 超時 ({WATCHDOG_TIMEOUT}s)！直接放棄該網格點，倒車脫困！")
                        self.ctx['planner'].mark_target_reached()
                        self.current_target_cache = None
                        target = None
                        if self.ctx.get('bt'):
                            self.ctx['bt'].send_new_action("B")
                            self.last_cmd = "B"
                            self.last_send_time = time.time()
                            self.retry_count = 0
                        return # 提早結束這幀，先退後再說
                elif target is None:
                    self.current_target_cache = None

                # 確認指令 ACK
                if self.ctx.get('bt'):
                    if self.ctx['bt'].is_cmd_acked:
                        if self.last_cmd == "P": self.eraser_on = True
                        elif self.last_cmd == "Y": self.eraser_on = False

                    if getattr(self.ctx['bt'], 'is_action_finished', False):
                        self.ctx['bt'].is_action_finished = False
                    
                    pixel_dist, target_abs_angle, delta_angle = 0.0, 0.0, 0.0
                    force_backward = False # 🌟 新增盲區倒車旗標

                    if target is not None:
                        # 角度永遠以 ArUco 為基準
                        delta_angle, _, target_abs_angle = self.ctx['planner'].get_relative_movement(
                            self.ctx['robot'].aruco_x, self.ctx['robot'].aruco_y, self.ctx['robot'].angle, target[0], target[1]
                        )
                        
                        if self.state == "RETURNING":
                            pixel_dist = np.hypot(self.ctx['robot'].aruco_x - target[0], self.ctx['robot'].aruco_y - target[1])
                        else:
                            # 距離以板擦為基準
                            pixel_dist = np.hypot(self.ctx['robot'].x - target[0], self.ctx['robot'].y - target[1])
                            
                            # 🌟 ArUco 軸心投影法 + 橫向偏移防護 (Dead Zone & Lateral Check)
                            F_vec = np.array([self.ctx['robot'].x - self.ctx['robot'].aruco_x, 
                                              self.ctx['robot'].y - self.ctx['robot'].aruco_y])
                            T_vec = np.array([target[0] - self.ctx['robot'].aruco_x, 
                                              target[1] - self.ctx['robot'].aruco_y])
                            
                            length_F = np.linalg.norm(F_vec)
                            dist_T = np.linalg.norm(T_vec)
                            
                            if length_F > 0:
                                projection = np.dot(T_vec, F_vec) / length_F
                                # 計算橫向偏移量
                                val = dist_T**2 - projection**2
                                lateral_dist = np.sqrt(val) if val > 0 else 0.0
                                
                                marker_len = getattr(self.ctx['robot'], 'marker_pixel_length', 40)
                                
                                # 車腹死區：投影在車身內，且「橫向距離極小 (卡在底盤)」才強制退後
                                if 0 <= projection <= length_F and lateral_dist < (0.8 * marker_len):
                                    force_backward = True
                                    print(f"⚠️ [盲區防護] 目標卡在底盤下方，強制拉開！(投影:{projection:.1f}, 橫向:{lateral_dist:.1f})")
                                # 背後死區：目標在 ArUco 後方，且距離太近
                                elif projection < 0 and dist_T < (1.5 * marker_len):
                                    force_backward = True
                                    print(f"⚠️ [盲區防護] 目標緊貼車尾，強制拉開距離！(距離:{dist_T:.1f})")

                    # 產生新指令
                    if self.state == "CLEANING" and target is not None:
                        if not self.eraser_on:
                            new_cmd = "P"  
                        else:
                            if pixel_dist < int(ARRIVAL_DIST_BASE * current_scale):
                                new_cmd = "S"
                                self.ctx['planner'].mark_target_reached()
                                
                                # 🌟 貪婪吞噬：瞬間消耗掉佇列中所有已經被踩在腳底下的超近目標
                                arrival_radius = int(ARRIVAL_DIST_BASE * current_scale)
                                while self.ctx['planner'].task_queue:
                                    next_pt = self.ctx['planner'].task_queue[0]
                                    if np.hypot(self.ctx['robot'].x - next_pt[0], self.ctx['robot'].y - next_pt[1]) < arrival_radius:
                                        self.ctx['planner'].task_queue.pop(0)
                                        print(f"🧹 [路徑優化] 順便清除腳底下的重疊網格點: {next_pt}")
                                    else:
                                        break
                                        
                            elif force_backward:
                                new_cmd = "B" # 🌟 觸發盲區倒車
                            elif abs(delta_angle) > TURN_ANGLE_THRESH:
                                direction = "R" if delta_angle > 0 else "L"
                                new_cmd = f"{direction}{target_abs_angle:.1f}"
                            else:
                                new_cmd = "F"
                    elif self.state in ["RETURNING", "VERIFYING", "IDLE"]:
                        if self.eraser_on:
                            new_cmd = "Y"  
                        else:
                            if self.state == "RETURNING" and target is not None:
                                arrival_radius = int(ARRIVAL_DIST_BASE * current_scale)
                                if getattr(self, 'is_aligning_home', False):
                                    arrival_radius = int(ARRIVAL_DIST_BASE * current_scale * 2.5) 

                                if pixel_dist < arrival_radius:
                                    self.is_aligning_home = True
                                    angle_diff = self.home_angle - self.ctx['robot'].angle
                                    if angle_diff > 180: angle_diff -= 360
                                    elif angle_diff < -180: angle_diff += 360
                                    
                                    if abs(angle_diff) > HOME_ANGLE_TOLERANCE:
                                        direction = "R" if angle_diff > 0 else "L"
                                        new_cmd = f"{direction}{self.home_angle:.1f}"
                                    else:
                                        print("🏠 [自動復位成功] 已安全退回基地！啟動視覺複檢程序...")
                                        new_cmd = "S"
                                        self.state = "VERIFYING"
                                        self.verify_timer = time.time()
                                        self.is_aligning_home = False
                                elif abs(delta_angle) > TURN_ANGLE_THRESH:
                                    self.is_aligning_home = False
                                    direction = "R" if delta_angle > 0 else "L"
                                    new_cmd = f"{direction}{target_abs_angle:.1f}"
                                else:
                                    self.is_aligning_home = False
                                    new_cmd = "F"
                            else:
                                new_cmd = "S"
                    else:
                        new_cmd = "S"

                    # 防撞智能圍籬
                    if target is not None:
                        dist_front = cv2.pointPolygonTest(self.cached_roi_array, (self.ctx['robot'].x, self.ctx['robot'].y), True)
                        dist_back = cv2.pointPolygonTest(self.cached_roi_array, (self.ctx['robot'].aruco_x, self.ctx['robot'].aruco_y), True)
                        safe_margin = int(SAFE_MARGIN_BASE * current_scale)
                        if (0 <= dist_front < safe_margin) or (0 <= dist_back < safe_margin):
                            if dist_front < dist_back:
                                print(f"🛑 [智能圍籬] 車頭面壁，強制倒車迴避！")
                                new_cmd = "B"
                            else:
                                print(f"🛑 [智能圍籬] 車尾靠牆，強制前進脫離！")
                                new_cmd = "F"

                    # ==========================================
                    # 🌟 終極發送指令機制 (防震盪 + 防洪 + 狀態鎖)
                    # ==========================================
                    current_time = time.time()
                    is_override = new_cmd[0] in ["S", "B", "P", "Y"]

                    # 1. 角度防震盪 (Angle Debouncer)
                    if self.last_cmd is not None and new_cmd[0] in ['L', 'R'] and self.last_cmd[0] in ['L', 'R']:
                        try:
                            last_ang = float(self.last_cmd[1:])
                            new_ang = float(new_cmd[1:])
                            if abs(new_ang - last_ang) < 5.0:
                                new_cmd = self.last_cmd
                        except ValueError:
                            pass

                    # 2. 藍牙防洪閘門 (Un-ACKed Block)
                    is_blocked_by_ack = self.ctx.get('bt') and not self.ctx['bt'].is_cmd_acked and not is_override
                    if is_blocked_by_ack:
                        if current_time - self.last_send_time < 2.0:
                            new_cmd = self.last_cmd
                        else:
                            # 🌟 超時解鎖：如果等 ACK 已經超過 2 秒，強制破除死結放行新指令
                            print("⚠️ [防洪閘門] 等待 ACK 超時 (2s)，強制解鎖放行新指令！")
                            self.ctx['bt'].is_cmd_acked = True

                    # 3. 0.4秒絕對狀態鎖 (Time Lock)
                    if not is_override and self.last_cmd is not None and current_time < getattr(self, 'cmd_lock_expiry', 0):
                        new_cmd = self.last_cmd

                    # 正式發送判定
                    if new_cmd != self.last_cmd:
                        if target is not None:
                            print(f"📤 切換動作: {new_cmd} (距離: {pixel_dist:.1f}, 目標絕對角: {target_abs_angle:.1f})")
                        self.ctx['bt'].send_new_action(new_cmd)
                        self.last_cmd = new_cmd
                        self.last_send_time = current_time
                        self.retry_count = 0
                        self.ack_start_pos = (self.ctx['robot'].x, self.ctx['robot'].y)
                        self.ack_start_angle = self.ctx['robot'].angle
                        
                        if new_cmd[0] in ["F", "L", "R"]:
                            self.cmd_lock_expiry = current_time + 0.4
                        else:
                            self.cmd_lock_expiry = 0
                    else:
                        if current_time - self.last_send_time > self.RETRY_COOLDOWN:
                            if not self.ctx['bt'].is_cmd_acked:
                                if self.retry_count < self.MAX_RETRIES:
                                    self.retry_count += 1
                                    print(f"⚠️ 強硬重發 ({self.retry_count}/{self.MAX_RETRIES}): {new_cmd}")
                                    self.ctx['bt'].resend_action()
                                    self.last_send_time = current_time
                                else:
                                    self.ctx['bt'].is_cmd_acked = True
                            else:
                                if current_time - self.last_send_time > 3.0:
                                    self.last_send_time = current_time
        hud_frame = self.ctx['visualizer'].draw_hud(frame, self.ctx['robot'], self.ctx['whiteboard'], self.ctx['planner'], robot_corners, dirty_rects, robot_mask_pts=robot_mask_pts)
        
        is_bt_connected = self.ctx.get('bt') and self.ctx['bt'].is_connected
        bt_status = "Connected" if is_bt_connected else "DISCONNECTED"
        bt_color = (0, 255, 0) if is_bt_connected else (0, 0, 255)
        cv2.putText(hud_frame, f"MODE 0: AUTO ({self.state}) | BT: {bt_status}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bt_color, 2)
        
        if robot_mask_pts is not None:
            box_pts = getattr(self.ctx['robot'], 'box_3d_pts', None)
            if box_pts is not None and len(box_pts) == 8:
                edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
                for start, end in edges:
                    cv2.line(hud_frame, tuple(box_pts[start]), tuple(box_pts[end]), (0, 255, 255), 2) 
            cv2.polylines(hud_frame, [np.array(robot_mask_pts, dtype=np.int32)], True, (255, 0, 255), 2)
        
        self.ctx['visualizer'].show_windows(hud_frame, aruco_mask, ink_mask)

    def handle_key(self, key):
        if key == ord('h'):
            if self.ctx['robot'].x is not None:
                self.home_pos = (self.ctx['robot'].aruco_x, self.ctx['robot'].aruco_y)
                self.home_angle = self.ctx['robot'].angle
                print(f"\n🏠 [熱鍵設定] 更新復位基地為: {self.home_pos}")
                
        elif key == ord('s'):
            if self.ctx['robot'].x is not None:
                print("\n▶️ [Mode 0] 拍下快照，建立任務清單！")
                if getattr(self, 'home_pos', None) is None:
                    self.home_pos = (self.ctx['robot'].aruco_x, self.ctx['robot'].aruco_y)
                    self.home_angle = self.ctx['robot'].angle
                    print(f"📍 [自動紀錄] 建立專屬基地位置: {self.home_pos}")
                
                dirty_list = self.ctx['whiteboard'].get_dirty_list()
                
                # 🌟 取得當下的標籤像素大小並傳入
                marker_length = getattr(self.ctx['robot'], 'marker_pixel_length', None)
                has_tasks = self.ctx['planner'].generate_task_queue(
                    dirty_list, self.ctx['robot'].x, self.ctx['robot'].y, marker_length
                )

                if has_tasks:
                    self.ctx['planner'].reset_count = 0
                    self.state = "CLEANING"
                else:
                    print("✨ 畫面很乾淨，不需要擦拭！")
            else:
                print("\n❌ [錯誤] 尚未辨識到車體標籤！")

        elif key == ord('p'):
            print("\n⏸️ [Mode 0] 暫停待機")
            self.state = "IDLE"
            if self.ctx.get('bt'): self.ctx['bt'].send_new_action("S")
        elif key == ord('z'):
            if self.ctx.get('bt'): self.ctx['bt'].send_new_action("Z")
        elif key == ord('e'):
            if self.ctx.get('bt'): self.ctx['bt'].send_new_action("P")
        elif key == ord('q'):
            if self.ctx.get('bt'): self.ctx['bt'].send_new_action("Y")
# ==========================================
#  3. Mode 1: 純視覺除錯模式 (Pure Vision Debug Mode)
# ==========================================
class VisionDebugMode(BaseMode):
    def activate(self):
        print("\n [Mode 1] 已切換至純視覺除錯模式 (自動關閉藍牙)")
        if self.ctx.get('bt'):
            self.ctx['bt'].disable_auto_connect()
        self.ctx['planner'].current_target = None 

    def process_frame(self, frame):
        aruco_mask = self.ctx['vision'].get_aruco_ready_mask(frame, roi_polygon=self.ctx['roi_polygon'])
        robot_center, robot_corners = self.ctx['extractor'].extract_robot_pose(aruco_mask)
        
        robot_mask_pts = None
        if robot_center is not None:
            self.ctx['robot'].update_state(robot_center, robot_corners)
            robot_mask_pts = self.ctx['robot'].get_mask_polygon()

        ink_mask = self.ctx['vision'].get_ink_clean_mask(frame, robot_mask_pts=robot_mask_pts, roi_polygon=self.ctx['roi_polygon'])
        dirty_rects = self.ctx['extractor'].extract_dirty_rects(ink_mask)

        self.ctx['whiteboard'].update_dirty_matrix(dirty_rects)

        hud_frame = self.ctx['visualizer'].draw_hud(frame, self.ctx['robot'], self.ctx['whiteboard'], self.ctx['planner'], robot_corners, dirty_rects, robot_mask_pts=robot_mask_pts)
        cv2.putText(hud_frame, "MODE 1: PURE VISION DEBUG (BT Off)", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if robot_mask_pts is not None:
            box_pts = getattr(self.ctx['robot'], 'box_3d_pts', None)
            if box_pts is not None and len(box_pts) == 8:
                edges = [
                    (0,1), (1,2), (2,3), (3,0), # 車頂
                    (4,5), (5,6), (6,7), (7,4), # 車底
                    (0,4), (1,5), (2,6), (3,7)  # 柱子
                ]
                for start, end in edges:
                    # 加上 int() 確保 OpenCV 畫線不會因為 numpy 型別報錯
                    pt1 = (int(box_pts[start][0]), int(box_pts[start][1]))
                    pt2 = (int(box_pts[end][0]), int(box_pts[end][1]))
                    cv2.line(hud_frame, pt1, pt2, (0, 255, 255), 2)
            cv2.polylines(hud_frame, [np.array(robot_mask_pts, dtype=np.int32)], True, (255, 0, 255), 2)

        self.ctx['visualizer'].show_windows(hud_frame, aruco_mask, ink_mask)

    def handle_key(self, key): pass


# ==========================================
#  4. Mode 2: 測試遙控與精準角度模式
# ==========================================
class ManualControlMode(BaseMode):
    def __init__(self, shared_context):
        super().__init__(shared_context)
        self.target_angle = 0.0  

        self.last_cmd = None
        self.last_send_time = 0
        self.RETRY_COOLDOWN = 0.5
        self.MAX_RETRIES = 50         
        self.retry_count = 0
        self.eraser_on = False
        
        # 🌟 新增：紀錄視覺代償姿態
        self.ack_start_pos = None
        self.ack_start_angle = None

    def activate(self):
        print("\n🕹️ [Mode 2] 已切換至測試遙控模式！(拔除失明防暴走)")
        if self.ctx.get('bt'):
            self.ctx['bt'].enable_auto_connect()
        self.ctx['planner'].current_target = None

    def process_frame(self, frame):
        aruco_mask = self.ctx['vision'].get_aruco_ready_mask(frame, roi_polygon=self.ctx['roi_polygon'])
        robot_center, robot_corners = self.ctx['extractor'].extract_robot_pose(aruco_mask)
        
        robot_mask_pts = None
        if robot_center is not None:
            self.ctx['robot'].update_state(robot_center, robot_corners)
            robot_mask_pts = self.ctx['robot'].get_mask_polygon()

            # 🌟 視覺代償邏輯：如果車子有標籤，且正在等待 ACK，檢查是否已經移動
            current_scale = self.ctx.get('res_scale', 1.0)
            if self.ctx.get('bt') and not self.ctx['bt'].is_cmd_acked and self.ack_start_pos is not None:
                dist = np.hypot(self.ctx['robot'].x - self.ack_start_pos[0], self.ctx['robot'].y - self.ack_start_pos[1])
                delta_ang = abs(self.ctx['robot'].angle - self.ack_start_angle)
                if delta_ang > 180: delta_ang = 360 - delta_ang
                
                if dist > (VISUAL_COMP_DIST_BASE * current_scale) or delta_ang > VISUAL_COMP_ANGLE:
                    print(f"👀 [視覺代償] 遙控指令 {self.last_cmd} 已引發實體動作，強制停止重傳！")
                    self.ctx['bt'].is_cmd_acked = True
                    self.ack_start_pos = None

        ink_mask = self.ctx['vision'].get_ink_clean_mask(frame, robot_mask_pts=robot_mask_pts, roi_polygon=self.ctx['roi_polygon'])
        dirty_rects = self.ctx['extractor'].extract_dirty_rects(ink_mask)
        self.ctx['whiteboard'].update_dirty_matrix(dirty_rects)

        if self.ctx.get('bt') and self.ctx['bt'].is_cmd_acked:
            if self.last_cmd == "P": self.eraser_on = True
            elif self.last_cmd == "Y": self.eraser_on = False

        # ❌ 已拔除 robot_center is None (失明) 時的 MAX_LOST_FRAMES 煞車機制

        if self.last_cmd is not None and self.ctx.get('bt') and not self.ctx['bt'].is_cmd_acked:
            current_time = time.time()
            if current_time - self.last_send_time > self.RETRY_COOLDOWN:
                if self.retry_count < self.MAX_RETRIES:
                    self.retry_count += 1
                    print(f"⚠️ [Mode 2] 尚未收到回傳，繼續重發 ({self.retry_count}/{self.MAX_RETRIES}): {self.last_cmd}")
                    self.ctx['bt'].resend_action()
                    self.last_send_time = current_time
                else:
                    self.ctx['bt'].is_cmd_acked = True

        hud_frame = self.ctx['visualizer'].draw_hud(frame, self.ctx['robot'], self.ctx['whiteboard'], self.ctx['planner'], robot_corners, dirty_rects, robot_mask_pts=robot_mask_pts)
        
        is_bt_connected = self.ctx.get('bt') and self.ctx['bt'].is_connected
        bt_status = "Connected" if is_bt_connected else "DISCONNECTED"
        bt_color = (0, 255, 0) if is_bt_connected else (0, 0, 255)
        eraser_str = "ON" if self.eraser_on else "OFF"
        cv2.putText(hud_frame, f"MODE 2: TEST RC | BT: {bt_status} | Eraser: {eraser_str}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bt_color, 2)
        cv2.putText(hud_frame, f"Target Abs Angle: {self.target_angle:.1f} (I/K adjust)", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        self.ctx['visualizer'].show_windows(hud_frame, aruco_mask, ink_mask)

    def _send_manual_cmd(self, cmd):
        bt = self.ctx.get('bt')
        if not bt: return
        if cmd == self.last_cmd: return

        print(f"🕹️ 遙控發送: {cmd}")
        bt.send_new_action(cmd)
        
        self.last_cmd = cmd
        self.last_send_time = time.time()
        self.retry_count = 0
        # 🌟 發送指令瞬間紀錄車體當下座標
        if self.ctx['robot'].x is not None:
            self.ack_start_pos = (self.ctx['robot'].x, self.ctx['robot'].y)
            self.ack_start_angle = self.ctx['robot'].angle

    def handle_key(self, key):
        if key in [ord('i'), ord('I')]:
            self.target_angle = (self.target_angle + 5.0) % 360
            if self.target_angle > 180.0: self.target_angle -= 360.0
            print(f"🔄 絕對角度設定為: {self.target_angle:.1f}°")
        elif key in [ord('k'), ord('K')]:
            self.target_angle -= 5.0
            if self.target_angle < -180.0: self.target_angle += 360.0
            print(f"🔄 絕對角度設定為: {self.target_angle:.1f}°")
        elif key in [ord('w'), ord('W')]: self._send_manual_cmd("F")
        elif key in [ord('x'), ord('X')]: self._send_manual_cmd("B")  
        elif key in [ord('s'), ord('S'), 32]: self._send_manual_cmd("S")
        elif key in [ord('l'), ord('L')]: self._send_manual_cmd(f"L{self.target_angle:.1f}")
        elif key in [ord('r'), ord('R')]: self._send_manual_cmd(f"R{self.target_angle:.1f}")
        elif key in [ord('z'), ord('Z')]: self._send_manual_cmd("Z")
        elif key in [ord('e'), ord('E')]: self._send_manual_cmd("P")
        elif key in [ord('y'), ord('Y')]: self._send_manual_cmd("Y")