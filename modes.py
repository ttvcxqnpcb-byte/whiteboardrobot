# modes.py
import cv2
import time
from abc import ABC, abstractmethod

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
        self.is_cleaning = False
        self.last_cmd = None
        self.last_send_time = 0
        self.eraser_on = False

        self.RETRY_COOLDOWN = 1.5
        self.MAX_RETRIES = 50         
        self.retry_count = 0

        self.lost_frames_count = 0
        self.MAX_LOST_FRAMES = 5

    def activate(self):
        print("\n📡 [Mode 0] 切換至全自動控制模式 (藍牙自動連線已啟動)")
        if self.ctx.get('bt'):
            self.ctx['bt'].enable_auto_connect()

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

        if self.is_cleaning:
            if robot_center is None:
                self.lost_frames_count += 1
                if self.lost_frames_count >= self.MAX_LOST_FRAMES:
                    if self.ctx.get('bt') and self.last_cmd != "S": 
                        print("⚠️ 警告：丟失車體標籤！啟動緊急停止並等待確認！")
                        self.ctx['bt'].send_new_action("S")
                        self.last_cmd = "S"
                        self.last_send_time = time.time()
                        self.retry_count = 0
                    self.lost_frames_count = self.MAX_LOST_FRAMES 
            else:
                self.lost_frames_count = 0
                
                dirty_list = self.ctx['whiteboard'].get_dirty_list()
                target = self.ctx['planner'].plan_next_target(dirty_list, self.ctx['robot'].x, self.ctx['robot'].y)

                if self.ctx.get('bt'):
                    if self.ctx['bt'].is_cmd_acked:
                        if self.last_cmd == "P":
                            self.eraser_on = True
                        elif self.last_cmd == "Y":
                            self.eraser_on = False

                    if self.is_cleaning and target is not None and not self.eraser_on:
                        new_cmd = "P"  
                    elif (not self.is_cleaning or target is None) and self.eraser_on:
                        new_cmd = "Y"  
                    else:
                        if target is not None:
                            delta_angle, pixel_dist, target_abs_angle = self.ctx['planner'].get_relative_movement(
                                self.ctx['robot'].x, self.ctx['robot'].y, self.ctx['robot'].angle, target[0], target[1]
                            )

                            # 解除對 main.py 全域變數的依賴，改用 ctx 傳入的 res_scale
                            current_scale = self.ctx.get('res_scale', 1.0)
                            if pixel_dist < int(5 * current_scale):  
                                new_cmd = "S"
                                self.ctx['planner'].mark_as_visited(target[0], target[1])
                                self.ctx['planner'].current_target = None
                            elif abs(delta_angle) > 15:
                                direction = "R" if delta_angle > 0 else "L"
                                new_cmd = f"{direction}{target_abs_angle:.1f}"
                            else:
                                new_cmd = "F"
                        else:
                            new_cmd = "S"

                    current_time = time.time()
                    new_base_cmd = new_cmd[0]
                    last_base_cmd = self.last_cmd[0] if self.last_cmd else None
                    
                    if new_base_cmd != last_base_cmd:
                        if target is not None:
                            print(f"📤 切換動作: {new_cmd} (距離: {pixel_dist:.1f}, 目標絕對角: {target_abs_angle:.1f})")
                        else:
                            print(f"🎉 擦拭完畢 (無殘留字跡)！發送停止指令: {new_cmd}")
                            
                        self.ctx['bt'].send_new_action(new_cmd)
                        self.last_cmd = new_cmd
                        self.last_send_time = current_time
                        self.retry_count = 0
                    else:
                        if current_time - self.last_send_time > self.RETRY_COOLDOWN:
                            if not self.ctx['bt'].is_cmd_acked:
                                if self.retry_count < self.MAX_RETRIES:
                                    self.retry_count += 1
                                    print(f"⚠️ 未收到正確格式確認，強硬重發 ({self.retry_count}/{self.MAX_RETRIES}): {new_cmd}")
                                    self.ctx['bt'].resend_action()
                                    self.last_send_time = current_time
                                else:
                                    print(f"❌ 放棄重試指令: {self.last_cmd}，強制放行防止系統死鎖。")
                                    self.ctx['bt'].is_cmd_acked = True
                            else:
                                if new_cmd != self.last_cmd:
                                    print(f"🔄 更新角度: {new_cmd}")
                                    self.ctx['bt'].send_new_action(new_cmd)
                                    self.last_cmd = new_cmd
                                    self.last_send_time = current_time
                                    self.retry_count = 0
        else:
            self.ctx['planner'].current_target = None

        hud_frame = self.ctx['visualizer'].draw_hud(frame, self.ctx['robot'], self.ctx['whiteboard'], self.ctx['planner'], robot_corners, dirty_rects, robot_mask_pts=robot_mask_pts)
        
        is_bt_connected = self.ctx.get('bt') and self.ctx['bt'].is_connected
        bt_status = "Connected" if is_bt_connected else "DISCONNECTED"
        bt_color = (0, 255, 0) if is_bt_connected else (0, 0, 255)
        state_str = "Running" if self.is_cleaning else "Standby"
        
        cv2.putText(hud_frame, f"MODE 0: AUTO ({state_str}) | BT: {bt_status}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bt_color, 2)
        self.ctx['visualizer'].show_windows(hud_frame, aruco_mask, ink_mask)

    def handle_key(self, key):
        if key == ord('s'):
            print("\n▶️ [Mode 0] 開始自動擦拭")
            self.is_cleaning = True
        elif key == ord('p'):
            print("\n⏸️ [Mode 0] 暫停待機")
            self.is_cleaning = False
            if self.ctx.get('bt'): 
                self.ctx['bt'].send_new_action("S")
        elif key == ord('z'):
            print("\n📤 [Mode 0] 發送陀螺儀校準指令: Z")
            if self.ctx.get('bt'): 
                self.ctx['bt'].send_new_action("Z")
        elif key == ord('e'):
            print("\n📤 [Mode 0] 啟動板擦馬達: P")
            if self.ctx.get('bt'): 
                self.ctx['bt'].send_new_action("P")
        elif key == ord('q'):
            print("\n📤 [Mode 0] 關閉板擦馬達: Y")
            if self.ctx.get('bt'): 
                self.ctx['bt'].send_new_action("Y")


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
        self.lost_frames_count = 0
        self.MAX_LOST_FRAMES = 5
        self.eraser_on = False

    def activate(self):
        print("\n🕹️ [Mode 2] 已切換至測試遙控模式！(藍牙自動連線已啟動)")
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

        ink_mask = self.ctx['vision'].get_ink_clean_mask(frame, robot_mask_pts=robot_mask_pts, roi_polygon=self.ctx['roi_polygon'])
        dirty_rects = self.ctx['extractor'].extract_dirty_rects(ink_mask)

        self.ctx['whiteboard'].update_dirty_matrix(dirty_rects)

        if self.ctx.get('bt') and self.ctx['bt'].is_cmd_acked:
            if self.last_cmd == "P": self.eraser_on = True
            elif self.last_cmd == "Y": self.eraser_on = False

        if robot_center is None:
            self.lost_frames_count += 1
            if self.lost_frames_count >= self.MAX_LOST_FRAMES:
                if self.last_cmd and self.last_cmd[0] not in ["S", "Z"]:
                    if self.ctx.get('bt'): 
                        print("⚠️ [Mode 2] 警告：遙控時丟失標籤！強硬煞車！")
                        self.ctx['bt'].send_new_action("S")
                        self.last_cmd = "S"
                        self.last_send_time = time.time()
                        self.retry_count = 0
                self.lost_frames_count = self.MAX_LOST_FRAMES 
        else:
            self.lost_frames_count = 0

        # 重傳機制
        if self.last_cmd is not None and self.ctx.get('bt') and not self.ctx['bt'].is_cmd_acked:
            current_time = time.time()
            if current_time - self.last_send_time > self.RETRY_COOLDOWN:
                if self.retry_count < self.MAX_RETRIES:
                    self.retry_count += 1
                    print(f"⚠️ [Mode 2] 未收到嚴格確認，強硬重發 ({self.retry_count}/{self.MAX_RETRIES}): {self.last_cmd}")
                    self.ctx['bt'].resend_action()
                    self.last_send_time = current_time
                else:
                    print(f"❌ [Mode 2] 放棄重試遙控指令: {self.last_cmd}")
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
        if not bt:
            print("⏳ 藍牙未連線，指令暫時無效...")
            return
        
        if cmd == self.last_cmd:
            return

        print(f"🕹️ 遙控發送: {cmd}")
        bt.send_new_action(cmd)
        
        self.last_cmd = cmd
        self.last_send_time = time.time()
        self.retry_count = 0

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