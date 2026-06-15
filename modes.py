# modes.py
import cv2
import time
from abc import ABC, abstractmethod
import numpy as np


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
        
        # 🌟 新增：紀錄發送指令當下的實體狀態，用於視覺代償
        self.ack_start_pos = None
        self.ack_start_angle = None

        self.home_pos = None
        self.is_returning_home = False

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
                    if self.ctx.get('bt') : 
                        if self.last_cmd == "F":
                            rescue_cmd = "B" 
                            print("⚠️ 警告：丟失標籤！判斷為過衝，嘗試盲退自救！")
                        else:
                            rescue_cmd = "S" 
                            print("⚠️ 警告：丟失標籤！啟動緊急停止！")
                            
                        if self.last_cmd != rescue_cmd:
                            self.ctx['bt'].send_new_action(rescue_cmd)
                            self.last_cmd = rescue_cmd
                            self.last_send_time = time.time()
                            self.retry_count = 0

                    self.lost_frames_count = self.MAX_LOST_FRAMES 
            else:
                self.lost_frames_count = 0
                
                # 🌟 視覺代償邏輯 (判斷車子是否已經真的在動了)
                current_scale = self.ctx.get('res_scale', 1.0)
                if self.ctx.get('bt') and not self.ctx['bt'].is_cmd_acked and self.ack_start_pos is not None:
                    dist = np.hypot(self.ctx['robot'].x - self.ack_start_pos[0], self.ctx['robot'].y - self.ack_start_pos[1])
                    delta_ang = abs(self.ctx['robot'].angle - self.ack_start_angle)
                    if delta_ang > 180: delta_ang = 360 - delta_ang
                    
                    # 若移動超過 10px 或旋轉超過 5度，視為指令已被成功執行
                    if dist > (10 * current_scale) or delta_ang > 5.0:
                        print(f"👀 [視覺代償] 偵測到車體已作動，強制放行重傳指令: {self.last_cmd}")
                        self.ctx['bt'].is_cmd_acked = True
                        self.ack_start_pos = None

                dirty_list = self.ctx['whiteboard'].get_dirty_list()
                
                if self.is_returning_home:
                    target = self.home_pos # 如果在復位模式，死命盯著起點走就好
                else:
                    target = self.ctx['planner'].plan_next_target(dirty_list, self.ctx['robot'].x, self.ctx['robot'].y)
                    # 如果字跡擦完了，且有起點紀錄，無縫切換到自動復位模式
                    if target is None and self.home_pos is not None:
                        print("🎉 全部的擦拭結束！啟動自動復位，準備回家...")
                        self.is_returning_home = True
                        target = self.home_pos

                if self.ctx.get('bt'):
                    if self.ctx['bt'].is_cmd_acked:
                        if self.last_cmd == "P": self.eraser_on = True
                        elif self.last_cmd == "Y": self.eraser_on = False

                    if getattr(self.ctx['bt'], 'is_action_finished', False):
                        self.last_cmd = None  
                        self.ctx['bt'].is_action_finished = False
                    
                    pixel_dist = 0.0
                    target_abs_angle = 0.0
                    delta_angle = 0.0
                    if target is not None:
                        # 🌟 角度計算：永遠以 ArUco 中點 (旋轉軸心) 當作方向基準
                        delta_angle, _, target_abs_angle = self.ctx['planner'].get_relative_movement(
                            self.ctx['robot'].aruco_x, self.ctx['robot'].aruco_y, self.ctx['robot'].angle, target[0], target[1]
                        )
                        
                        # 🌟 距離計算：依據任務不同，切換實體參考點
                        if self.is_returning_home:
                            # 回家了沒 -> 用 ArUco 中點當距離依據
                            pixel_dist = np.hypot(self.ctx['robot'].aruco_x - target[0], self.ctx['robot'].aruco_y - target[1])
                        else:
                            # 擦字跡到了沒 -> 用板擦中點 (robot.x, robot.y) 當距離依據
                            pixel_dist = np.hypot(self.ctx['robot'].x - target[0], self.ctx['robot'].y - target[1])

                    # 🌟 修改：馬達控制訊號與抵達判定
                    if self.is_cleaning and target is not None and not self.eraser_on and not self.is_returning_home:
                        new_cmd = "P"  # 清潔時開馬達
                    elif (not self.is_cleaning or target is None or self.is_returning_home) and self.eraser_on:
                        new_cmd = "Y"  # 只要準備回家，或者結束了，二話不說先關馬達
                    else:
                        if target is not None:
                            if pixel_dist < int(20 * current_scale):  
                                if self.is_returning_home:
                                    angle_diff = self.home_angle - self.ctx['robot'].angle
                                    if angle_diff > 180: angle_diff -= 360
                                    elif angle_diff < -180: angle_diff += 360
                                    
                                    if abs(angle_diff) > 10:
                                        direction = "R" if angle_diff > 0 else "L"
                                        new_cmd = f"{direction}{abs(angle_diff):.1f}"
                                        print(f"🔄 [姿態校正] 已達原點，原地轉正車頭中: {new_cmd}")
                                    else:
                                        print("🏠 [自動復位成功] 已安全回到基地且車頭對齊！(基地座標將持續保留)")
                                        new_cmd = "S"
                                        self.is_cleaning = False
                                        self.is_returning_home = False
                                else:
                                    new_cmd = "S"
                                    self.ctx['planner'].mark_as_visited(target[0], target[1])
                                    self.ctx['planner'].current_target = None
                            elif abs(delta_angle) > 165:  
                                new_cmd = "B"
                            elif abs(delta_angle) > 15:
                                direction = "R" if delta_angle > 0 else "L"
                                new_cmd = f"{direction}{abs(delta_angle):.1f}"
                            else:
                                new_cmd = "F"
                        else:
                            new_cmd = "S"

                    if target is not None:
                        # 將 ROI 轉為 numpy array 以便計算
                        roi_array = np.array(self.ctx['roi_polygon'], dtype=np.int32)
                        # 計算車子中心與邊界的最短距離
                        dist_to_edge = cv2.pointPolygonTest(roi_array, (self.ctx['robot'].x, self.ctx['robot'].y), True)
                        
                        safe_margin = int(25 * current_scale) # 安全邊距 (約 35-40 像素)
                        
                        # 如果距離邊界太近 (大於 0 代表在內側，但小於安全距離)
                        if 0 <= dist_to_edge < safe_margin:
                            print(f"🛑 [電子圍籬觸發] 距離邊界僅 {dist_to_edge:.1f}px，緊急迴避！")
                            # 如果本來想前進，強制改成後退
                            if new_cmd == "F":
                                new_cmd = "B"
                            # 如果是其他的，強制煞車
                            elif new_cmd != "B":
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
                        # 🌟 紀錄送出指令當下的姿態
                        self.ack_start_pos = (self.ctx['robot'].x, self.ctx['robot'].y)
                        self.ack_start_angle = self.ctx['robot'].angle
                    else:
                        if current_time - self.last_send_time > self.RETRY_COOLDOWN:
                            if not self.ctx['bt'].is_cmd_acked:
                                if self.retry_count < self.MAX_RETRIES:
                                    self.retry_count += 1
                                    print(f"⚠️ 未收到格式確認，強硬重發 ({self.retry_count}/{self.MAX_RETRIES}): {new_cmd}")
                                    self.ctx['bt'].resend_action()
                                    self.last_send_time = current_time
                                else:
                                    print(f"❌ 放棄重試指令: {self.last_cmd}，強制放行。")
                                    self.ctx['bt'].is_cmd_acked = True
                            else:
                                if new_cmd != self.last_cmd:
                                    print(f"🔄 更新角度: {new_cmd}")
                                    self.ctx['bt'].send_new_action(new_cmd)
                                    self.last_cmd = new_cmd
                                    self.last_send_time = current_time
                                    self.retry_count = 0
                                    # 🌟 更新姿態紀錄
                                    self.ack_start_pos = (self.ctx['robot'].x, self.ctx['robot'].y)
                                    self.ack_start_angle = self.ctx['robot'].angle
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
        if key == ord('h'):
            if self.ctx['robot'].x is not None:
                # 🌟 改用 ArUco 座標當作基地位置
                self.home_pos = (self.ctx['robot'].aruco_x, self.ctx['robot'].aruco_y)
                self.home_angle = self.ctx['robot'].angle
                print(f"\n🏠 [熱鍵設定] 更新復位基地(ArUco軸心)為: {self.home_pos}, 角度: {self.home_angle:.1f}°")
            else:
                print("\n❌ [錯誤] 畫面上找不到車子標籤，無法設定起點！請確保車子在視線內。")
                
        elif key == ord('s'):
            if self.ctx['robot'].x is not None:
                print("\n▶️ [Mode 0] 開始自動擦拭")
                
                if getattr(self, 'home_pos', None) is None:
                    # 🌟 自動記錄時也用 ArUco 座標
                    self.home_pos = (self.ctx['robot'].aruco_x, self.ctx['robot'].aruco_y)
                    self.home_angle = self.ctx['robot'].angle
                    print(f"📍 [自動紀錄] 建立專屬基地位置: {self.home_pos}, 角度: {self.home_angle:.1f}°")
                else:
                    print(f"📍 [保留設定] 任務結束將回到專屬基地: {self.home_pos}, 角度: {self.home_angle:.1f}°")
                    
                self.is_returning_home = False
                self.is_cleaning = True
            else:
                print("\n❌ [錯誤] 尚未辨識到車體標籤，請確保 ArUco 在畫面中再啟動！")
                
        elif key == ord('p'):
            print("\n⏸️ [Mode 0] 暫停待機")
            self.is_cleaning = False
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
                
                if dist > (10 * current_scale) or delta_ang > 5.0:
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