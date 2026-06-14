# main.py
import cv2
import numpy as np
import time
import threading
from abc import ABC, abstractmethod

from vision import VisionManager
from extractor import FeatureExtractor
from robot import Robot
from whiteboard import Whiteboard
from planner import CleaningPlanner
from display import Visualizer

try:
    from bluetooth import BTInterface
    BT_AVAILABLE = True
except ImportError:
    print("⚠️ 找不到 bluetooth.py，藍牙控制將無法正常運作。")
    BT_AVAILABLE = False

BLUETOOTH_PORT = '/dev/tty.usbserial-140'  # 請確認您的 COM Port


# ==========================================
#  全域背景執行緒：自動連線、斷線重連與嚴格 ACK 驗證
# ==========================================
def bluetooth_worker(ctx):
    """背景執行緒：負責藍牙連線，並絕對嚴格驗證 Arduino 回傳的 ACK 格式"""
    last_attempt_time = 0
    while True:
        if ctx.get('bt_should_connect') and BT_AVAILABLE:
            if ctx.get('bt') is None:
                if time.time() - last_attempt_time > 3.0:
                    last_attempt_time = time.time()
                    print(f"\n🔄 [藍牙背景] 嘗試連線至 {BLUETOOTH_PORT}...")
                    try:
                        bt_instance = BTInterface(port=BLUETOOTH_PORT)
                        ctx['bt'] = bt_instance
                        ctx['is_cmd_acked'] = True
                        print("\n✅ [藍牙背景] 藍牙連線成功！")
                    except Exception as e:
                        pass
            else:
                try:
                    msg = ctx['bt'].listen()
                    if msg:
                        msg_str = str(msg).strip()
                        if msg_str:
                            print(f"\n🟢 [智慧車回傳]: {msg_str}")
                            
                            # ==================================================
                            # 🔥 絕對嚴格 ACK 驗證邏輯 🔥
                            # ==================================================
                            pending = ctx.get('pending_cmd')
                            if pending and not ctx.get('is_cmd_acked'):
                                expected_c = pending[0].upper()
                                expected_ang = float(pending[1:]) if len(pending) > 1 else 0.0
                                
                                # 為了防範一次收到多行 (例如 Echo 跟 Cmd 黏在一起)，我們用換行符號切開找
                                lines = msg_str.split('\n')
                                for line in reversed(lines):
                                    line = line.strip()
                                    
                                    # 嚴格比對 Arduino 的 "Cmd: X,Ang: Y.YY"
                                    if "Cmd:" in line and ",Ang:" in line:
                                        try:
                                            parts = line.split(",")
                                            ret_c = parts[0].split(":")[1].strip()
                                            ret_ang = float(parts[1].split(":")[1].strip())
                                            
                                            # 驗證字元是否相符，且浮點數誤差小於 0.1
                                            if ret_c == expected_c and abs(ret_ang - expected_ang) < 0.1:
                                                ctx['is_cmd_acked'] = True
                                                print(f"✅ [ACK 嚴格確認] 成功驗證指令: {pending}")
                                                break # 驗證成功就跳出檢查
                                        except Exception:
                                            continue
                                            
                                    # 針對 Z 校準指令的特別放行條件 ('y' 或 '開始校準')
                                    elif expected_c == 'Z' and ('開始校準' in line or 'y' in line):
                                        ctx['is_cmd_acked'] = True
                                        print(f"✅ [ACK 嚴格確認] 陀螺儀校準已觸發！")
                                        break
                            # ==================================================

                except Exception as e:
                    print(f"\n❌ [藍牙背景] 連線異常中斷 ({e})，準備重新連線...")
                    ctx['bt'] = None
        else:
            if ctx.get('bt') is not None:
                ctx['bt'] = None
        
        time.sleep(0.05)


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

        self.RETRY_COOLDOWN = 1.5
        self.MAX_RETRIES = 50         # 提高重試次數，確保死命塞到他回傳為止
        self.retry_count = 0

        self.lost_frames_count = 0
        self.MAX_LOST_FRAMES = 5

    def activate(self):
        print("\n📡 [Mode 0] 切換至全自動控制模式 (藍牙自動連線已啟動)")
        self.ctx['bt_should_connect'] = True  

    def process_frame(self, frame):
        aruco_mask = self.ctx['vision'].get_aruco_ready_mask(frame, roi_polygon=self.ctx['roi_polygon'])
        robot_center, robot_corners = self.ctx['extractor'].extract_robot_pose(aruco_mask)
        ink_mask, robot_mask_pts = self.ctx['vision'].get_ink_clean_mask(frame, exclude_polygon=robot_corners, roi_polygon=self.ctx['roi_polygon'])
        dirty_rects = self.ctx['extractor'].extract_dirty_rects(ink_mask)

        if robot_center is not None:
            self.ctx['robot'].update_state(robot_center, robot_corners)
            self.ctx['whiteboard'].update_dirty_matrix(dirty_rects)

        if self.is_cleaning:
            if robot_center is None:
                self.lost_frames_count += 1
                if self.lost_frames_count >= self.MAX_LOST_FRAMES:
                    if self.ctx.get('bt') and self.last_cmd != "S": 
                        print("⚠️ 警告：丟失車體標籤！啟動緊急停止並等待確認！")
                        self.ctx['pending_cmd'] = "S"
                        self.ctx['is_cmd_acked'] = False
                        self.ctx['bt'].send_action("S")
                        self.last_cmd = "S"
                        self.last_send_time = time.time()
                        self.retry_count = 0
                    self.lost_frames_count = self.MAX_LOST_FRAMES 
            else:
                self.lost_frames_count = 0
                
                dirty_list = self.ctx['whiteboard'].get_dirty_list()
                target = self.ctx['planner'].plan_next_target(dirty_list, self.ctx['robot'].x, self.ctx['robot'].y)

                if self.ctx.get('bt'):
                    if target is not None:
                        # 有目標，計算路徑與絕對角度
                        delta_angle, pixel_dist, target_abs_angle = self.ctx['planner'].get_relative_movement(
                            self.ctx['robot'].x, self.ctx['robot'].y, self.ctx['robot'].angle, target[0], target[1]
                        )

                        if pixel_dist < 5:
                            new_cmd = "S"
                            self.ctx['planner'].mark_as_visited(target[0], target[1])
                            self.ctx['planner'].current_target = None
                        elif abs(delta_angle) > 15:
                            direction = "R" if delta_angle > 0 else "L"
                            new_cmd = f"{direction}{target_abs_angle:.1f}"
                        else:
                            new_cmd = "F"
                    else:
                        # 🔥【新增】白板乾淨了，或是所有點都在黑名單內，強制原地停止！
                        new_cmd = "S"

                    # 藍牙發送與節流控制（無目標時也同樣受到嚴格 ACK 保護）
                    current_time = time.time()
                    new_base_cmd = new_cmd[0]
                    last_base_cmd = self.last_cmd[0] if self.last_cmd else None
                    
                    if new_base_cmd != last_base_cmd:
                        # 如果動作改變 (例如從 F 變成 S)
                        if target is not None:
                            print(f"📤 切換動作: {new_cmd} (距離: {pixel_dist:.1f}, 目標絕對角: {target_abs_angle:.1f})")
                        else:
                            print(f"🎉 擦拭完畢 (無殘留字跡)！發送停止指令: {new_cmd}")
                            
                        self.ctx['pending_cmd'] = new_cmd   # 註冊待確認指令
                        self.ctx['is_cmd_acked'] = False    # 剝奪確認狀態
                        self.ctx['bt'].send_action(new_cmd)
                        self.last_cmd = new_cmd
                        self.last_send_time = current_time
                        self.retry_count = 0
                    else:
                        if current_time - self.last_send_time > self.RETRY_COOLDOWN:
                            if not self.ctx['is_cmd_acked']:
                                if self.retry_count < self.MAX_RETRIES:
                                    self.retry_count += 1
                                    print(f"⚠️ 未收到正確格式確認，強硬重發 ({self.retry_count}/{self.MAX_RETRIES}): {new_cmd}")
                                    self.ctx['pending_cmd'] = new_cmd 
                                    self.ctx['bt'].send_action(new_cmd)
                                    self.last_cmd = new_cmd
                                    self.last_send_time = current_time
                                else:
                                    print(f"❌ 放棄重試指令: {self.last_cmd}，強制放行防止系統死鎖。")
                                    self.ctx['is_cmd_acked'] = True
                            else:
                                if new_cmd != self.last_cmd:
                                    print(f"🔄 更新角度: {new_cmd}")
                                    self.ctx['pending_cmd'] = new_cmd 
                                    self.ctx['is_cmd_acked'] = False
                                    self.ctx['bt'].send_action(new_cmd)
                                    self.last_cmd = new_cmd
                                    self.last_send_time = current_time
                                    self.retry_count = 0
        else:
            self.ctx['planner'].current_target = None

        hud_frame = self.ctx['visualizer'].draw_hud(frame, self.ctx['robot'], self.ctx['whiteboard'], self.ctx['planner'], robot_corners, dirty_rects, robot_mask_pts=robot_mask_pts)
        bt_status = "Connected" if self.ctx.get('bt') else "DISCONNECTED"
        bt_color = (0, 255, 0) if self.ctx.get('bt') else (0, 0, 255)
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
                self.ctx['pending_cmd'] = "S"
                self.ctx['is_cmd_acked'] = False
                self.ctx['bt'].send_action("S")
        elif key == ord('z'):
            print("\n📤 [Mode 0] 發送陀螺儀校準指令: Z")
            if self.ctx.get('bt'): 
                self.ctx['pending_cmd'] = "Z"
                self.ctx['is_cmd_acked'] = False
                self.ctx['bt'].send_action("Z")


# ==========================================
#  3. Mode 1: 純視覺除錯模式 (Pure Vision Debug Mode)
# ==========================================
class VisionDebugMode(BaseMode):
    def activate(self):
        print("\n👁️ [Mode 1] 已切換至純視覺除錯模式 (自動關閉藍牙)")
        self.ctx['bt_should_connect'] = False  
        self.ctx['planner'].current_target = None 

    def process_frame(self, frame):
        aruco_mask = self.ctx['vision'].get_aruco_ready_mask(frame, roi_polygon=self.ctx['roi_polygon'])
        robot_center, robot_corners = self.ctx['extractor'].extract_robot_pose(aruco_mask)
        ink_mask, robot_mask_pts = self.ctx['vision'].get_ink_clean_mask(frame, exclude_polygon=robot_corners, roi_polygon=self.ctx['roi_polygon'])
        dirty_rects = self.ctx['extractor'].extract_dirty_rects(ink_mask)

        if robot_center is not None:
            self.ctx['robot'].update_state(robot_center, robot_corners)
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
        self.MAX_RETRIES = 5         
        self.retry_count = 0
        self.lost_frames_count = 0
        self.MAX_LOST_FRAMES = 5

    def activate(self):
        print("\n🕹️ [Mode 2] 已切換至測試遙控模式！(藍牙自動連線已啟動)")
        self.ctx['bt_should_connect'] = True
        self.ctx['planner'].current_target = None

    def process_frame(self, frame):
        aruco_mask = self.ctx['vision'].get_aruco_ready_mask(frame, roi_polygon=self.ctx['roi_polygon'])
        robot_center, robot_corners = self.ctx['extractor'].extract_robot_pose(aruco_mask)
        ink_mask, robot_mask_pts = self.ctx['vision'].get_ink_clean_mask(frame, exclude_polygon=robot_corners, roi_polygon=self.ctx['roi_polygon'])
        dirty_rects = self.ctx['extractor'].extract_dirty_rects(ink_mask)

        if robot_center is not None:
            self.ctx['robot'].update_state(robot_center, robot_corners)
        self.ctx['whiteboard'].update_dirty_matrix(dirty_rects)

        if robot_center is None:
            self.lost_frames_count += 1
            if self.lost_frames_count >= self.MAX_LOST_FRAMES:
                if self.last_cmd and self.last_cmd[0] not in ["S", "Z"]:
                    if self.ctx.get('bt'): 
                        print("⚠️ [Mode 2] 警告：遙控時丟失標籤！強硬煞車！")
                        self.ctx['pending_cmd'] = "S"
                        self.ctx['is_cmd_acked'] = False
                        self.ctx['bt'].send_action("S")
                        self.last_cmd = "S"
                        self.last_send_time = time.time()
                        self.retry_count = 0
                self.lost_frames_count = self.MAX_LOST_FRAMES 
        else:
            self.lost_frames_count = 0

        # 重傳死纏爛打機制
        if self.last_cmd is not None and not self.ctx.get('is_cmd_acked', True):
            current_time = time.time()
            if current_time - self.last_send_time > self.RETRY_COOLDOWN:
                if self.retry_count < self.MAX_RETRIES:
                    self.retry_count += 1
                    print(f"⚠️ [Mode 2] 未收到嚴格確認，強硬重發 ({self.retry_count}/{self.MAX_RETRIES}): {self.last_cmd}")
                    if self.ctx.get('bt'): 
                        self.ctx['pending_cmd'] = self.last_cmd
                        self.ctx['bt'].send_action(self.last_cmd)
                    self.last_send_time = current_time
                else:
                    print(f"❌ [Mode 2] 放棄重試遙控指令: {self.last_cmd}")
                    self.ctx['is_cmd_acked'] = True

        hud_frame = self.ctx['visualizer'].draw_hud(frame, self.ctx['robot'], self.ctx['whiteboard'], self.ctx['planner'], robot_corners, dirty_rects, robot_mask_pts=robot_mask_pts)
        bt_status = "Connected" if self.ctx.get('bt') else "DISCONNECTED"
        bt_color = (0, 255, 0) if self.ctx.get('bt') else (0, 0, 255)
        cv2.putText(hud_frame, f"MODE 2: TEST RC | BT: {bt_status}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bt_color, 2)
        cv2.putText(hud_frame, f"Target Abs Angle: {self.target_angle:.1f} (I/K adjust)", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        self.ctx['visualizer'].show_windows(hud_frame, aruco_mask, ink_mask)

    def _send_manual_cmd(self, cmd):
        bt = self.ctx.get('bt')
        if not bt: return
        print(f"🕹️ 遙控發送: {cmd}")
        self.ctx['pending_cmd'] = cmd
        self.ctx['is_cmd_acked'] = False
        bt.send_action(cmd)
        
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
        elif key in [ord('s'), ord('S'), 32]: self._send_manual_cmd("S")
        elif key in [ord('l'), ord('L')]: self._send_manual_cmd(f"L{self.target_angle:.1f}")
        elif key in [ord('r'), ord('R')]: self._send_manual_cmd(f"R{self.target_angle:.1f}")
        elif key in [ord('z'), ord('Z')]: self._send_manual_cmd("Z")


# ==========================================
#  邊界手動點擊函式
# ==========================================
clicked_points = []
def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        print(f"📍 頂點 {len(clicked_points)}: ({x}, {y})")

def setup_roi_manually(cap):
    global clicked_points
    clicked_points = []
    cv2.namedWindow("Select ROI (Whiteboard Area)")
    cv2.setMouseCallback("Select ROI (Whiteboard Area)", mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        display_frame = frame.copy()
        for i, pt in enumerate(clicked_points):
            cv2.circle(display_frame, tuple(pt), 5, (0, 0, 255), -1)
            if i > 0: cv2.line(display_frame, tuple(clicked_points[i-1]), tuple(clicked_points[i]), (0, 255, 0), 2)
        if len(clicked_points) == 4:
            cv2.line(display_frame, tuple(clicked_points[3]), tuple(clicked_points[0]), (0, 255, 0), 2)
        cv2.imshow("Select ROI (Whiteboard Area)", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'): clicked_points = []
        elif key == 13 or key == 32:  
            if len(clicked_points) == 4: break
    cv2.destroyWindow("Select ROI (Whiteboard Area)")
    return clicked_points


# ==========================================
#  5. 主程式控制中樞 (Main Context)
# ==========================================
def main():
    print("\n" + "=" * 40)
    print("🚗 智慧自動擦拭機器人 - 絕對嚴格通訊版")
    print("=" * 40)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    roi_polygon = setup_roi_manually(cap)
    if not roi_polygon or len(roi_polygon) != 4:
        cap.release()
        return

    shared_context = {
        'vision': VisionManager(),
        'extractor': FeatureExtractor(),
        'robot': Robot(),
        'whiteboard': Whiteboard(),
        'planner': CleaningPlanner(),
        'visualizer': Visualizer(roi_polygon),  
        'roi_polygon': roi_polygon,
        'bt': None,                 
        'is_cmd_acked': True,       
        'pending_cmd': None,        # 🔥 新增：紀錄目前正在等待確認的精確指令
        'bt_should_connect': False  
    }

    bt_thread = threading.Thread(target=bluetooth_worker, args=(shared_context,), daemon=True)
    bt_thread.start()

    modes = {
        0: FullControlMode(shared_context),
        1: VisionDebugMode(shared_context),
        2: ManualControlMode(shared_context) 
    }

    current_mode_idx = 1 
    current_mode = modes[current_mode_idx]
    current_mode.activate()

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_mode.process_frame(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key in [ord('0'), ord('1'), ord('2')]:
            target_idx = int(chr(key))
            if target_idx != current_mode_idx and target_idx in modes:
                if current_mode_idx in [0, 2] and shared_context.get('bt'):
                    shared_context['pending_cmd'] = "S"
                    shared_context['is_cmd_acked'] = False
                    shared_context['bt'].send_action("S")

                current_mode_idx = target_idx
                current_mode = modes[current_mode_idx]
                current_mode.activate()
        else:
            current_mode.handle_key(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()