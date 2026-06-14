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

BLUETOOTH_PORT = 'COM9'  # 請確認您的 COM Port


# ==========================================
#  全域背景執行緒：自動連線、斷線重連與接收監聽
# ==========================================
def bluetooth_worker(ctx):
    """背景執行緒：負責藍牙的自動連線、斷線重連與訊息接收"""
    last_attempt_time = 0
    while True:
        # 只有在模式 0 和模式 2，且擁有 bluetooth.py 時才嘗試連線
        if ctx.get('bt_should_connect') and BT_AVAILABLE:
            if ctx.get('bt') is None:
                # 斷線狀態：避免瘋狂重試卡死 CPU，限制每 3 秒重試一次
                if time.time() - last_attempt_time > 3.0:
                    last_attempt_time = time.time()
                    print(f"\n🔄 [藍牙背景] 嘗試連線至 {BLUETOOTH_PORT}...")
                    try:
                        # BTInterface 實例化時會嘗試 Reset 並連線
                        bt_instance = BTInterface(port=BLUETOOTH_PORT)
                        ctx['bt'] = bt_instance
                        ctx['is_cmd_acked'] = True
                        print("\n✅ [藍牙背景] 藍牙連線成功！")
                    except Exception as e:
                        pass
            else:
                # 已連線狀態：持續監聽訊息
                try:
                    msg = ctx['bt'].listen()
                    if msg:
                        print(f"\n🟢 [智慧車回傳]: {msg}")
                        ctx['is_cmd_acked'] = True
                except Exception as e:
                    print(f"\n❌ [藍牙背景] 連線異常中斷 ({e})，準備重新連線...")
                    ctx['bt'] = None
        else:
            # 模式 1 (純視覺) 時，主動斷開連線省電
            if ctx.get('bt') is not None:
                ctx['bt'] = None
        
        time.sleep(0.1) # 給背景一點呼吸空間，降低 CPU 負載


# ==========================================
#  1. 模式基底類別 (Strategy Interface)
# ==========================================
class BaseMode(ABC):
    def __init__(self, shared_context):
        self.ctx = shared_context

    @abstractmethod
    def activate(self):
        pass

    @abstractmethod
    def process_frame(self, frame):
        pass

    @abstractmethod
    def handle_key(self, key):
        pass


# ==========================================
#  2. Mode 0: 完整控制模式 (Full System Mode)
# ==========================================
class FullControlMode(BaseMode):
    def __init__(self, shared_context):
        super().__init__(shared_context)
        self.is_cleaning = False
        self.last_cmd = None
        self.last_send_time = 0

        self.RETRY_COOLDOWN = 0.5
        self.MAX_RETRIES = 3         
        self.retry_count = 0

        self.lost_frames_count = 0
        self.MAX_LOST_FRAMES = 5

    def activate(self):
        print("\n📡 [Mode 0] 切換至全自動控制模式 (藍牙自動連線已啟動)")
        self.ctx['bt_should_connect'] = True  # 通知背景執行緒開始連線藍牙

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
                    print("⚠️ 警告：連續丟失車體標籤！啟動緊急停止！")
                    if self.ctx.get('bt'): 
                        self.ctx['bt'].send_action("S")
                        self.last_cmd = "S"
                        self.last_send_time = time.time()
                        self.ctx['is_cmd_acked'] = False
                        self.retry_count = 0
                    self.lost_frames_count = self.MAX_LOST_FRAMES 
            else:
                self.lost_frames_count = 0
                
                dirty_list = self.ctx['whiteboard'].get_dirty_list()
                target = self.ctx['planner'].plan_next_target(dirty_list, self.ctx['robot'].x, self.ctx['robot'].y)

                if target is not None and self.ctx.get('bt'):
                    # 【協議整合點】從 planner 拿取絕對角度
                    delta_angle, pixel_dist, target_abs_angle = self.ctx['planner'].get_relative_movement(
                        self.ctx['robot'].x, self.ctx['robot'].y, self.ctx['robot'].angle, target[0], target[1]
                    )

                    if pixel_dist < 30:
                        new_cmd = "S"
                        self.ctx['planner'].mark_as_visited(target[0], target[1])
                        self.ctx['planner'].current_target = None
                    elif abs(delta_angle) > 15:
                        # 【協議整合點】決定轉向捷徑(R/L)，並附上絕對角度
                        direction = "R" if delta_angle > 0 else "L"
                        new_cmd = f"{direction}{target_abs_angle:.1f}"
                    else:
                        new_cmd = "F"

                    current_time = time.time()
                    new_base_cmd = new_cmd[0]
                    last_base_cmd = self.last_cmd[0] if self.last_cmd else None
                    
                    if new_base_cmd != last_base_cmd:
                        print(f"📤 切換動作: {new_cmd} (距離: {pixel_dist:.1f}, 目標絕對角: {target_abs_angle:.1f})")
                        self.ctx['bt'].send_action(new_cmd)
                        self.last_cmd = new_cmd
                        self.last_send_time = current_time
                        self.ctx['is_cmd_acked'] = False
                        self.retry_count = 0
                    else:
                        if current_time - self.last_send_time > self.RETRY_COOLDOWN:
                            if not self.ctx['is_cmd_acked']:
                                if self.retry_count < self.MAX_RETRIES:
                                    self.retry_count += 1
                                    print(f"⚠️ 未收到確認，重發 ({self.retry_count}/{self.MAX_RETRIES}): {new_cmd}")
                                    self.ctx['bt'].send_action(new_cmd)
                                    self.last_cmd = new_cmd
                                    self.last_send_time = current_time
                                else:
                                    print(f"❌ 放棄重試指令: {self.last_cmd}，強制放行。")
                                    self.ctx['is_cmd_acked'] = True
                            else:
                                if new_cmd != self.last_cmd:
                                    print(f"🔄 更新角度: {new_cmd}")
                                    self.ctx['bt'].send_action(new_cmd)
                                    self.last_cmd = new_cmd
                                    self.last_send_time = current_time
                                    self.ctx['is_cmd_acked'] = False
                                    self.retry_count = 0
        else:
            self.ctx['planner'].current_target = None

        hud_frame = self.ctx['visualizer'].draw_hud(frame, self.ctx['robot'], self.ctx['whiteboard'], self.ctx['planner'], robot_corners, dirty_rects, robot_mask_pts=robot_mask_pts)
        
        bt_status = "Connected" if self.ctx.get('bt') else "DISCONNECTED (Reconnecting...)"
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
            if self.ctx.get('bt'): self.ctx['bt'].send_action("S")
        elif key == ord('z'):
            print("\n📤 [Mode 0] 發送陀螺儀校準指令: Z")
            if self.ctx.get('bt'): self.ctx['bt'].send_action("Z")


# ==========================================
#  3. Mode 1: 純視覺除錯模式 (Pure Vision Debug Mode)
# ==========================================
class VisionDebugMode(BaseMode):
    def activate(self):
        print("\n👁️ [Mode 1] 已切換至純視覺除錯模式 (自動關閉藍牙)")
        self.ctx['bt_should_connect'] = False  # 通知背景執行緒斷開藍牙
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

    def handle_key(self, key):
        pass


# ==========================================
#  4. Mode 2: 測試遙控與精準角度模式
# ==========================================
class ManualControlMode(BaseMode):
    def __init__(self, shared_context):
        super().__init__(shared_context)
        self.target_angle = 0.0  # 測試用的自訂絕對角度變數

        # 【安全整合】保留藍牙重傳機制，已移除失明防暴走的相關計數器
        self.last_cmd = None
        self.last_send_time = 0
        self.RETRY_COOLDOWN = 0.5
        self.MAX_RETRIES = 3
        self.retry_count = 0

    def activate(self):
        print("\n🕹️ [Mode 2] 已切換至測試遙控模式！(無須標籤也可遙控)")
        print("操作說明 (請點擊影像視窗後按鍵):")
        print("  [W] 車子直走 (F)            | [X] 車子後退 (B)")
        print("  [S] 或 [空白鍵] 車子停止 (S)")
        print("  [I] 增加目標角度 (+5°)      | [K] 減少目標角度 (-5°)")
        print("  [L] 左轉捷徑至設定絕對角度  | [R] 右轉捷徑至設定絕對角度")
        print("  [Z] 陀螺儀校準 (Z)")
        self.ctx['bt_should_connect'] = True
        self.ctx['planner'].current_target = None

    def process_frame(self, frame):
        aruco_mask = self.ctx['vision'].get_aruco_ready_mask(frame, roi_polygon=self.ctx['roi_polygon'])
        robot_center, robot_corners = self.ctx['extractor'].extract_robot_pose(aruco_mask)
        ink_mask, robot_mask_pts = self.ctx['vision'].get_ink_clean_mask(frame, exclude_polygon=robot_corners,
                                                                         roi_polygon=self.ctx['roi_polygon'])
        dirty_rects = self.ctx['extractor'].extract_dirty_rects(ink_mask)

        if robot_center is not None:
            self.ctx['robot'].update_state(robot_center, robot_corners)
        self.ctx['whiteboard'].update_dirty_matrix(dirty_rects)

        # 【安全整合】藍牙指令重傳機制
        if self.last_cmd is not None and not self.ctx.get('is_cmd_acked', True):
            current_time = time.time()
            if current_time - self.last_send_time > self.RETRY_COOLDOWN:
                if self.retry_count < self.MAX_RETRIES:
                    self.retry_count += 1
                    print(f"⚠️ [Mode 2] 未收到車子確認，重發 ({self.retry_count}/{self.MAX_RETRIES}): {self.last_cmd}")
                    if self.ctx.get('bt'): self.ctx['bt'].send_action(self.last_cmd)
                    self.last_send_time = current_time
                else:
                    print(f"❌ [Mode 2] 放棄重試遙控指令: {self.last_cmd}")
                    self.ctx['is_cmd_acked'] = True

        hud_frame = self.ctx['visualizer'].draw_hud(frame, self.ctx['robot'], self.ctx['whiteboard'],
                                                    self.ctx['planner'], robot_corners, dirty_rects,
                                                    robot_mask_pts=robot_mask_pts)

        bt_status = "Connected" if self.ctx.get('bt') else "DISCONNECTED (Reconnecting...)"
        bt_color = (0, 255, 0) if self.ctx.get('bt') else (0, 0, 255)
        cv2.putText(hud_frame, f"MODE 2: TEST RC (No Tag Req.) | BT: {bt_status}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, bt_color, 2)

        angle_text = f"Target Abs Angle: {self.target_angle:.1f} deg (Press I/K to adjust)"
        cv2.putText(hud_frame, angle_text, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        self.ctx['visualizer'].show_windows(hud_frame, aruco_mask, ink_mask)

    def _send_manual_cmd(self, cmd):
        """封裝發送指令，觸發重傳與 ACK 檢查"""
        bt = self.ctx.get('bt')
        if not bt:
            print("⏳ 藍牙未連線，指令暫時無效...")
            return
        print(f"🕹️ 遙控發送: {cmd}")
        bt.send_action(cmd)

        self.last_cmd = cmd
        self.last_send_time = time.time()
        self.ctx['is_cmd_acked'] = False
        self.retry_count = 0

    def handle_key(self, key):
        if key in [ord('i'), ord('I')]:
            self.target_angle += 5.0
            if self.target_angle > 180.0: self.target_angle -= 360.0
            print(f"🔄 手動絕對角度設定為: {self.target_angle:.1f}°")
        elif key in [ord('k'), ord('K')]:
            self.target_angle -= 5.0
            if self.target_angle < -180.0: self.target_angle += 360.0
            print(f"🔄 手動絕對角度設定為: {self.target_angle:.1f}°")

        elif key in [ord('w'), ord('W')]:
            self._send_manual_cmd("F")
        elif key in [ord('x'), ord('X')]:  # 🌟 新增：按下 X 鍵後退
            self._send_manual_cmd("B")
        elif key in [ord('s'), ord('S'), 32]:
            self._send_manual_cmd("S")
        elif key in [ord('l'), ord('L')]:
            self._send_manual_cmd(f"L{self.target_angle:.1f}")
        elif key in [ord('r'), ord('R')]:
            self._send_manual_cmd(f"R{self.target_angle:.1f}")
        elif key in [ord('z'), ord('Z')]:
            self._send_manual_cmd("Z")
# ==========================================
#  邊界手動點擊函式
# ==========================================
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"📍 已選擇頂點 {len(clicked_points)}: ({x}, {y})")

def setup_roi_manually(cap):
    global clicked_points
    clicked_points = []
    
    cv2.namedWindow("Select ROI (Whiteboard Area)")
    cv2.setMouseCallback("Select ROI (Whiteboard Area)", mouse_callback)
    
    print("\n" + "=" * 40)
    print("👆 [系統初始化] 請設定白板邊界")
    print("請在彈出的視窗中，依序點擊 4 個點來框出白板範圍。")
    print("建議順序：左上 -> 右上 -> 右下 -> 左下")
    print("點完後按 'Enter' 或 '空白鍵' 確認。按 'r' 可以重新點擊。")
    print("=" * 40 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法讀取攝影機畫面，請檢查設備！")
            break
            
        # ❌ 【徹底拔除】不翻轉畫面，對齊物理世界與視覺世界！
        display_frame = frame.copy()
        
        for i, pt in enumerate(clicked_points):
            cv2.circle(display_frame, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if i > 0:
                cv2.line(display_frame, tuple(clicked_points[i-1]), tuple(clicked_points[i]), (0, 255, 0), 2)
        
        if len(clicked_points) == 4:
            cv2.line(display_frame, tuple(clicked_points[3]), tuple(clicked_points[0]), (0, 255, 0), 2)
            cv2.putText(display_frame, "Press ENTER to confirm", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Select ROI (Whiteboard Area)", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            clicked_points = []
            print("🔄 重新選擇")
        elif key == 13 or key == 32:  
            if len(clicked_points) == 4:
                print("✅ 邊界設定完成！")
                break
            else:
                print(f"⚠️ 還沒點完 4 個點喔！目前只點了 {len(clicked_points)} 個。")

    cv2.destroyWindow("Select ROI (Whiteboard Area)")
    return clicked_points


# ==========================================
#  5. 主程式控制中樞 (Main Context)
# ==========================================
def main():
    print("\n" + "=" * 40)
    print("🚗 智慧自動擦拭機器人 - 多核心架構系統")
    print("=" * 40)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    roi_polygon = setup_roi_manually(cap)
    
    if not roi_polygon or len(roi_polygon) != 4:
        print("❌ 未正確設定白板邊界，系統安全退出。")
        cap.release()
        return

    # 封裝全域共享上下文 (Context)
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
        'bt_should_connect': False  
    }

    # 🔥 啟動全域藍牙背景管理執行緒 🔥
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

    print("\n⌨️ 系統模式切換熱鍵：")
    print("👉 按 '0' 鍵 -> 切換至 [Mode 0: 完整自動控制模式]")
    print("👉 按 '1' 鍵 -> 切換至 [Mode 1: 純視覺除錯模式]")
    print("👉 按 '2' 鍵 -> 切換至 [Mode 2: 測試與角度遙控模式]")
    print("👉 按 'q' 鍵 -> 離開系統\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # ❌ 【徹底拔除】不翻轉畫面，對齊物理世界與視覺世界！
        # frame = cv2.flip(frame, 1)

        # 執行當前模式的每幀處理邏輯
        current_mode.process_frame(frame)

        # 鍵盤全域監聽與事件分流
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('0'), ord('1'), ord('2')]:
            target_idx = int(chr(key))
            if target_idx != current_mode_idx and target_idx in modes:
                # 若從模式 0 或 2 切換到其他模式，先送出緊急停止指令保護硬體
                if current_mode_idx in [0, 2] and shared_context.get('bt'):
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