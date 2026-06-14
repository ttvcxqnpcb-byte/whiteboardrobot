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
    print("⚠️ 找不到 bluetooth.py，Full 模式將無法正常發送指令。")
    BT_AVAILABLE = False

BLUETOOTH_PORT = 'COM9'


# ==========================================
#  1. 模式基底類別 (Strategy Interface)
# ==========================================
class BaseMode(ABC):
    """
    所有系統模式的基底類別。
    未來若要新增任何模式（例如自動校準、定點測試），只需繼承此類別並實作方法。
    """
    def __init__(self, shared_context):
        self.ctx = shared_context  # 共享的系統元件 (攝影機、機器人狀態、白板等)

    @abstractmethod
    def activate(self):
        """當切換進入此模式時會觸發一次的初始化邏輯"""
        pass

    @abstractmethod
    def process_frame(self, frame):
        """核心主迴圈邏輯：每張影像進來時該做什麼"""
        pass

    @abstractmethod
    def handle_key(self, key):
        """此模式下專屬的按鍵回應邏輯"""
        pass


# ==========================================
#  2. Mode 0: 完整控制模式 (Full System Mode)
# ==========================================
class FullControlMode(BaseMode):
    def __init__(self, shared_context):
        super().__init__(shared_context)
        self.bt = None
        self.is_cleaning = False
        self.last_cmd = None
        self.last_send_time = 0
        self.is_cmd_acked = True
        self.RETRY_COOLDOWN = 0.5

    def _background_listener(self):
        """持續接收車子 Arduino 傳回來的訊息"""
        while self.bt is not None:
            try:
                msg = self.bt.listen()
                if msg:
                    print(f"\n🟢 [車子回傳]: {msg}")
                    self.is_cmd_acked = True
            except Exception:
                pass
            time.sleep(0.1)

    def activate(self):
        print("\n📡 [Mode 0] 正在初始化藍牙控制模組...")
        if not BT_AVAILABLE:
            print("❌ 錯誤：未偵測到藍牙模組檔案，無法啟動完整控制。")
            return

        if self.bt is None:
            print(f"正在與車子 (Port: {BLUETOOTH_PORT}) 建立藍牙連線...")
            while self.bt is None:
                try:
                    self.bt = BTInterface(port=BLUETOOTH_PORT)
                    print("✅ 藍牙連線成功！大腦已與車體連線。")
                    listener_thread = threading.Thread(target=self._background_listener, daemon=True)
                    listener_thread.start()
                except Exception as e:
                    print(f"⚠️ 連線失敗 ({e})。3秒後自動重試...")
                    time.sleep(3)

    def process_frame(self, frame):
        # 影像處理解析
        aruco_mask = self.ctx['vision'].get_aruco_ready_mask(frame, roi_polygon=self.ctx['roi_polygon'])
        robot_center, robot_corners = self.ctx['extractor'].extract_robot_pose(aruco_mask)
        
        # 取得優化後的筆跡遮罩與除錯用的紫色框頂點
        ink_mask, robot_mask_pts = self.ctx['vision'].get_ink_clean_mask(frame, exclude_polygon=robot_corners, roi_polygon=self.ctx['roi_polygon'])
        dirty_rects = self.ctx['extractor'].extract_dirty_rects(ink_mask)

        if robot_center is not None:
            self.ctx['robot'].update_state(robot_center, robot_corners)
        self.ctx['whiteboard'].update_dirty_matrix(dirty_rects)

        # 自動導航擦拭邏輯
        if self.is_cleaning:
            dirty_list = self.ctx['whiteboard'].get_dirty_list()
            target = self.ctx['planner'].plan_next_target(dirty_list, self.ctx['robot'].x, self.ctx['robot'].y) if robot_center is not None else None

            if target is not None:
                delta_angle, pixel_dist = self.ctx['planner'].get_relative_movement(
                    self.ctx['robot'].x, self.ctx['robot'].y, self.ctx['robot'].angle, target[0], target[1]
                )
                target_abs_angle = (self.ctx['robot'].angle + delta_angle) % 360

                if pixel_dist < 30:
                    new_cmd = "S"
                    self.ctx['planner'].mark_as_visited(target[0], target[1])
                    self.ctx['planner'].current_target = None
                elif abs(delta_angle) > 15:
                    direction = "R" if delta_angle > 0 else "L"
                    new_cmd = f"{direction}{target_abs_angle:.1f}"
                else:
                    new_cmd = "F"

                # 藍牙發送與節流控制
                current_time = time.time()
                if new_cmd != self.last_cmd:
                    print(f"📤 發送新指令: {new_cmd} (距離: {pixel_dist:.1f}, 誤差角: {delta_angle:.1f})")
                    if self.bt: self.bt.send_action(new_cmd)
                    self.last_cmd = new_cmd
                    self.last_send_time = current_time
                    self.is_cmd_acked = False
                elif not self.is_cmd_acked:
                    if current_time - self.last_send_time > self.RETRY_COOLDOWN:
                        print(f"⚠️ 尚未收到確認，重發指令: {new_cmd}")
                        if self.bt: self.bt.send_action(new_cmd)
                        self.last_send_time = current_time
        else:
            self.ctx['planner'].current_target = None

        # 繪製 HUD 主畫面 (傳入除錯框頂點)
        hud_frame = self.ctx['visualizer'].draw_hud(
            frame, self.ctx['robot'], self.ctx['whiteboard'], self.ctx['planner'], 
            robot_corners, dirty_rects, robot_mask_pts=robot_mask_pts
        )
        
        # 覆蓋目前模式文字
        mode_text = "MODE 0: FULL SYSTEM (Running)" if self.is_cleaning else "MODE 0: FULL SYSTEM (Standby)"
        cv2.putText(hud_frame, mode_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        self.ctx['visualizer'].show_windows(hud_frame, aruco_mask, ink_mask)

    def handle_key(self, key):
        if key == ord('s'):
            print("\n▶️ [Mode 0] 開始自動擦拭")
            self.is_cleaning = True
        elif key == ord('p'):
            print("\n⏸️ [Mode 0] 暫停待機")
            self.is_cleaning = False
            if self.bt: self.bt.send_action("S")
        elif key == ord('z'):
            print("\n📤 [Mode 0] 發送陀螺儀校準指令: Z")
            if self.bt: self.bt.send_action("Z")


# ==========================================
#  3. Mode 1: 純視覺除錯模式 (Pure Vision Debug Mode)
# ==========================================
class VisionDebugMode(BaseMode):
    """
    完全不依賴藍牙硬體，專門用來實時觀察並調校非等比例方向性遮罩的範圍。
    """
    def activate(self):
        print("\n👁️ [Mode 1] 已切換至純視覺除錯模式 (免連線藍牙)")
        print("👉 請在此模式下微調 vision.py 的延伸引數，觀察紫紅色 Mask 是否完美覆蓋車體。")
        self.ctx['planner'].current_target = None # 清除導航殘留目標

    def process_frame(self, frame):
        # 影像解析 (與控制模式共用同一套演算法)
        aruco_mask = self.ctx['vision'].get_aruco_ready_mask(frame, roi_polygon=self.ctx['roi_polygon'])
        robot_center, robot_corners = self.ctx['extractor'].extract_robot_pose(aruco_mask)
        
        # 取得黑白 Mask 以及除錯用的車體方向性四角頂點
        ink_mask, robot_mask_pts = self.ctx['vision'].get_ink_clean_mask(frame, exclude_polygon=robot_corners, roi_polygon=self.ctx['roi_polygon'])
        dirty_rects = self.ctx['extractor'].extract_dirty_rects(ink_mask)

        if robot_center is not None:
            self.ctx['robot'].update_state(robot_center, robot_corners)
        self.ctx['whiteboard'].update_dirty_matrix(dirty_rects)

        # 繪製 HUD，並將除錯框渲染在畫面上
        hud_frame = self.ctx['visualizer'].draw_hud(
            frame, self.ctx['robot'], self.ctx['whiteboard'], self.ctx['planner'], 
            robot_corners, dirty_rects, robot_mask_pts=robot_mask_pts
        )
        
        # 顯示當前模式提示
        cv2.putText(hud_frame, "MODE 1: PURE VISION DEBUG (No BT)", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        self.ctx['visualizer'].show_windows(hud_frame, aruco_mask, ink_mask)

    def handle_key(self, key):
        # 除錯模式下不需要 s, p, z 的反應，保留給未來特殊除錯按鍵擴充
        pass


# ==========================================
#  4. 主程式控制中樞 (Main Context)
# ==========================================
def main():
    print("\n" + "=" * 40)
    print("🚗 智慧自動擦拭機器人 - 多核心架構系統")
    print("=" * 40)

    # 初始化公用硬體與演算法物件
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    roi_polygon = [[60, 50], [580, 50], [580, 430], [60, 430]]

    # 封裝全域共享上下文 (Context)
    shared_context = {
        'vision': VisionManager(),
        'extractor': FeatureExtractor(),
        'robot': Robot(),
        'whiteboard': Whiteboard(),
        'planner': CleaningPlanner(),
        'visualizer': Visualizer(roi_polygon),
        'roi_polygon': roi_polygon
    }

    # 註冊所有可用的模式 (高擴充性)
    # 未來若有 Mode 2，只需在這裡實例化並加入字典： 2: CalibrationMode(shared_context)
    modes = {
        0: FullControlMode(shared_context),
        1: VisionDebugMode(shared_context)
    }

    # 預設啟動模式 (您可以根據需求改成預設為 1，免去開機連不上一動也不動的困擾)
    current_mode_idx = 1 
    current_mode = modes[current_mode_idx]
    current_mode.activate()

    print("\n⌨️ 系統模式切換熱鍵：")
    print("👉 按 '0' 鍵 -> 切換至 [Mode 0: 完整控制模式]")
    print("👉 按 '1' 鍵 -> 切換至 [Mode 1: 純視覺除錯模式]")
    print("👉 按 'q' 鍵 -> 離開系統\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # 執行當前模式的每幀處理邏輯
        current_mode.process_frame(frame)

        # 鍵盤全域監聽與事件分流
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('0'), ord('1')]:
            # 動態切換模式
            target_idx = int(chr(key))
            if target_idx != current_mode_idx and target_idx in modes:
                current_mode_idx = target_idx
                current_mode = modes[current_mode_idx]
                current_mode.activate()
        else:
            # 將其他案件（例如 s, p, z）派發給當前模式內部的處理機制
            current_mode.handle_key(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()