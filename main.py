# main.py
import cv2
from vision import VisionManager
from extractor import FeatureExtractor
from robot import Robot
from whiteboard import Whiteboard
from planner import CleaningPlanner
from display import Visualizer
from modes import FullControlMode, VisionDebugMode, ManualControlMode
from config.system_settings import CAMERA_ID, BLUETOOTH_PORT, RES_SCALE, GRID_CELL_SIZE

try:
    from bluetooth import BTInterface
    BT_AVAILABLE = True
except ImportError:
    print("⚠️ 找不到 bluetooth.py，藍牙控制將無法正常運作。")
    BT_AVAILABLE = False

BLUETOOTH_PORT = '/dev/tty.usbserial-140'  # 請確認您的 COM Port
RES_SCALE = 1.5
CAMERA_ID = 0

# ==========================================
#  邊界手動點擊函式 (可於未來繼續拆分至 UI 模組)
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
#  主程式控制中樞
# ==========================================
def main():
    print("\n" + "=" * 40)
    print("🚗 智慧自動擦拭機器人 - 極致輕量化架構版")
    print("=" * 40)

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"❌ 無法開啟攝影機 ID: {CAMERA_ID}，請嘗試更改 CAMERA_ID 變數 (例如改為 1 或 2)")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(640 * RES_SCALE))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(480 * RES_SCALE))

    roi_polygon = setup_roi_manually(cap)
    if not roi_polygon or len(roi_polygon) != 4:
        cap.release()
        return

    bt_manager = None
    if BT_AVAILABLE:
        bt_manager = BTInterface(port=BLUETOOTH_PORT)

    # 上下文管理：將 res_scale 註冊進去，徹底消除外部依賴
    shared_context = {
        'res_scale': RES_SCALE,
        'vision': VisionManager(RES_SCALE),
        'extractor': FeatureExtractor(RES_SCALE),
        'robot': Robot(),
        'whiteboard': Whiteboard(width=int(640 * RES_SCALE), 
                                 height=int(480 * RES_SCALE), 
                                 cell_size=int(GRID_CELL_SIZE * RES_SCALE)),
        'planner': CleaningPlanner(RES_SCALE),
        'visualizer': Visualizer(roi_polygon),  
        'roi_polygon': roi_polygon,
        'bt': bt_manager                 
    }

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
                    shared_context['bt'].send_new_action("S")

                current_mode_idx = target_idx
                current_mode = modes[current_mode_idx]
                current_mode.activate()
        else:
            current_mode.handle_key(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()