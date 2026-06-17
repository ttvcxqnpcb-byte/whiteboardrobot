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
    print(" 找不到 bluetooth.py，藍牙控制將無法正常運作。")
    BT_AVAILABLE = False


# ==========================================
#  一體化 UI：邊界設定與字跡保留區框選
# ==========================================
setup_state = 'ROI'
roi_points = []
exclude_bboxes = []
drawing_box = False
box_start = (-1, -1)
current_box = None

def mouse_callback(event, x, y, flags, param):
    global setup_state, roi_points, exclude_bboxes, drawing_box, box_start, current_box

    # 【階段一】點擊設定白板 4 個角
    if setup_state == 'ROI':
        if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
            roi_points.append([x, y])
            print(f"📍 白板頂點 {len(roi_points)}: ({x}, {y})")

    # 【階段二】拖曳框選字跡保留區
    elif setup_state == 'KEEPOUT':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_box = True
            box_start = (x, y)
            current_box = (x, y, 0, 0)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing_box:
                current_box = (box_start[0], box_start[1], x - box_start[0], y - box_start[1])
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing_box:
                drawing_box = False
                w = x - box_start[0]
                h = y - box_start[1]
                # 確保反向拖曳也能畫出正確的矩形 (將長寬轉為正整數)
                bx, by = min(box_start[0], x), min(box_start[1], y)
                bw, bh = abs(w), abs(h)
                if bw > 5 and bh > 5: # 避免手抖誤點產生無效框
                    exclude_bboxes.append((bx, by, bw, bh))
                    print(f"🛡️ 新增字跡保留區: 座標({bx}, {by}), 大小 {bw}x{bh}")
                current_box = None

def unified_setup_ui(cap):
    global setup_state, roi_points, exclude_bboxes, current_box
    setup_state = 'ROI'
    roi_points = []
    exclude_bboxes = []
    current_box = None
    
    window_name = "Smart Whiteboard Setup"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n" + "=" * 40)
    print("🟢 步驟一：設定白板邊界")
    print("請在畫面上依序點擊四個角 (左上、右上、右下、左下)")
    print("=" * 40)

    while True:
        ret, frame = cap.read()
        if not ret: break
        display_frame = frame.copy()

        # 繪製白板邊界 (ROI)
        for i, pt in enumerate(roi_points):
            cv2.circle(display_frame, tuple(pt), 5, (0, 0, 255), -1)
            if i > 0: 
                cv2.line(display_frame, tuple(roi_points[i-1]), tuple(roi_points[i]), (0, 255, 0), 2)
        if len(roi_points) == 4:
            cv2.line(display_frame, tuple(roi_points[3]), tuple(roi_points[0]), (0, 255, 0), 2)

        # 繪製已經畫好的保留區 (Exclude Boxes)
        for (ex, ey, ew, eh) in exclude_bboxes:
            cv2.rectangle(display_frame, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
            cv2.putText(display_frame, "KEEP OUT", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 繪製滑鼠正在拖曳中的暫存黃框
        if current_box is not None and setup_state == 'KEEPOUT':
            cx, cy, cw, ch = current_box
            x1, y1 = cx, cy
            x2, y2 = cx + cw, cy + ch
            cv2.rectangle(display_frame, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), (0, 255, 255), 2)

        # 顯示 UI 提示文字
        if setup_state == 'ROI':
            cv2.putText(display_frame, "[Step 1] Click 4 points for Whiteboard.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if len(roi_points) == 4:
                cv2.putText(display_frame, "Press 'Enter' to confirm boundary.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif setup_state == 'KEEPOUT':
            cv2.putText(display_frame, "[Step 2] Drag to select specific handwriting.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'Enter' to finish and START.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 'Z' to undo last box.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r') and setup_state == 'ROI': 
            roi_points = []
            print("🔄 重新選擇白板邊界")
        elif key == ord('z') and setup_state == 'KEEPOUT':
            if exclude_bboxes: 
                removed = exclude_bboxes.pop()
                print(f"↩️ 已復原上一個保留區: {removed}")
        elif key == 13 or key == 32: # Enter 或 Space
            # 判斷切換階段
            if setup_state == 'ROI' and len(roi_points) == 4:
                setup_state = 'KEEPOUT'
                print("\n" + "=" * 40)
                print("🛡️ 步驟二：框選字跡保留區 (不擦拭)")
                print("請直接在畫面上【拖曳滑鼠】框出不要擦的字跡。可以畫多個。")
                print("畫錯可以按 'Z' 復原。全部選完後請按 [Enter] 正式啟動系統。")
                print("=" * 40)
            elif setup_state == 'KEEPOUT':
                break # 設定全部完成，跳出迴圈
                
    cv2.destroyWindow(window_name)
    cv2.waitKey(1) # 強制刷新 UI 避免視窗殘留卡死
    return roi_points, exclude_bboxes

# ==========================================
#  主程式控制中樞
# ==========================================
def main():
    print("\n" + "=" * 40)
    print("🚗 智慧自動擦拭機器人 - 單視窗一體化 UI 升級版")
    print("=" * 40)

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"❌ 無法開啟攝影機 ID: {CAMERA_ID}，請嘗試更改 CAMERA_ID 變數 (例如改為 1 或 2)")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(640 * RES_SCALE))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(480 * RES_SCALE))

    # 🌟 [核心修改] 呼叫一體化 UI 函式，同時取得邊界與保留區
    roi_polygon, exclude_bboxes = unified_setup_ui(cap)
    
    if not roi_polygon or len(roi_polygon) != 4:
        cap.release()
        print("❌ 設定未完成，系統結束。")
        return

    bt_manager = None
    if BT_AVAILABLE:
        bt_manager = BTInterface(port=BLUETOOTH_PORT)

    # 上下文管理
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
        'exclude_bboxes': exclude_bboxes, # 🌟 將收集到的所有字跡保留區傳入核心
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