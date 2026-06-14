# main.py
import cv2
import numpy as np
import time
import threading  # [新增] 用於背景監聽藍牙訊息
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
    print("⚠️ 找不到 bluetooth.py，將以純視覺模式運行。")
    BT_AVAILABLE = False

# ================= [ 使用者設定區 ] =================
# 請修改為你的藍牙 Port (例如 Windows 是 'COM9', Mac 是 '/dev/cu.usbserial-xxx')
BLUETOOTH_PORT = 'COM9'

is_cmd_acked = True
# ==================================================


def background_listener(bt_interface):
    """【新增】在背景持續接收並印出車子 Arduino 傳回來的訊息"""
    while True:
        try:
            msg = bt_interface.listen()
            if msg:
                print(f"\n🟢 [車子回傳]: {msg}")
                is_cmd_acked = True
        except Exception:
            pass
        time.sleep(0.1)


def main():
    print("\n" + "=" * 40)
    print("🚗 智慧自動擦拭機器人 - 系統啟動")
    print("=" * 40)

    bt = None

    # --- [階段 1] 不斷嘗試連線藍牙 ---
    if BT_AVAILABLE:
        print(f"\n📡 開始嘗試與車子 (Port: {BLUETOOTH_PORT}) 建立藍牙連線...")
        print("💡 請確認車子電源已開啟，且藍牙燈號正在閃爍或已連線。")

        while bt is None:
            try:
                bt = BTInterface(port=BLUETOOTH_PORT)
                print("\n✅ 藍牙連線成功！大腦已與車體連線。")

                # 連線成功後，啟動背景監聽執行緒來接收車子回饋
                listener_thread = threading.Thread(target=background_listener, args=(bt,), daemon=True)
                listener_thread.start()
                break  # 成功連線，跳出等待迴圈

            except Exception as e:
                print(f"⚠️ 連線失敗 ({e})。3秒後自動重試...")
                time.sleep(3)
    # ----------------------------------

    print("\n📷 正在啟動影像辨識模組與攝影機...")
    time.sleep(1)  # 給一點緩衝時間

    # --- [階段 2] 啟動影像辨識與控制 ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    vision = VisionManager()
    extractor = FeatureExtractor()
    my_robot = Robot()
    whiteboard = Whiteboard()
    planner = CleaningPlanner()

    roi_x1, roi_y1 = 60, 50
    roi_x2, roi_y2 = 580, 430
    roi_polygon = [[roi_x1, roi_y1], [roi_x2, roi_y1], [roi_x2, roi_y2], [roi_x1, roi_y2]]

    display = Visualizer(roi_polygon)

    # 系統狀態與藍牙節流閥設定
    is_cleaning = False
    last_cmd = None
    last_send_time = 0
    SEND_COOLDOWN = 0.2

    print("\n✨ 系統準備就緒，目前為【待機模式】。")
    print("👉 按 's' 鍵 -> 開始擦拭 (自動追蹤並控制車子)")
    print("👉 按 'p' 鍵 -> 暫停擦拭 (車子停止)")
    print("👉 按 'z' 鍵 -> 陀螺儀校準")
    print("👉 按 'q' 鍵 -> 離開程式\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        aruco_mask = vision.get_aruco_ready_mask(frame, roi_polygon=roi_polygon)
        robot_center, robot_corners = extractor.extract_robot_pose(aruco_mask)

        ink_mask = vision.get_ink_clean_mask(frame, exclude_polygon=robot_corners, roi_polygon=roi_polygon)
        dirty_rects = extractor.extract_dirty_rects(ink_mask)

        if robot_center is not None:
            my_robot.update_state(robot_center, robot_corners)

        whiteboard.update_dirty_matrix(dirty_rects)

        if is_cleaning:
            dirty_list = whiteboard.get_dirty_list()

            if robot_center is not None:
                target = planner.plan_next_target(dirty_list, my_robot.x, my_robot.y)
            else:
                target = None
                planner.current_target = None

            if target is not None:
                delta_angle, pixel_dist = planner.get_relative_movement(
                    my_robot.x, my_robot.y, my_robot.angle, target[0], target[1]
                )

                target_abs_angle = (my_robot.angle + delta_angle) % 360

                if pixel_dist < 30:
                    new_cmd = "S"
                    planner.mark_as_visited(target[0], target[1])
                    planner.current_target = None
                elif abs(delta_angle) > 15:
                    direction = "R" if delta_angle > 0 else "L"
                    new_cmd = f"{direction}{target_abs_angle:.1f}"
                else:
                    new_cmd = "F"

                current_time = time.time()
                if (new_cmd != last_cmd):
                    print(f"📤 發送新指令: {new_cmd} (距離: {pixel_dist:.1f}, 誤差角: {delta_angle:.1f})")
                    if bt is not None:
                        try:
                            bt.send_action(new_cmd)
                        except:
                            pass
                    last_cmd = new_cmd
                    last_send_time = current_time
                    is_cmd_acked = False
                elif not is_cmd_acked:
                    if (current_time - last_send_time > RETRY_COOLDOWN):
                        print(f"⚠️ 尚未收到確認，重發指令: {new_cmd}")
                        if bt is not None:
                            try:
                                bt.send_action(new_cmd)
                            except:
                                pass
                        last_send_time = current_time
        else:
            planner.current_target = None

        # 顯示層
        hud_frame = display.draw_hud(frame, my_robot, whiteboard, planner, robot_corners, dirty_rects)

        mode_text = "MODE: CLEANING (Running)" if is_cleaning else "MODE: STANDBY (Press 's' to start)"
        color = (0, 0, 255) if is_cleaning else (0, 255, 255)
        cv2.putText(hud_frame, mode_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        display.show_windows(hud_frame, aruco_mask, ink_mask)

        # 鍵盤監聽
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            print("\n📤 [System] 發送陀螺儀校準指令: Z")
            if bt is not None:
                bt.send_action("Z")
        elif key == ord('s'):
            print("\n▶️ [System] 模式切換：開始擦拭")
            is_cleaning = True
        elif key == ord('p'):
            print("\n⏸️ [System] 模式切換：暫停待機")
            is_cleaning = False
            if bt is not None:
                bt.send_action("S")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()