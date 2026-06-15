# test_cam.py
import cv2

def test_cameras():
    print("🔍 正在尋找可用的攝影機 ID...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ 找到可用鏡頭！ID: {i}")
            cap.release()
        else:
            print(f"❌ ID: {i} 無法使用")

if __name__ == "__main__":
    test_cameras()