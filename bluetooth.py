import sys
import time
from bridge.hm10_esp32_bridge import HM10ESP32Bridge

class BTInterface:
    def __init__(self, port, expected_name="HM10_Mega"):
        print(f"正在透過 {port} 初始化 ESP32...")
        self.bridge = HM10ESP32Bridge(port=port)

        if self.bridge.get_hm10_name() != expected_name:
            print(f"正在將 ESP32 尋找目標設定為: {expected_name}...")
            self.bridge.set_hm10_name(expected_name)
            self.bridge.reset()
            print("等待 ESP32 重啟並連線...")
            time.sleep(3)  # 等待 ESP32 重啟
            self.bridge = HM10ESP32Bridge(port=port) 

        status = self.bridge.get_status()
        if status != "CONNECTED":
            print(f"[錯誤] 藍牙狀態: {status}。無法連線至 {expected_name}！")
            print("請檢查：1. 車子是否開機? 2. 藍牙名稱是否正確? 3. 距離是否太遠?")
            # 拋出例外，讓 main.py 的 try-except 可以接住並退回視覺模式
            raise ConnectionError("藍牙硬體連線失敗")

        print(f"✨ 成功連線至智慧車 ({expected_name})！")

    def get_status(self):
        return self.bridge.get_status()

    def get_name(self):
        return self.bridge.get_hm10_name()

    def set_name(self, name):
        self.bridge.set_hm10_name(name)
        self.bridge.reset()

    def send_action(self, cmd):
        # 隊友的 Arduino 程式碼需要換行符號 ('\n') 來判斷指令結束
        self.bridge.send(cmd + "\n")

    def listen(self):
        return self.bridge.listen()