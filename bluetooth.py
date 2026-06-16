import sys
import time
import threading
from bridge.hm10_esp32_bridge import HM10ESP32Bridge

class BTInterface:
    def __init__(self, port, expected_name="HM10_Mega"):
        self.port = port
        self.expected_name = expected_name
        self.bridge = None
        
        # --- 狀態管理 ---
        self.should_connect = False
        self.is_connected = False
        
        # --- ACK 驗證 ---
        self.is_cmd_acked = True
        self.pending_cmd = None
        self.is_action_finished = False
        
        # 初始化時自動啟動背景監聽與連線執行緒
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _connect_hardware(self):
        """處理硬體層的初始化與配對"""
        print(f"\n🔄 [藍牙背景] 嘗試連線至 {self.port}...")
        try:
            self.bridge = HM10ESP32Bridge(port=self.port)
            
            if self.bridge.get_hm10_name() != self.expected_name:
                print(f"正在將 ESP32 尋找目標設定為: {self.expected_name}...")
                self.bridge.set_hm10_name(self.expected_name)
                self.bridge.reset()
                time.sleep(3)  # 等待 ESP32 重啟
                self.bridge = HM10ESP32Bridge(port=self.port) 

            status = self.bridge.get_status()
            if status != "CONNECTED":
                raise ConnectionError("藍牙硬體連線失敗")

            self.is_connected = True
            self.is_cmd_acked = True
            print(f"\n✅ [藍牙背景] ✨ 成功連線至智慧車 ({self.expected_name})！")
        except Exception as e:
            self.bridge = None
            self.is_connected = False

    def _worker_loop(self):
        """背景執行緒：負責斷線重連與接收解析"""
        last_attempt_time = 0
        while True:
            if self.should_connect:
                if not self.is_connected:
                    # 避免瘋狂重試，每 3 秒嘗試一次
                    if time.time() - last_attempt_time > 3.0:
                        last_attempt_time = time.time()
                        self._connect_hardware()
                else:
                    try:
                        msg = self.bridge.listen()
                        if msg:
                            msg_str = str(msg).strip()
                            if msg_str:
                                print(f"\n🟢 [智慧車回傳]: {msg_str}")
                                self._verify_ack(msg_str)

                                if "finish" in msg_str.lower():
                                    print("✅ [狀態] 收到 finish")
                                    self.is_action_finished = True

                    except Exception as e:
                        print(f"\n❌ [藍牙背景] 連線異常中斷 ({e})，準備重新連線...")
                        self.is_connected = False
                        self.bridge = None
            else:
                # 若外部要求斷開，主動清空連線省電
                if self.is_connected:
                    self.bridge = None
                    self.is_connected = False
            
            time.sleep(0.05)

    def _verify_ack(self, msg_str):
        """絕對嚴格 ACK 驗證邏輯"""
        if self.pending_cmd and not self.is_cmd_acked:
            expected_c = self.pending_cmd[0].upper()
            expected_ang = float(self.pending_cmd[1:]) if len(self.pending_cmd) > 1 else 0.0
            
            lines = msg_str.split('\n')
            for line in reversed(lines):
                line = line.strip()
                
                if "Cmd:" in line and ",Ang:" in line:
                    try:
                        parts = line.split(",")
                        ret_c = parts[0].split(":")[1].strip()
                        ret_ang = float(parts[1].split(":")[1].strip())
                        
                        if ret_c == expected_c and abs(ret_ang - expected_ang) < 0.1:
                            self.is_cmd_acked = True
                            print(f"✅ [ACK 嚴格確認] 成功驗證指令: {self.pending_cmd}")
                            break
                    except Exception:
                        continue
                        
                elif expected_c == 'Z' and ('開始校準' in line or 'y' in line):
                    self.is_cmd_acked = True
                    print(f"✅ 陀螺儀校準已觸發！")
                    break
                elif expected_c == 'P' and 'ExtraMotor: ON' in line:
                    self.is_cmd_acked = True
                    print(f"✅ 板擦馬達已啟動！(P)")
                    break
                elif expected_c == 'Y' and 'ExtraMotor: OFF' in line:
                    self.is_cmd_acked = True
                    print(f"✅ 板擦馬達已關閉！(Y)")
                    break

    # ==========================================
    #  提供給外部 (main.py) 呼叫的 API 介面
    # ==========================================
    def enable_auto_connect(self):
        """通知背景執行緒開始連線"""
        self.should_connect = True

    def disable_auto_connect(self):
        """通知背景執行緒斷開連線"""
        self.should_connect = False

    def send_new_action(self, cmd):
        """發送新指令，並註冊至 pending 以啟動嚴格驗證機制"""
        self.pending_cmd = cmd
        self.is_cmd_acked = False
        self._send_raw(cmd)

    def resend_action(self):
        """重發目前尚未獲得確認的指令"""
        if self.pending_cmd:
            self._send_raw(self.pending_cmd)

    def _send_raw(self, cmd):
        """底層發送，負責加上換行符號"""
        if self.is_connected and self.bridge:
            # 【修正】加上 \r\n (Carriage Return + Line Feed)
            self.bridge.send(cmd + "\r\n")