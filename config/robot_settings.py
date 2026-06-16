# config/robot_settings.py

# 板擦中心相對於 ArUco 標籤的偏移比例
ERASER_OFFSET_RATIO = 0.8  

# --- 車體自訂遮罩形狀比例 (相對於標籤長寬) ---
MASK_MOTOR_FWD = 1.80      # 馬達前緣凸出比例
MASK_WHEEL_START = -0.10   # 輪胎起點比例
MASK_BODY_BWD = 1.20       # 車體後緣凸出比例
MASK_NARROW_SIDE = 1.2    # 窄邊寬度比例 (不含輪)
MASK_WHEEL_SIDE = 1.5      # 寬邊寬度比例 (含輪)