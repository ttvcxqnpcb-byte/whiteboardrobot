# config/robot_settings.py

# 板擦中心相對於 ArUco 標籤的偏移比例 (保留)
ERASER_OFFSET_RATIO = 0.8  


MARKER_SIZE = 1.0  

# 車體 3D 長方體相對於標籤中心的邊界比例 (可依實體狀況微調，寧可稍大)
ROBOT_3D_FRONT  = 1.7   # 車頭往前凸出多少
ROBOT_3D_BACK   = 1.5   # 車尾往後凸出多少
ROBOT_3D_LEFT   = 1.2   # 車身往左凸出多少
ROBOT_3D_RIGHT  = 1.2   # 車身往右凸出多少
ROBOT_3D_TOP    = 0.5   # 車頂高度 (最重要！決定傾斜時遮罩往外擴展的範圍)
ROBOT_3D_BOTTOM = 0.0   # 車底貼地