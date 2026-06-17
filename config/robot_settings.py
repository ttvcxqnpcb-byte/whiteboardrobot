# config/robot_settings.py

# 板擦中心相對於 ArUco 標籤的偏移比例
ERASER_OFFSET_RATIO = 0.8  

MARKER_SIZE = 1.0  

# 車體 3D 長方體相對於標籤中心的邊界比例
ROBOT_3D_FRONT  = 1.7   
ROBOT_3D_BACK   = 1.3   
ROBOT_3D_LEFT   = 1.2   
ROBOT_3D_RIGHT  = 1.2   
ROBOT_3D_TOP    = 1.1   
ROBOT_3D_BOTTOM = 0.1   

# ======================
#  (Movement & Safety)
# ======================
# 安全邊界距離 (以原始解析度為基準，會乘上 RES_SCALE)
SAFE_MARGIN_BASE = 10
# 抵達目標的距離寬容值
ARRIVAL_DIST_BASE = 15
# 回到基地時，判定車頭對齊的容許角度誤差
HOME_ANGLE_TOLERANCE = 10
# 判定需要直線後退的觸發角度 (大於此角度才後退)
BACKWARD_ANGLE_THRESH = 165
# 判定需要原地旋轉的觸發角度 (大於此角度才旋轉)
TURN_ANGLE_THRESH = 15

# ==========================
#   (Vision & Camera)
# ==========================
# 偽造相機焦距倍率 (用於 3D 遮罩透視投影，值越大透視變形越小)
CAMERA_FOCAL_MULTIPLIER = 2.5

# ==========================
#   (Failsafe & Comp.)
# ==========================
# 容許連續丟失標籤的最大幀數 
MAX_LOST_FRAMES = 5
# 視覺代償：判定指令已實際作動的移動距離 (px)
VISUAL_COMP_DIST_BASE = 10
# 視覺代償：判定指令已實際作動的旋轉角度 (度)
VISUAL_COMP_ANGLE = 5.

# ==========================
#   (Planner & Grid)
# ==========================
# 板擦有效覆蓋寬度 (px)，用於將大塊髒污網格化切碎
ERASER_ARUCO_RATIO = 0.3
# 頑固污漬重試次數上限 (二階段複檢的補刀次數)
MAX_RESETS = 1
# 目標攻堅超時時間 - 秒 (解決幽靈目標卡死)
WATCHDOG_TIMEOUT = 8.0