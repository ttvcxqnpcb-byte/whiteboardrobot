# config/vision_settings.py

# ArUco 標籤二值化閾值
ARUCO_THRESH = 103

# 尋找髒污的 GaussianBlur 基礎大小 (會乘上 RES_SCALE)
BLUR_BASE_SIZE = 5

# 尋找髒污的 AdaptiveThreshold 區塊大小 (會乘上 RES_SCALE)
BLOCK_BASE_SIZE = 11

# AdaptiveThreshold 亮度常數補償值
ADAPTIVE_C = 7