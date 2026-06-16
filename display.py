# display.py
import cv2
import numpy as np

class Visualizer:
    def __init__(self, roi_polygon):
        self.roi_pts = np.array([roi_polygon], dtype=np.int32)

    def draw_hud(self, frame, robot, whiteboard, planner, aruco_corners, dirty_rects, robot_mask_pts=None):        
        overlay = frame.copy()
        cv2.fillPoly(overlay, self.roi_pts, (0, 255, 0))
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, self.roi_pts, isClosed=True, color=(0, 255, 0), thickness=2)

        for x, y, w, h, tx, ty in dirty_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (tx, ty), 3, (0, 0, 255), -1)

        if planner.task_queue or planner.current_target:
            pts = []
            if planner.current_target:
                pts.append(planner.current_target)
            pts.extend(planner.task_queue)
            
            # 從車頭連一條線到第一個目標點
            if robot.x is not None and robot.y is not None and len(pts) > 0:
                cv2.line(frame, (robot.x, robot.y), pts[0], (0, 255, 255), 2)

            # 將序列中的任務點用線連起來，並畫出網格點
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i+1], (255, 100, 100), 2)
                cv2.circle(frame, pts[i], 4, (255, 200, 0), -1)
            
            if len(pts) > 0:
                cv2.circle(frame, pts[-1], 4, (255, 200, 0), -1)
                
        if planner.current_target and robot.x is not None and robot.y is not None:
            tx, ty = planner.current_target
            cv2.line(frame, (robot.x, robot.y), (tx, ty), (0, 255, 255), 2)
            cv2.drawMarker(frame, (tx, ty), (0, 165, 255), cv2.MARKER_CROSS, 15, 2)

        if aruco_corners is not None:
            cv2.polylines(frame, [np.array(aruco_corners, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.circle(frame, (robot.aruco_x, robot.aruco_y), 4, (255, 0, 0), -1)
            cv2.circle(frame, (robot.x, robot.y), 4, (0, 255, 255), -1)
            cv2.line(frame, (robot.aruco_x, robot.aruco_y), (robot.x, robot.y), (255, 255, 255), 1)
        
        if robot_mask_pts is not None:
            cv2.polylines(frame, [robot_mask_pts], isClosed=True, color=(255, 0, 255), thickness=2)
            cv2.putText(frame, "Mask Area", tuple(robot_mask_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        status_text = f"Pos:({robot.x},{robot.y}) Ang:{robot.angle:.1f} | Dirty Cells: {whiteboard.get_dirty_count()}"
        cv2.putText(frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if planner.current_target:
            target_text = f"Target: ({tx}, {ty})"
        else:
            target_text = "Target: None (Searching...)"
        cv2.putText(frame, target_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        return frame

    def show_windows(self, main_frame, aruco_mask, ink_mask):
        cv2.imshow('Main (Color HUD)', main_frame)
        cv2.imshow('Robot Mask (Debug)', aruco_mask)   
        cv2.imshow('Ink Mask (Debug)', ink_mask)