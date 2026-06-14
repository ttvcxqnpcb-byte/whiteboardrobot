import numpy as np

class Whiteboard:
    def __init__(self, width=640, height=480, cell_size=20):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        self.cols = width // cell_size
        self.rows = height // cell_size
        self.dirty_list = []

    def update_dirty_matrix(self, dirty_rects):
        self.dirty_list.clear()
        
        for x, y, w, h, tx, ty in dirty_rects:
            center_x = tx
            center_y = ty
            
            if 0 <= center_x < self.width and 0 <= center_y < self.height:
                dirty_info = {
                    "cx": center_x, 
                    "cy": center_y,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
                self.dirty_list.append(dirty_info)

    def get_dirty_list(self):
        return self.dirty_list

    def get_dirty_count(self):
        return len(self.dirty_list)