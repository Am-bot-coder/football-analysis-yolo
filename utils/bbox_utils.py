def get_centre_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    centre_x = (x1 + x2) / 2
    centre_y = (y1 + y2) / 2
    return int(centre_x), int(centre_y)

def get_width_bbox(bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    return width