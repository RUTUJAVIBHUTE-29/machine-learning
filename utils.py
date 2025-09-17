import json
import numpy as np
import cv2

def load_nutrition_db(path="nutrition_db.json"):
    with open(path,"r") as f:
        return json.load(f)

def estimate_calories(mass_g, kcal_per_100g):
    return (mass_g * kcal_per_100g) / 100.0

def estimate_mass(area_cm2, thickness_cm, density_g_per_cm3):
    volume_cm3 = area_cm2 * thickness_cm
    mass_g = volume_cm3 * density_g_per_cm3
    return mass_g

def pixel_to_cm2(marker_pixel_area, marker_real_width_cm, marker_real_height_cm):
    real_area_cm2 = marker_real_width_cm * marker_real_height_cm
    return real_area_cm2 / marker_pixel_area

def mask_area_pixels(mask_binary):
    return int(np.count_nonzero(mask_binary))

def find_marker_bbox_by_color(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0,100,100]); upper1 = np.array([10,255,255])
    lower2 = np.array([160,100,100]); upper2 = np.array([179,255,255])
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1,m2)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    if w*h < 100: return None
    return (x,y,w,h)
