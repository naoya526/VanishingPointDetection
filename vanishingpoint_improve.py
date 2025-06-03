import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os

def calculate_angle(x1, y1, x2, y2):
    """ calclate two point's angle """
    dx, dy = x2 - x1, y2 - y1
    return np.arctan2(dy, dx) * 180 / np.pi

def extend_line(x1, y1, x2, y2, img_shape):
    """ 線を画像の端まで延長 """
    rows, cols = img_shape[:2]
    if x2 == x1:  # 垂直線
        return x1, 0, x1, rows
    
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    y_left, y_right = int(intercept), int(slope * cols + intercept)
    return 0, y_left, cols, y_right

def group_parallel_lines(lines, eps=2.0, min_samples=2):
    """ 並行線をクラスタリング """
    angles = np.array([[calculate_angle(*line[0])] for line in lines])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(angles)
    labels = clustering.labels_
    return labels, angles.flatten()

def detect_linear(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    return cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=60, maxLineGap=10)

def save_image(image, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"保存完了: {output_path}")

def calculate_intersection(line1, line2):
    """ 2つの線の交点を計算 """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # 傾きを計算
    if x2 != x1:  # 直線1の傾き
        m1 = (y2 - y1) / (x2 - x1)
    else:
        m1 = float('inf')  # 垂直線の場合
    
    if x4 != x3:  # 直線2の傾き
        m2 = (y4 - y3) / (x4 - x3)
    else:
        m2 = float('inf')  # 垂直線の場合
    
    # 傾きが等しい場合、交点は無い（平行な直線）
    if m1 == m2:
        return None
    
    # y切片を計算
    b1 = y1 - m1 * x1
    b2 = y3 - m2 * x3
    
    # 交点を計算
    if m1 != m2:
        # 交点のx座標
        x = (b2 - b1) / (m1 - m2)
        
        # 交点のy座標
        y = m1 * x + b1
        
        # 交点が画像の範囲内かチェック
        if 0 <= x < 640 and 0 <= y < 480:  # 画像サイズに合わせて調整
            return (int(x), int(y))
    
    return None

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像読み込み失敗: {image_path}")
        return
    
    lines = detect_linear(img)
    if lines is None:
        print("直線検出なし")
        return
    
    line_img1 = np.zeros_like(img)
    for line in lines:
        cv2.line(line_img1, tuple(line[0][:2]), tuple(line[0][2:]), (255, 255, 255), 2)
    save_image(line_img1, "output/detected_lines.jpg")
    
    labels, angles = group_parallel_lines(lines)
    
    # グループ化された線の角度の標準偏差を用いてスレッショルドを決定
    angle_std = np.std(angles)
    angle_threshold = max(2.0, angle_std * 0.5)  # 標準偏差の50%をしきい値に設定
    
    filtered_img = np.zeros_like(img)
    extended_img = np.zeros_like(img)
    
    for line, angle in zip(lines, angles):
        if angle_threshold < abs(angle) < (90 - angle_threshold):
            cv2.line(filtered_img, tuple(line[0][:2]), tuple(line[0][2:]), (255, 255, 255), 2)
            x1_ext, y1_ext, x2_ext, y2_ext = extend_line(*line[0], img.shape)
            cv2.line(extended_img, (x1_ext, y1_ext), (x2_ext, y2_ext), (255, 255, 255), 2)
    
    save_image(filtered_img, "output/filtered_lines.jpg")
    save_image(extended_img, "output/extended_lines.jpg")
    
    # 元の画像にextended_linesを重ねる
    overlayed_img = cv2.addWeighted(img, 0.7, extended_img, 0.3, 0)
    save_image(overlayed_img, "output/overlayed_extended_lines.jpg")

if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(os.path.join(input_dir, filename))
