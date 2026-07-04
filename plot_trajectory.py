import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 配置（改成你的路径）=====
video_path = "/home/featurize/work/TrackNet_dataset_frame/"/rally1.mp4"          # test.py 输出的视频
csv_path = "output/rally1_ball.csv"       # 轨迹坐标
save_img = "paper_like_compare.png"        # 输出对比图

# 加载坐标
df = pd.read_csv(csv_path)
X_pred = df['X'].values
Y_pred = df['Y'].values
vis_pred = df['Visibility'].values

# 打开视频
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 取中间 30 帧做可视化（论文风格）
start = 50
end = start + 30

# 画轨迹
trail_len = 8
frames = []
for i in range(start, end):
    ret, frame = cap.read()
    if not ret:
        break
    # 画历史轨迹（淡蓝线）
    for j in range(max(0, i-trail_len), i):
        if vis_pred[j] == 1:
            cv2.line(frame,
                     (int(X_pred[j]), int(Y_pred[j])) ,
                     (int(X_pred[j+1]), int(Y_pred[j+1])) ,
                     (255,200,0), 2)
    # 当前点（红色）
    if vis_pred[i] == 1:
        cv2.circle(frame, (int(X_pred[i]), int(Y_pred[i])), 5, (0,0,255), -1)
    frames.append(frame)

cap.release()

# 拼成论文风格对比图（4帧一行）
rows = 3
cols = 10
fig, axes = plt.subplots(rows, cols, figsize=(20, 6))
idx = 0
for r in range(rows):
    for c in range(cols):
        if idx < len(frames):
            axes[r,c].imshow(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB))
            axes[r,c].axis('off')
        else:
            axes[r,c].axis('off')
        idx += 1

plt.tight_layout()
plt.savefig(save_img, dpi=200, bbox_inches='tight')
print("✅ 已保存论文同款轨迹图：", save_img)