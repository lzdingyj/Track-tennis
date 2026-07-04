# import os
# import parse
# import shutil
#
# from dataset import data_dir
# from utils.general import list_dirs, generate_data_frames, get_num_frames, get_match_median
# from utils.visualize import plot_median_files
#
#
# # Replace csv to corrected csv in test set
# if os.path.exists('corrected_test_label'):
#     match_dirs = list_dirs(os.path.join(data_dir, 'test'))
#     match_dirs = sorted(match_dirs, key=lambda s: int(s.split('match')[-1]))
#     for match_dir in match_dirs:
#         file_format_str = os.path.join('{}', 'test', '{}')
#         _, match_dir = parse.parse(file_format_str, match_dir)
#         if not os.path.exists(os.path.join(data_dir, 'test', match_dir, 'corrected_csv')):
#             shutil.copytree(os.path.join('corrected_test_label', match_dir, 'corrected_csv'),
#                             os.path.join(data_dir, 'test', match_dir, 'corrected_csv'))
#             shutil.copy(os.path.join('corrected_test_label', 'drop_frame.json'),
#                         os.path.join(data_dir, 'drop_frame.json'))
#
# # Generate frames from videos
# for split in ['train', 'test']:
#     split_frame_count = 0
#     match_dirs = list_dirs(os.path.join(data_dir, split))
#     for match_dir in match_dirs:
#         match_frame_count = 0
#         file_format_str = os.path.join('{}', 'match{}')
#         _, match_id = parse.parse(file_format_str, match_dir)
#         video_files = list_dirs(os.path.join(match_dir, 'video'))
#         for video_file in video_files:
#             generate_data_frames(video_file)
#             file_format_str = os.path.join('{}', 'video', '{}.mp4')
#             _, video_name = parse.parse(file_format_str, video_file)
#             rally_dir = os.path.join(match_dir, 'frame', video_name)
#             video_frame_count = get_num_frames(rally_dir)
#             print(f'[{split} / match{match_id} / {video_name}]\tvideo frames: {video_frame_count}')
#             match_frame_count += video_frame_count
#         get_match_median(match_dir)
#         print(f'[{split} / match{match_id}]:\ttotal frames: {match_frame_count}')
#         split_frame_count += match_frame_count
#
#     print(f'[{split}]:\ttotal frames: {split_frame_count}')
#
# # Form validation set
# if not os.path.exists(os.path.join(data_dir, 'val')):
#     match_dirs = list_dirs(os.path.join(data_dir, 'train'))
#     match_dirs = sorted(match_dirs, key=lambda s: int(s.split('match')[-1]))
#     for match_dir in match_dirs:
#         # Pick last rally in each match as validation set
#         video_files = list_dirs(os.path.join(match_dir, 'video'))
#         file_format_str = os.path.join('{}', 'train', '{}', 'video','{}.mp4')
#         _, match_dir, rally_id = parse.parse(file_format_str, video_files[-1])
#         os.makedirs(os.path.join(data_dir, 'val', match_dir, 'csv'), exist_ok=True)
#         os.makedirs(os.path.join(data_dir, 'val', match_dir, 'video'), exist_ok=True)
#         shutil.move(os.path.join(data_dir, 'train', match_dir, 'csv', f'{rally_id}_ball.csv'),
#                     os.path.join(data_dir, 'val', match_dir, 'csv', f'{rally_id}_ball.csv'))
#         shutil.move(os.path.join(data_dir, 'train', match_dir, 'video', f'{rally_id}.mp4'),
#                     os.path.join(data_dir, 'val', match_dir, 'video', f'{rally_id}.mp4'))
#         shutil.move(os.path.join(data_dir, 'train', match_dir, 'frame', rally_id),
#                     os.path.join(data_dir, 'val', match_dir, 'frame', rally_id))
#         shutil.copy(os.path.join(data_dir, 'train', match_dir, 'median.npz'),
#                     os.path.join(data_dir, 'val', match_dir, 'median.npz'))
#
# # Plot median frames, save at <data_dir>/median
# plot_median_files(data_dir)
#
# print('Done.')


import os
import cv2
import shutil
import pandas as pd

# ===================== 【你的路径】 =====================
data_dir = "E:/TrackNetV2_dataset"   # 你的真实路径
TARGET_ROOT = "E:/TrackNet_dataset_frame"               # 输出目录（自动创建）
# data_dir = "./TrackNetV2_dataset"   # 你的真实路径
# TARGET_ROOT = "./TrackNet_dataset_frame"               # 输出目录（自动创建）
# ======================================================

# 自动划分 train / val / test
split_rules = {
    "train": {
        "Amateur": ["match1", "match2"],
        "Professional": ["match1", "match2", "match3", "match4", "match5", "match6", "match7"]
    },
    "val": {
        "Amateur": ["match3"],
        "Professional": ["match8", "match9"]
    },
    "test": {
        "Professional": ["match10"]
    }
}

match_idx = 1

for split, categories in split_rules.items():
    for cat, matches in categories.items():
        for match in matches:
            src_match = os.path.join(data_dir, cat, match)
            dst_match = os.path.join(TARGET_ROOT, split, f"match{match_idx}")

            os.makedirs(dst_match, exist_ok=True)
            os.makedirs(os.path.join(dst_match, "csv"), exist_ok=True)
            os.makedirs(os.path.join(dst_match, "video"), exist_ok=True)

            # 复制 csv 和 video
            if os.path.exists(os.path.join(src_match, "csv")):
                shutil.copytree(os.path.join(src_match, "csv"),
                                os.path.join(dst_match, "csv"), dirs_exist_ok=True)
            if os.path.exists(os.path.join(src_match, "video")):
                shutil.copytree(os.path.join(src_match, "video"),
                                os.path.join(dst_match, "video"), dirs_exist_ok=True)

            # 自动拆帧
            video_dir = os.path.join(dst_match, "video")
            for vid in os.listdir(video_dir):
                if vid.endswith(".mp4"):
                    vid_path = os.path.join(video_dir, vid)
                    rally = vid.replace(".mp4", "")
                    frame_dir = os.path.join(dst_match, "frame", rally)
                    os.makedirs(frame_dir, exist_ok=True)

                    cap = cv2.VideoCapture(vid_path)
                    cnt = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        cv2.imwrite(os.path.join(frame_dir, f"{cnt}.png"), frame)
                        cnt += 1
                    cap.release()

            print(f"✅ {split} -> {cat}/{match} → match{match_idx}")
            match_idx += 1

print("\n🎉 数据集全部自动生成完成！")
print("生成目录：data/train, data/val, data/test")