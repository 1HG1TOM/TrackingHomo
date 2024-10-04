# conda active opencv_env->python /Users/1hg1tom/Desktop/動画試行錯誤/トラッキングとホモグラフィ/track_fencers.py
#test
import cv2
import numpy as np
import csv
import os

# 保存先のフォルダを指定
output_folder = "/Users/1hg1tom/Desktop/all/研究/fencing_csv/2024年全日本選手権個人戦女子エペ/岸本_小佐井_64"  # 適宜変更
output_file = os.path.join(output_folder, "kishimoto_kosai_2.csv")

# フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# トラッキング用変数
tracked_players = {"subject": None, "opponent": None}
selected_points = []
homography_points = []
pause_tracking = False  # スペースキーでトラッキングを一時停止するためのフラグ

# マウスクリックイベントのコールバック関数
def select_points(event, x, y, flags, param):
    global selected_points, homography_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # 2つの人物の選択が終わっていない場合
        if len(selected_points) < 2:
            selected_points.append((x, y))
            print(f"選手が選択されました: {x}, {y}")
        # 人物選択後、ホモグラフィ変換用の12点を選択
        elif len(homography_points) < 12:
            homography_points.append((x, y))
            print(f"ホモグラフィ変換用の点が選択されました: {x}, {y}")

# 動画の読み込み
cap = cv2.VideoCapture("/Users/1hg1tom/Desktop/all/研究/fencing_movie/2024年全日本選手権個人戦女子エペ/岸本_小佐井_2.mov")

# 最初のフレームを取得
ret, first_frame = cap.read()
if not ret:
    print("動画を読み込めませんでした")
    exit()

# CSVファイルを開く
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Transformed_Subject_X", "Transformed_Subject_Y", 
                     "Transformed_Opponent_X", "Transformed_Opponent_Y"])

    # 最初のフレームを表示して選手とホモグラフィ変換用の点を指定
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", select_points)

    while True:
        display_frame = first_frame.copy()

        # 赤い点で人物を表示
        for point in selected_points:
            cv2.circle(display_frame, point, 5, (0, 0, 255), -1)

        # 緑の点でホモグラフィ変換用の点を表示
        for point in homography_points:
            cv2.circle(display_frame, point, 5, (0, 255, 0), -1)

        cv2.imshow("Select Points", display_frame)

        # 2人の選手と12点のホモグラフィ変換用の点が選択されたら終了
        if len(selected_points) == 2 and len(homography_points) == 12:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # qキーで終了
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.destroyWindow("Select Points")

    # ホモグラフィ変換用の12点（実空間の対応点）
    destination_points = np.array([
        [-100, 100], [-100, -100], [100, -100], [100, 100],    # 1つ目のグリッド
        [-250, 100], [-250, -100], [250, -100], [250, 100],    # 2つ目のグリッド
        [-350, 100], [-350, -100], [350, -100], [350, 100]     # 3つ目のグリッド
    ], dtype=np.float32)

    # ホモグラフィ変換を実行
    if len(homography_points) == 12:
        src_points = np.array(homography_points, dtype=np.float32)
        homography_matrix, _ = cv2.findHomography(src_points, destination_points)

        print("ホモグラフィ変換行列:")
        print(homography_matrix)

        # トラッキングの準備
        tracker_subject = cv2.TrackerCSRT_create()
        tracker_opponent = cv2.TrackerCSRT_create()

        # 「取材対象」と「対戦相手」のトラッキング範囲を指定
        bbox_subject = (selected_points[0][0], selected_points[0][1], 50, 100)  # 50x100の範囲で仮設定
        bbox_opponent = (selected_points[1][0], selected_points[1][1], 50, 100)

        # トラッカーの初期化
        tracker_subject.init(first_frame, bbox_subject)
        tracker_opponent.init(first_frame, bbox_opponent)

        frame_count = 0

        # 動画を再生しながらホモグラフィ変換された座標を追跡
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # トラッキングの一時停止処理
            if pause_tracking:
                cv2.putText(frame, "Tracking paused. Press space to resume or click to reselect.", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Tracking", frame)

                key = cv2.waitKey(0)  # スペースキーで一時停止
                if key == ord(' '):  # スペースキーで再開
                    pause_tracking = False
                    selected_points = []
                    cv2.namedWindow("Select Points")
                    cv2.setMouseCallback("Select Points", select_points)

                    # 再度選手だけを選択
                    while True:
                        display_frame = frame.copy()

                        for point in selected_points:
                            cv2.circle(display_frame, point, 5, (0, 0, 255), -1)

                        for point in homography_points:  # ホモグラフィ点は固定
                            cv2.circle(display_frame, point, 5, (0, 255, 0), -1)

                        cv2.imshow("Select Points", display_frame)

                        if len(selected_points) == 2:
                            break

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            exit()

                    cv2.destroyWindow("Select Points")

                    # 新しい選手をトラッキング
                    tracker_subject = cv2.TrackerCSRT_create()
                    tracker_opponent = cv2.TrackerCSRT_create()
                    bbox_subject = (selected_points[0][0], selected_points[0][1], 50, 100)
                    bbox_opponent = (selected_points[1][0], selected_points[1][1], 50, 100)
                    tracker_subject.init(frame, bbox_subject)
                    tracker_opponent.init(frame, bbox_opponent)
                continue

            # トラッキング
            success_subject, bbox_subject = tracker_subject.update(frame)
            success_opponent, bbox_opponent = tracker_opponent.update(frame)

            if success_subject and success_opponent:
                # 両足の中間の座標（バウンディングボックスの底辺中央）
                subject_point = np.array([[bbox_subject[0] + bbox_subject[2] / 2, bbox_subject[1] + bbox_subject[3]]], dtype=np.float32)
                opponent_point = np.array([[bbox_opponent[0] + bbox_opponent[2] / 2, bbox_opponent[1] + bbox_opponent[3]]], dtype=np.float32)

                # 座標をホモグラフィ変換
                transformed_subject = cv2.perspectiveTransform(np.array([subject_point]), homography_matrix)
                transformed_opponent = cv2.perspectiveTransform(np.array([opponent_point]), homography_matrix)

                # 座標をCSVに書き出す
                writer.writerow([frame_count, transformed_subject[0][0][0], transformed_subject[0][0][1],
                                 transformed_opponent[0][0][0], transformed_opponent[0][0][1]])

                # トラッキング結果を描画
                p1_subject = (int(bbox_subject[0]), int(bbox_subject[1]))
                p2_subject = (int(bbox_subject[0] + bbox_subject[2]), int(bbox_subject[1] + bbox_subject[3]))
                cv2.rectangle(frame, p1_subject, p2_subject, (255, 0, 0), 2, 1)  # 青で「取材対象」を描画

                p1_opponent = (int(bbox_opponent[0]), int(bbox_opponent[1]))
                p2_opponent = (int(bbox_opponent[0] + bbox_opponent[2]), int(bbox_opponent[1] + bbox_opponent[3]))
                cv2.rectangle(frame, p1_opponent, p2_opponent, (0, 255, 0), 2, 1)  # 緑で「対戦相手」を描画

            # フレームを表示
            cv2.imshow("Tracking", frame)
            frame_count += 1

            # キー入力の処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # qキーで終了
                break
            elif key == ord(' '):  # スペースキーで一時停止
                pause_tracking = True

cap.release()
cv2.destroyAllWindows()