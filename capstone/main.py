import os
import cv2
import scipy.io
from PIL import Image

# 경로 설정
video_dirs = [
    r'C:\Users\User\Desktop\capstone\finalProject\datasets\VIS_Onboard\Videos',
    r'C:\Users\User\Desktop\capstone\finalProject\datasets\VIS_Onshore\Videos'
]
mat_dirs = [
    r'C:\Users\User\Desktop\capstone\finalProject\datasets\VIS_Onboard\ObjectGT',
    r'C:\Users\User\Desktop\capstone\finalProject\datasets\VIS_Onshore\ObjectGT'
]
output_base_dir = r'C:\Users\User\Desktop\capstone\finalProject\dataset_preparation'
output_frame_dir = os.path.join(output_base_dir, 'frames')
output_label_dir = os.path.join(output_base_dir, 'yolo_labels')

# 클래스 이름을 ID로 매핑하는 사전
class_name_to_id = {
    'Boat': 0,
    'Vessel/ship': 1,
    'Ferry': 2,
    'Kayak': 3,
    'Buoy': 4,
    'Sail boat': 5,
    'Other': 6
}

# 디렉토리 생성 함수
def create_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# 디렉토리 생성
create_dirs([output_base_dir, output_frame_dir, output_label_dir])

# 프레임 추출 및 주석 변환 함수
def extract_frames_and_convert_annotations(video_path, mat_path, output_frame_dir, output_label_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_capture = cv2.VideoCapture(video_path)

    # Load .mat file
    mat = scipy.io.loadmat(mat_path)
    struct = mat['structXML']
    bbs = struct['BB'][0]  # 'BB' 키를 사용하여 바운딩 박스 데이터를 가져옴
    object_types = struct['ObjectType'][0]  # 'ObjectType' 키를 사용하여 객체 타입 데이터를 가져옴

    base_name_parts = os.path.basename(mat_path).split('_')
    video_id = base_name_parts[1]

    # 프레임 타입을 추출합니다. (예: 'VIS_Haze', 'VIS_OB', 'VIS')
    if "Haze" in base_name_parts:
        frame_type = "VIS_Haze"
    elif "OB" in base_name_parts:
        frame_type = "VIS_OB"
    else:
        frame_type = "VIS"

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_file = os.path.join(output_frame_dir, f'{video_name}_frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_file, frame)

        label_file_name = f'{video_name}_frame_{frame_count:04d}.txt'
        label_file_path = os.path.join(output_label_dir, label_file_name)

        img = Image.open(frame_file)
        img_width, img_height = img.size

        with open(label_file_path, 'w') as f:
            if frame_count < len(bbs):
                frame_bb = bbs[frame_count]
                frame_object_type = object_types[frame_count]
                for bb, obj_type in zip(frame_bb, frame_object_type):
                    if len(obj_type) == 0:
                        print(f"Warning: Object type is empty for frame {frame_file}")
                        continue  # obj_type이 빈 배열인 경우 건너뜁니다.

                    class_name = obj_type[0]  # 클래스 이름 추출
                    class_name_str = str(class_name).strip("[]'")  # numpy 배열을 문자열로 변환 및 불필요한 문자 제거
                    if class_name_str in class_name_to_id:
                        class_id = class_name_to_id[class_name_str]  # 클래스 ID로 변환

                        x_center = (bb[0] + bb[2] / 2) / img_width
                        y_center = (bb[1] + bb[3] / 2) / img_height
                        width = bb[2] / img_width
                        height = bb[3] / img_height

                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    else:
                        print(f"Warning: Unknown class name '{class_name_str}' found in file {label_file_name}")

        frame_count += 1

    video_capture.release()

# 모든 .avi 파일에 대해 프레임 추출 및 주석 변환 수행
for video_dir, mat_dir in zip(video_dirs, mat_dirs):
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.avi'):
            video_path = os.path.join(video_dir, video_file)
            mat_file_name = video_file.replace('.avi', '_ObjectGT.mat')
            mat_path = os.path.join(mat_dir, mat_file_name)

            if not os.path.exists(mat_path):
                print(f"Warning: Corresponding .mat file for {video_path} not found.")
                continue

            extract_frames_and_convert_annotations(video_path, mat_path, output_frame_dir, output_label_dir)
