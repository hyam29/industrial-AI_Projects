import yaml
from ultralytics import YOLO

def main():
    # Load the hyperparameters from the YAML file
    with open("hyp.yaml", "r") as file:
        hyp_params = yaml.safe_load(file)  # YAML 파일을 읽어와 딕셔너리로 변환

    # Load the model
    model = YOLO("yolov8n.pt")  # 사전 훈련된 모델 8n 불러오기

    # Train the model with hyperparameters
    results = model.train(
        data="data.yaml",  # 데이터 설정 파일
        epochs=15,  # 훈련 에폭 수
        imgsz=640,         # 이미지 크기
        **hyp_params,      # 딕셔너리의 하이퍼파라미터를 풀어서 전달
        device=0           # 사용할 장치 지정 (예: 첫 번째 GPU)
    )

if __name__ == '__main__':
    main()
