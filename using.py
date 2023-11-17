'''
Conv2d와 ReLU를 합쳐서 양자화를 진행하는....->
두개를 합쳐서 진행할려고 하니 입력사이즈가 커널 사이즈랑 달라서 계속 안된다고 함
방법을 아예 새로 고안 해야할듯 -->
일단 autoencoder를 양자화 하기에 필요한 데이터 ->캘리브레이션을 위한 데이터를 찾아야함
대표적인 정상 데이터를 찾아서 양자화에 활용
파라미터를 찾아야하기 때문에
'''

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 모델과 데이터셋 경로 설정
model_path = 'output/small_v1_10_25/trainings/mvtec_ad'
dataset_path = 'Data_quanti'

# GPU 사용 가능 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
autoencoder = torch.load(model_path+'/autoencoder_final.pth', map_location=device)

# 양자화를 위한 설정
autoencoder.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 모델을 양자화 준비 상태로 변환
autoencoder_prepared = torch.quantization.prepare(autoencoder, inplace=False)

# 데이터셋 로드를 위한 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize는 데이터셋에 따라 평균과 표준편차 값을 조정해야 할 수 있습니다.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 양자화 캘리브레이션을 위한 데이터셋과 데이터 로더 준비
calibration_dataset = ImageFolder(root=dataset_path, transform=transform)
calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=True)

# 캘리브레이션 수행
autoencoder_prepared.eval()
with torch.no_grad():
    for data, _ in calibration_loader:
        autoencoder_prepared(data.to(device))

# 모델을 최종적으로 양자화하여 int8 버전으로 변환
autoencoder_int8 = torch.quantization.convert(autoencoder_prepared, inplace=False)

# 양자화된 모델 저장
torch.save(autoencoder_int8, model_path+'/autoencoder_int8.pth')
print(autoencoder_int8)