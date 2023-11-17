import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.quantization import QConfig
from torch.quantization.observer import MovingAverageMinMaxObserver, MinMaxObserver

# 필요한 경우 자신의 custom class 또는 함수를 임포트하세요.
# from needmodule import ImageFolderWithPath

# 데이터셋 경로와 저장 경로 설정
data_path = './Data_quanti'
save_path = './model_save'

# 모델과 이미지 크기 설정
image_size = 256
batch_size = 32  # 혹은 데이터셋에 맞는 적절한 배치 사이즈를 설정하세요.

# 이미지 전처리
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 로딩
test_set = datasets.ImageFolder(data_path, transform=default_transform)  # ImageFolderWithPath를 ImageFolder로 대체
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 모델 로딩
path = 'output/small_v1_10_25/trainings/mvtec_ad'
auto = torch.load(path + '/autoencoder_final.pth')
stu = torch.load(path + '/student_final.pth')
tea = torch.load(path + '/teacher_final.pth')

stu.to('cpu')
stu.eval()

# 모듈 융합 (fuse modules)
fusion_layers = [
    ['0', '1'],  # Conv2d + ReLU
    ['3', '4'],  # Conv2d + ReLU
    ['6', '7']   # 필요하다면 더 추가하세요.
]
for layers_to_fuse in fusion_layers:
    torch.quantization.fuse_modules(stu, layers_to_fuse, inplace=True)

# 양자화 구성 설정
my_qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
)

# 양자화 준비
stu.qconfig = my_qconfig
torch.quantization.prepare(stu, inplace=True)

# 모델 교정 - 대표 데이터를 사용하여 양자화 통계를 수집
for images, _ in test_loader:
    stu(images)

# 양자화 적용
torch.quantization.convert(stu, inplace=True)

# 모델 저장
torch.save(stu, os.path.join(save_path, 'student_quantized.pth'))
