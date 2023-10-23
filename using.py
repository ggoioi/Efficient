import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from common import get_autoencoder, get_pdn_small

seed = 1234
out_channels = 384

def predict(image, teacher, student, autoencoder):
    teacher_output = teacher(image)
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined

# 모델 및 기타 설정
model_size = 'small'
model = torch.load('./output/small_v2/trainings/mvtec_ad/teacher_final.pth')
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 불러오기
# 모델 가중치 불러오기
model.to(device)
model.eval()

# Autoencoder 모델 로드

autoencoder_model= torch.load('./output/small_v2/trainings/mvtec_ad/autoencoder_final.pth')
# autoencoder_model.load_state_dict(torch.load(autoencoder_weights_path, map_location=device))
autoencoder_model.to(device)
autoencoder_model.eval()

# 이미지 파일이 있는 디렉토리 경로 설정
image_dir = 'Data2/test/good'  # 예측할 이미지 파일들이 있는 디렉토리 경로

# 결과를 저장할 디렉토리 경로 설정
output_dir = 'predicted_images'  # 결과 이미지를 저장할 디렉토리 경로
os.makedirs(output_dir, exist_ok=True)

# 디렉토리 안의 모든 이미지 파일에 대한 예측 수행
for image_filename in os.listdir(image_dir):
    if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
        image_path = os.path.join(image_dir, image_filename)

        # 이미지 불러오기 및 전처리
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_image = transform(image).unsqueeze(0).to(device)

        # 이미지 예측 (Autoencoder 사용)
        with torch.no_grad():
            map_combined = predict(
                image=input_image, teacher=model, student=model, autoencoder=autoencoder_model)

        # 예측 결과를 이미지로 저장
        output_image = map_combined[0, 0].cpu().numpy()
        output_filename = os.path.splitext(image_filename)[0] + '_prediction.png'
        output_path = os.path.join(output_dir, output_filename)
        Image.fromarray((output_image * 255).astype(np.uint8)).save(output_path)

print("예측이 완료되었습니다. 결과 이미지는", output_dir, "디렉토리에 저장되었습니다.")
