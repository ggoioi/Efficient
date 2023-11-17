from torchvision.datasets import ImageFolder
import torch
import numpy as np
import time
import os
import tifffile
import cv2
class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path


def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    out_channels = 384
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

def visualize_one_sample(test_data, orig_height, orig_width, auto, stu, tea):
    with torch.no_grad():
        tea_mean = torch.tensor([[[[-0.2424]], [[0.0180]], [[-1.2244]], [[0.4333]], [[2.2549]], [[0.3087]], [[-0.7746]],
                                  [[2.4213]], [[0.4394]], [[0.8094]], [[-0.0130]], [[-0.2341]], [[1.1500]], [[1.3726]],
                                  [[0.5126]], [[-0.3067]], [[0.4348]], [[-0.3848]], [[-0.2135]], [[-1.0498]],
                                  [[-0.1907]], [[0.1004]], [[0.7071]], [[-0.6661]], [[0.4800]], [[0.4084]], [[-0.3454]],
                                  [[0.6030]], [[0.8339]], [[0.5146]], [[-0.7983]], [[-0.3483]], [[-0.0547]],
                                  [[-0.2370]], [[0.5813]], [[1.0180]], [[0.5915]], [[3.6247]], [[0.1870]], [[1.2598]],
                                  [[1.1357]], [[0.6711]], [[-1.1979]], [[-0.0224]], [[-2.4059]], [[-0.5170]],
                                  [[1.3364]], [[1.2592]], [[0.2997]], [[-0.2073]], [[-0.5946]], [[-1.3365]],
                                  [[-0.3937]], [[0.4321]], [[0.0828]], [[1.1451]], [[0.9940]], [[0.1768]], [[-0.8476]],
                                  [[0.2583]], [[-0.5574]], [[0.8534]], [[0.2820]], [[0.6477]], [[2.2245]], [[0.2918]],
                                  [[-0.7268]], [[0.6952]], [[0.8345]], [[-0.9721]], [[-0.7587]], [[-0.3005]],
                                  [[-0.0792]], [[2.3134]], [[-0.6635]], [[0.6427]], [[-0.5191]], [[-0.3675]],
                                  [[-0.4296]], [[-0.8419]], [[0.1169]], [[0.6322]], [[0.1008]], [[0.1002]], [[-0.4098]],
                                  [[0.1977]], [[-0.3924]], [[1.7113]], [[-0.8234]], [[1.3658]], [[0.4534]], [[-0.2050]],
                                  [[0.1855]], [[1.0878]], [[0.3548]], [[0.2249]], [[0.4145]], [[-0.8145]], [[-0.6464]],
                                  [[1.5516]], [[0.5539]], [[0.2049]], [[1.6282]], [[0.0600]], [[1.0955]], [[0.3521]],
                                  [[1.0512]], [[1.8840]], [[-0.1337]], [[-0.7040]], [[-0.6787]], [[0.0536]], [[0.6886]],
                                  [[0.1456]], [[0.4264]], [[0.1253]], [[0.5590]], [[-0.0407]], [[0.4941]], [[-0.3543]],
                                  [[-0.3783]], [[-1.2664]], [[-0.4316]], [[0.8372]], [[0.3482]], [[-0.1294]],
                                  [[0.4693]], [[0.1138]], [[0.7817]], [[0.7110]], [[0.4369]], [[0.1400]], [[0.3048]],
                                  [[1.2740]], [[-0.7819]], [[-0.4600]], [[-0.0387]], [[0.7414]], [[0.4185]],
                                  [[-0.1535]], [[3.0061]], [[0.1776]], [[-0.1079]], [[-0.3652]], [[2.5614]], [[1.6988]],
                                  [[1.6698]], [[-0.2106]], [[0.6597]], [[-0.3211]], [[2.2354]], [[1.8114]], [[0.9581]],
                                  [[0.1121]], [[0.7168]], [[1.4358]], [[1.3550]], [[0.9403]], [[0.8464]], [[-0.4723]],
                                  [[-0.4700]], [[0.4968]], [[0.6237]], [[1.4785]], [[0.0238]], [[-1.1924]], [[1.2337]],
                                  [[-0.2794]], [[-1.3830]], [[0.6770]], [[-0.7837]], [[-0.7739]], [[-0.8285]],
                                  [[-0.1270]], [[-0.1179]], [[1.1292]], [[1.1814]], [[0.3699]], [[2.5630]], [[3.3961]],
                                  [[-0.2012]], [[-0.2394]], [[0.4039]], [[0.1910]], [[-0.2213]], [[0.8900]], [[0.6101]],
                                  [[0.4319]], [[-0.8877]], [[-0.4223]], [[-0.6326]], [[0.1896]], [[0.0245]],
                                  [[-0.4403]], [[0.2686]], [[-0.0794]], [[0.9182]], [[0.2393]], [[0.5819]], [[-0.0086]],
                                  [[-0.0746]], [[0.0122]], [[2.3759]], [[-0.1232]], [[-0.1304]], [[-0.6694]],
                                  [[-0.3951]], [[0.1675]], [[0.2354]], [[0.6525]], [[0.2233]], [[-0.0319]], [[-0.1766]],
                                  [[0.1747]], [[0.5969]], [[1.1399]], [[-0.4665]], [[0.6711]], [[0.1004]], [[-0.5095]],
                                  [[0.0870]], [[0.5745]], [[0.5645]], [[0.2465]], [[0.1279]], [[0.2638]], [[0.4162]],
                                  [[0.4640]], [[0.3424]], [[-0.3763]], [[-0.0137]], [[0.2540]], [[0.6100]], [[0.6211]],
                                  [[0.1490]], [[-0.2034]], [[-0.5995]], [[0.2587]], [[1.1936]], [[0.8617]], [[-0.0699]],
                                  [[0.2738]], [[0.5776]], [[0.6630]], [[-0.0469]], [[0.2873]], [[0.0979]], [[0.4375]],
                                  [[0.2606]], [[0.1021]], [[0.1250]], [[-0.0700]], [[0.7698]], [[0.4721]], [[0.0626]],
                                  [[0.2847]], [[-0.2781]], [[0.0649]], [[-0.2791]], [[-0.2571]], [[0.2001]], [[0.1559]],
                                  [[0.2499]], [[0.0503]], [[-0.2458]], [[0.3467]], [[0.2351]], [[-0.2591]], [[0.0374]],
                                  [[0.3533]], [[0.7907]], [[-0.6353]], [[-0.1320]], [[1.6722]], [[0.6488]], [[0.0162]],
                                  [[0.0039]], [[0.3699]], [[0.3246]], [[0.2794]], [[0.3052]], [[-0.0513]], [[-0.3094]],
                                  [[-0.7135]], [[0.0858]], [[0.6111]], [[-0.0268]], [[-0.0424]], [[-0.1843]],
                                  [[-0.1536]], [[0.2587]], [[-0.6671]], [[0.1095]], [[-0.0782]], [[0.3441]],
                                  [[-0.1841]], [[0.5977]], [[0.0381]], [[0.4847]], [[-0.0143]], [[-0.3484]], [[0.1144]],
                                  [[-0.3306]], [[-0.6221]], [[-0.0487]], [[-0.7605]], [[-0.1255]], [[0.1018]],
                                  [[0.1617]], [[0.1804]], [[0.3279]], [[0.7773]], [[0.0157]], [[-0.1985]], [[0.1171]],
                                  [[0.6595]], [[0.4053]], [[0.1192]], [[-0.3045]], [[-0.0262]], [[0.8687]], [[-0.1381]],
                                  [[0.7852]], [[-0.3042]], [[1.3972]], [[0.2602]], [[0.2826]], [[-0.3253]], [[-0.2092]],
                                  [[0.0457]], [[-0.4530]], [[1.4129]], [[0.8688]], [[-0.0178]], [[0.1801]], [[-0.2290]],
                                  [[0.8903]], [[0.6979]], [[0.0858]], [[-1.6217]], [[-0.6389]], [[0.0740]], [[0.0388]],
                                  [[0.0357]], [[0.4174]], [[0.2463]], [[-0.1179]], [[0.3246]], [[0.4380]], [[-0.3920]],
                                  [[0.2633]], [[0.3018]], [[-0.1090]], [[-0.1113]], [[1.1089]], [[0.0816]], [[1.1597]],
                                  [[0.3939]], [[0.0697]], [[0.4118]], [[1.2493]], [[0.5115]], [[0.0897]], [[0.2284]],
                                  [[-0.7352]], [[0.3413]], [[-0.8868]], [[0.2866]], [[0.2025]], [[-0.2863]], [[0.6786]],
                                  [[0.2960]], [[0.1393]], [[0.7294]], [[-1.3546]], [[0.0352]], [[0.0748]], [[0.6259]],
                                  [[-0.1634]], [[0.0832]], [[0.4180]], [[0.0434]], [[0.0078]], [[2.1498]]]])
        tea_std = torch.tensor([[[[0.7210]], [[0.5510]], [[0.1756]], [[0.3689]], [[1.1653]], [[0.5342]], [[0.5155]],
                                 [[1.1279]], [[0.7137]], [[0.8472]], [[0.5637]], [[0.6594]], [[0.8288]], [[0.5970]],
                                 [[0.8553]], [[0.6885]], [[0.4438]], [[0.5344]], [[0.7042]], [[0.7407]], [[0.5766]],
                                 [[0.6027]], [[0.2672]], [[0.1472]], [[0.9607]], [[0.5485]], [[0.7639]], [[0.4091]],
                                 [[0.4971]], [[0.8395]], [[0.5056]], [[0.5421]], [[0.3973]], [[0.6759]], [[0.4809]],
                                 [[0.4321]], [[0.8861]], [[0.6481]], [[0.5048]], [[0.4008]], [[0.7506]], [[0.6084]],
                                 [[0.3158]], [[0.3349]], [[0.6043]], [[0.5047]], [[0.7788]], [[0.8131]], [[0.3778]],
                                 [[0.6161]], [[0.6500]], [[0.3115]], [[0.9674]], [[0.2604]], [[0.5646]], [[0.8395]],
                                 [[0.3224]], [[0.5152]], [[0.6625]], [[1.4017]], [[0.6083]], [[0.6749]], [[0.4757]],
                                 [[0.4315]], [[0.7209]], [[0.5809]], [[0.7694]], [[0.8016]], [[0.5392]], [[0.2144]],
                                 [[0.2943]], [[0.4629]], [[0.8375]], [[0.9581]], [[0.3819]], [[0.3109]], [[0.4916]],
                                 [[0.3079]], [[0.6088]], [[0.4744]], [[0.4179]], [[0.8474]], [[0.8471]], [[0.2543]],
                                 [[0.7540]], [[0.5572]], [[0.2409]], [[1.3656]], [[0.7816]], [[0.6941]], [[0.4841]],
                                 [[0.5072]], [[0.5098]], [[0.3263]], [[0.7505]], [[0.3600]], [[0.4255]], [[0.3874]],
                                 [[0.3643]], [[0.8366]], [[1.1025]], [[0.6757]], [[0.3546]], [[0.6256]], [[0.4599]],
                                 [[0.5319]], [[0.4146]], [[0.5138]], [[0.3113]], [[0.6139]], [[0.8033]], [[0.9233]],
                                 [[0.9090]], [[0.5072]], [[0.3017]], [[0.7769]], [[0.7690]], [[0.5336]], [[0.4226]],
                                 [[0.2112]], [[0.6279]], [[0.4222]], [[0.5391]], [[0.3494]], [[0.8159]], [[0.3488]],
                                 [[0.7291]], [[1.1794]], [[0.5642]], [[0.9828]], [[0.8908]], [[0.6626]], [[0.8610]],
                                 [[0.7487]], [[0.3485]], [[0.3363]], [[0.3605]], [[0.2994]], [[1.0799]], [[0.7403]],
                                 [[0.9149]], [[0.6123]], [[0.5648]], [[0.7379]], [[0.6049]], [[0.7474]], [[0.5518]],
                                 [[0.5179]], [[0.6257]], [[0.4046]], [[0.7643]], [[0.8585]], [[0.8220]], [[0.7664]],
                                 [[0.5025]], [[1.0998]], [[0.7341]], [[1.1151]], [[1.0138]], [[0.3045]], [[0.8004]],
                                 [[1.2534]], [[0.6919]], [[0.9202]], [[0.6395]], [[0.1228]], [[0.8022]], [[0.2877]],
                                 [[0.4879]], [[0.9513]], [[0.2743]], [[0.2507]], [[0.3188]], [[0.4822]], [[0.4292]],
                                 [[0.9042]], [[1.1545]], [[0.4849]], [[1.0645]], [[1.1706]], [[0.3080]], [[0.3965]],
                                 [[0.7950]], [[0.4070]], [[0.8170]], [[0.5476]], [[1.3167]], [[0.3492]], [[0.2034]],
                                 [[0.3934]], [[0.4979]], [[0.6763]], [[0.2277]], [[0.1950]], [[0.2000]], [[0.2353]],
                                 [[0.4220]], [[0.1850]], [[0.3050]], [[0.2267]], [[0.1698]], [[0.1867]], [[0.3933]],
                                 [[0.1538]], [[0.2297]], [[0.1867]], [[0.2854]], [[0.1907]], [[0.2618]], [[0.2243]],
                                 [[0.2880]], [[0.3052]], [[0.2343]], [[0.3003]], [[0.2646]], [[0.2228]], [[0.3912]],
                                 [[0.2337]], [[0.2273]], [[0.2721]], [[0.2492]], [[0.3921]], [[0.3012]], [[0.1781]],
                                 [[0.1880]], [[0.2290]], [[0.2713]], [[0.2218]], [[0.2250]], [[0.1876]], [[0.2058]],
                                 [[0.2008]], [[0.3595]], [[0.1802]], [[0.2227]], [[0.1679]], [[0.4108]], [[0.2981]],
                                 [[0.2658]], [[0.2694]], [[0.2684]], [[0.3033]], [[0.3195]], [[0.2191]], [[0.2375]],
                                 [[0.2065]], [[0.4021]], [[0.3655]], [[0.1965]], [[0.1980]], [[0.4287]], [[0.2575]],
                                 [[0.3284]], [[0.2775]], [[0.1743]], [[0.1959]], [[0.2101]], [[0.2388]], [[0.2233]],
                                 [[0.1743]], [[0.3751]], [[0.3008]], [[0.4390]], [[0.3692]], [[0.2242]], [[0.2374]],
                                 [[0.2338]], [[0.3188]], [[0.2684]], [[0.1843]], [[0.3204]], [[0.3001]], [[0.2305]],
                                 [[0.3170]], [[0.1957]], [[0.1844]], [[0.2399]], [[0.2165]], [[0.2171]], [[0.2479]],
                                 [[0.2398]], [[0.1985]], [[0.2347]], [[0.1968]], [[0.2566]], [[0.2874]], [[0.1870]],
                                 [[0.2915]], [[0.2888]], [[0.2765]], [[0.2224]], [[0.1611]], [[0.2563]], [[0.1869]],
                                 [[0.2116]], [[0.2220]], [[0.2086]], [[0.1510]], [[0.2312]], [[0.2360]], [[0.1982]],
                                 [[0.2747]], [[0.2078]], [[0.2211]], [[0.3243]], [[0.3483]], [[0.2272]], [[0.1841]],
                                 [[0.1953]], [[0.2353]], [[0.2449]], [[0.2781]], [[0.2308]], [[0.1614]], [[0.2176]],
                                 [[0.2263]], [[0.2109]], [[0.2336]], [[0.2006]], [[0.1877]], [[0.4143]], [[0.2132]],
                                 [[0.2995]], [[0.2466]], [[0.2636]], [[0.2122]], [[0.2315]], [[0.1929]], [[0.2226]],
                                 [[0.3948]], [[0.1915]], [[0.2638]], [[0.1955]], [[0.2188]], [[0.2426]], [[0.2557]],
                                 [[0.2117]], [[0.2976]], [[0.2352]], [[0.3359]], [[0.2575]], [[0.2648]], [[0.2415]],
                                 [[0.2259]], [[0.2207]], [[0.2703]], [[0.1552]], [[0.2362]], [[0.2190]], [[0.1421]],
                                 [[0.2817]], [[0.2124]], [[0.2000]], [[0.3805]], [[0.3599]], [[0.1654]], [[0.3081]],
                                 [[0.2008]], [[0.1478]], [[0.2974]], [[0.1801]], [[0.3408]], [[0.1623]], [[0.2289]],
                                 [[0.3081]], [[0.1890]], [[0.3107]], [[0.3488]], [[0.2816]], [[0.2739]], [[0.1929]],
                                 [[0.2158]], [[0.2668]], [[0.2065]], [[0.4749]], [[0.2022]], [[0.2767]], [[0.1630]],
                                 [[0.2128]], [[0.4182]], [[0.3068]], [[0.2370]], [[0.3492]], [[0.5978]]]])
        start_time = time.time()
        auto.eval()
        stu.eval()
        tea.eval()
        map_combined, map_st, map_ae = predict(test_data, teacher = tea, student = stu,
                                               autoencoder=auto, teacher_mean=tea_mean, teacher_std=tea_std)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()
        y_score = np.max(map_combined)
        if y_score <= 0.09:
            y_class = '정상'
        else:
            y_class = '불량'
        end_time = time.time()  # 프레임 처리가 끝난 후의 시간 기록
        fps = end_time - start_time  # FPS 계산
    return y_class, fps, map_combined


def anomaly(original_image, map_combined, path, image_name, y_class, threshold=0.5, color=(255, 0, 0)):
    defect_class = 'class'
    test_output_dir = os.path.join(path, 'test')
    # 폴더 구조 생성
    if not os.path.exists(os.path.join(test_output_dir, defect_class)):
        os.makedirs(os.path.join(test_output_dir, defect_class))
    normalized_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    # [0, 255] 범위로 스케일링 후 numpy 변환
    original_image_np = (normalized_image.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    original_image_np = np.ascontiguousarray(original_image_np)
    # Resize the anomaly map to the original image's shape.
    anomaly_map_resized = cv2.resize(map_combined, (original_image_np.shape[1], original_image_np.shape[0]))

    # 이미지를 0-255 범위로 normalize
    anomaly_map_normalized = cv2.normalize(anomaly_map_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 이미지 이진화 (binary thresholding)
    ret, thresh = cv2.threshold(anomaly_map_normalized, 127, 255, cv2.THRESH_BINARY)

    # 외곽선 찾기
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 원본 이미지에 외곽선 그리기
    highlighted_image = cv2.drawContours(original_image_np, contours, -1, color, 2)

    # 이미지 저장
    img_nm = os.path.splitext(image_name)[0]
    img_nm = img_nm + y_class + '_anomaly'
    file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
    tifffile.imwrite(file, highlighted_image)