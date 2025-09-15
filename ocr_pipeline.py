import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import io
import torchvision.models as models
import pyclipper
from shapely.geometry import Polygon

# --- 1. 모델 아키텍처 정의 (새로운 DBNet 버전) ---
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(self.maxpool(c1))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5

class DBNetFPN(nn.Module):
    def __init__(self, in_channels, inner_channels=256):
        super().__init__()
        self.in5 = nn.Conv2d(in_channels[3], inner_channels, 1)
        self.in4 = nn.Conv2d(in_channels[2], inner_channels, 1)
        self.in3 = nn.Conv2d(in_channels[1], inner_channels, 1)
        self.in2 = nn.Conv2d(in_channels[0], inner_channels, 1)
        self.out5 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1), nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1), nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1), nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1)
    def forward(self, features):
        c2, c3, c4, c5 = features
        p5 = self.in5(c5)
        p4 = self.in4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.in3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p2 = self.in2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        o2 = self.out2(p2)
        o3 = F.interpolate(self.out3(p3), size=o2.shape[2:], mode='nearest')
        o4 = F.interpolate(self.out4(p4), size=o2.shape[2:], mode='nearest')
        o5 = F.interpolate(self.out5(p5), size=o2.shape[2:], mode='nearest')
        fuse = torch.cat((o2, o3, o4, o5), 1)
        return fuse

class DBNetHead(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 3, padding=1)
        self.conv_bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2)
        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.ConvTranspose2d(in_channels // 4, out_channels, 2, 2)
    def forward(self, x):
        x = self.relu1(self.conv_bn1(self.conv1(x)))
        x = self.relu2(self.conv_bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class DBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.fpn = DBNetFPN(in_channels=[64, 128, 256, 512])
        self.head = DBNetHead(in_channels=256)
    def forward(self, x):
        features = self.backbone(x)
        fpn_out = self.fpn(features)
        maps = self.head(fpn_out)
        return maps

class ResNetBiLSTMCTC(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=256, lstm_layers=2, dropout_p=0.5):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=False, features_only=True, out_indices=[3], in_chans=1)
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 64, 400)
            dummy_features = self.backbone(dummy_input)
            feature_dim = dummy_features[0].shape[1]
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm = nn.LSTM(feature_dim, lstm_hidden_size, num_layers=lstm_layers, bidirectional=True, batch_first=True, dropout=dropout_p if lstm_layers > 1 else 0)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)
        self.dropout_p = dropout_p
    def forward(self, x):
        features = self.backbone(x)[0]
        features = self.pool(features)
        features = features.squeeze(2).permute(0, 2, 1)
        lstm_out, _ = self.lstm(features)
        lstm_out = F.dropout(lstm_out, p=self.dropout_p, training=self.training)
        output = self.fc(lstm_out)
        output = F.log_softmax(output, dim=2)
        return output.permute(1, 0, 2)

# --- 2. OCR 파이프라인 클래스 (새로운 DBNet 호환) ---
class OCR_Pipeline:
    def __init__(self, det_weights, rec_weights, char_map_path):
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        with open(char_map_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f]
        self.char_map = ['[blank]'] + chars + ['[UNK]']
        num_classes = len(self.char_map)

        self.det_model = DBNet().to(self.device)
        det_checkpoint = torch.load(det_weights, map_location=self.device)
        state_dict = det_checkpoint['model_state_dict'] if 'model_state_dict' in det_checkpoint else det_checkpoint
        self.det_model.load_state_dict(state_dict)
        self.det_model.eval()
        print(f"탐지 모델(DBNet) 로드 완료: {det_weights}")
        
        self.rec_model = ResNetBiLSTMCTC(num_classes=num_classes).to(self.device)
        rec_checkpoint = torch.load(rec_weights, map_location=self.device)
        state_dict = rec_checkpoint['model_state_dict'] if 'model_state_dict' in rec_checkpoint else rec_checkpoint
        self.rec_model.load_state_dict(state_dict)
        self.rec_model.eval()
        print("인식 모델 로드 완료.")

        self.det_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((960, 960)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        self.rec_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    def _get_polygons_from_dbnet_map(self, prob_map, box_thresh=0.5, unsharp_ratio=1.5, min_area=10):
        binary_map = prob_map > box_thresh
        contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area: continue
            points = contour.reshape(-1, 2)
            try:
                poly = Polygon(points)
                distance = poly.area * unsharp_ratio / poly.length
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                expanded_polygons = pco.Execute(distance)
                if not expanded_polygons: continue
                expanded_poly = np.array(expanded_polygons[0])
                polygons.append(expanded_poly)
            except Exception: continue
        return polygons

    def _crop_and_warp(self, image, polygon):
        rect = cv2.minAreaRect(polygon.astype(np.float32))
        box = cv2.boxPoints(rect)
        s = box.sum(axis=1)
        ordered_box = np.zeros((4, 2), dtype="float32")
        ordered_box[0] = box[np.argmin(s)]; ordered_box[2] = box[np.argmax(s)]
        diff = np.diff(box, axis=1)
        ordered_box[1] = box[np.argmin(diff)]; ordered_box[3] = box[np.argmax(diff)]
        (tl, tr, br, bl) = ordered_box
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)); widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)); heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        if maxWidth == 0 or maxHeight == 0: return None
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered_box, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def _ctc_decode(self, preds):
        preds_indices = preds.argmax(1).cpu().numpy()
        decoded_text = []
        last_char_index = -1
        for idx in preds_indices:
            if idx != 0 and idx != last_char_index:
                if idx < len(self.char_map): decoded_text.append(self.char_map[idx])
            last_char_index = idx
        return "".join(decoded_text)
    
    def _visualize_results(self, image_cv, results):
        vis_image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(vis_image_pil)
        try: font = ImageFont.truetype("malgun.ttf", 20)
        except IOError: font = ImageFont.load_default()
        for res in results:
            box_points = res['box']
            text = res['text']
            draw.polygon([tuple(p) for p in box_points], outline=(0, 255, 0), width=3)
            text_pos = (int(box_points[0][0]), int(box_points[0][1] - 25))
            draw.text(text_pos, text, fill=(255, 0, 0), font=font)
        return cv2.cvtColor(np.array(vis_image_pil), cv2.COLOR_RGB2BGR)

    def run_from_bytes(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original_image is None: raise ValueError("Cannot decode image from bytes")
        
        h_orig, w_orig, _ = original_image.shape
        image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        det_input = self.det_transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            det_output = self.det_model(det_input)
            prob_map = torch.sigmoid(det_output[:, 0, :, :]).squeeze(0).cpu().numpy()

        polygons = self._get_polygons_from_dbnet_map(prob_map, box_thresh=0.5)
        if not polygons: return [], original_image

        h_resized, w_resized = 960, 960
        cropped_images, scaled_polygons = [], []
        for poly in polygons:
            scaled_poly = (poly.astype(np.float32) * [w_orig / w_resized, h_orig / h_resized]).astype(np.int32)
            scaled_polygons.append(scaled_poly)
            warped_img = self._crop_and_warp(original_image, scaled_poly)
            if warped_img is None: continue
            warped_pil = Image.fromarray(warped_img).convert("L").resize((400, 64))
            cropped_images.append(self.rec_transform(warped_pil))
        
        if not cropped_images: return [], original_image
        
        rec_input = torch.stack(cropped_images).to(self.device)
        with torch.no_grad():
            rec_preds = self.rec_model(rec_input)
        
        results = []
        for i, pred in enumerate(rec_preds.permute(1, 0, 2)):
            text = self._ctc_decode(pred)
            results.append({"box": scaled_polygons[i].tolist(), "text": text})
        
        visualized_image = self._visualize_results(original_image, results)
        return results, visualized_image

# --- 3. API 서버 전체에서 사용할 파이프라인 인스턴스 생성 ---
# TODO: 아래 경로들을 실제 파일 위치에 맞게 수정해주세요.
DET_WEIGHTS_PATH = "models/detection_model.pth"
REC_WEIGHTS_PATH = "models/recognition_model.pth"
CHAR_MAP_PATH = "models/korean_char_map.txt"

print("OCR 파이프라인 인스턴스를 초기화합니다 (DBNet v2)...")
pipeline_instance = OCR_Pipeline(
    det_weights=DET_WEIGHTS_PATH,
    rec_weights=REC_WEIGHTS_PATH,
    char_map_path=CHAR_MAP_PATH
)
print("OCR 파이프라인 초기화 완료.")

# --- 4. main.py에서 호출할 함수 ---
def run_ocr(image_bytes: bytes):
    results, visualized_image = pipeline_instance.run_from_bytes(image_bytes=image_bytes)
    return results, visualized_image