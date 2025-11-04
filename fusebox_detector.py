import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random
import time
import os
import requests
import threading
from flask import Flask, jsonify

# --- 기존 이미지 처리 함수 (변경 없음) ---

def detect_components(image):
    """
    이미지에서 HSV 색상 기반으로 주요 부품(파란 몸체, 노란 퓨즈, 빨간 퓨즈)을 탐지합니다.
    노이즈 감소를 위해 가우시안 블러와 형태학적 변환을 적용합니다.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    components = {}
    color_ranges = {
        "blue_body": ([90, 80, 80], [130, 255, 255]),
        "yellow_fuse": ([20, 100, 100], [30, 255, 255]),
        "red_fuse": ([0, 120, 120], [10, 255, 255])
    }
    for name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)
        if name == "red_fuse":
            lower2 = np.array([170, 120, 120], dtype="uint8")
            upper2 = np.array([180, 255, 255], dtype="uint8")
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                components[name] = cv2.boundingRect(largest_contour)
    return components

def compare_fuse_boxes(data_img_path, target_img_path, ssim_threshold=0.8, position_tolerance=20, size_tolerance_ratio=0.5):
    data_img = cv2.imread(data_img_path)
    target_img = cv2.imread(target_img_path)
    if data_img is None or target_img is None:
        return "ERROR: Image not found", None
    data_components = detect_components(data_img)
    target_components = detect_components(target_img)
    annotated_image = target_img.copy()
    defects = []
    processed_defects = set()
    for name, data_bbox in data_components.items():
        dx, dy, dw, dh = data_bbox
        if name not in target_components:
            if (name, 'missing') not in processed_defects:
                defects.append(f"{name} 누락")
                cv2.rectangle(annotated_image, (dx, dy), (dx + dw, dy + dh), (0, 0, 255), 2)
                cv2.putText(annotated_image, f"{name} Missing", (dx, dy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                processed_defects.add((name, 'missing'))
            continue
        target_bbox = target_components[name]
        tx, ty, tw, th = target_bbox
        if abs(dx - tx) > position_tolerance or abs(dy - ty) > position_tolerance:
            if (name, 'misplaced') not in processed_defects:
                defects.append(f"{name} 위치 불량")
                cv2.rectangle(annotated_image, (tx, ty), (tx + tw, ty + th), (0, 165, 255), 2)
                cv2.putText(annotated_image, f"{name} Misplaced", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                processed_defects.add((name, 'misplaced'))
        width_diff = abs(dw - tw) / float(dw)
        height_diff = abs(dh - th) / float(dh)
        if width_diff > size_tolerance_ratio or height_diff > size_tolerance_ratio:
            if (name, 'size') not in processed_defects:
                defects.append(f"{name} 크기 불량")
                cv2.rectangle(annotated_image, (tx, ty), (tx + tw, ty + th), (255, 0, 255), 2)
                cv2.putText(annotated_image, f"{name} Size Mismatch", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                processed_defects.add((name, 'size'))
        data_roi = data_img[dy:dy+dh, dx:dx+dw]
        target_roi = target_img[ty:ty+th, tx:tx+tw]
        target_roi_resized = cv2.resize(target_roi, (dw, dh))
        data_gray_roi = cv2.cvtColor(data_roi, cv2.COLOR_BGR2GRAY)
        target_gray_roi = cv2.cvtColor(target_roi_resized, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(data_gray_roi, target_gray_roi, full=True)
        if score < ssim_threshold:
            if (name, 'shape') not in processed_defects:
                defects.append(f"{name} 모양 불량 (SSIM: {score:.2f})")
                cv2.rectangle(annotated_image, (tx, ty), (tx + tw, ty + th), (0, 255, 255), 2)
                cv2.putText(annotated_image, f"{name} Shape Mismatch", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                processed_defects.add((name, 'shape'))
    for name, target_bbox in target_components.items():
        if name not in data_components:
            if (name, 'extra') not in processed_defects:
                defects.append(f"{name} 추가됨")
                tx, ty, tw, th = target_bbox
                cv2.rectangle(annotated_image, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"{name} Extra", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                processed_defects.add((name, 'extra'))
    if not defects:
        result_string = "정상품"
    else:
        result_string = "불량: " + ", ".join(sorted(list(set(defects))))
    print(f"결과: {result_string}")
    return result_string, annotated_image

# --- 새로운 웹 서버 및 처리 로직 ---

app = Flask(__name__)

def run_detection_process():
    """
    메인 이미지 처리 및 API 전송 로직.
    백그라운드 스레드에서 실행됩니다.
    """
    print("백그라운드 처리 시작...")
    # 메인 서버의 API 엔드포인트 URL (라즈베리파이가 결과를 보고할 곳)
    # 이 URL은 실제 Node.js 서버의 주소여야 합니다.
    NODE_SERVER_URL = "http://<Node.js_서버_IP>:3000/defects"

    normal_image = "target/normal_fusebox.jpg"
    # target 폴더에 있는 모든 jpg 파일을 테스트 대상으로 합니다.
    test_images = [os.path.join("target", f) for f in os.listdir("target") if f.endswith('.jpg') and f.startswith('test')]
    random.shuffle(test_images)

    for i, test_image_path in enumerate(test_images, 1):
        print(f"--- 테스트 {i}/{len(test_images)}: {test_image_path} 비교 ---")
        result, annotated_image = compare_fuse_boxes(normal_image, test_image_path)
        print(f"최종 판정: {result}")

        is_defective = "불량" in result
        
        data = {
            "device_id": f"RASPBERRY_PI_01", # 기기 ID는 고정하거나 설정에 따라 변경
            "value": 101 if is_defective else 99,
        }

        try:
            with open(test_image_path, 'rb') as image_file:
                files = {'image': (os.path.basename(test_image_path), image_file, 'image/jpeg')}
                # 결과를 Node.js 서버로 전송
                response = requests.post(NODE_SERVER_URL, data=data, files=files)
                response.raise_for_status()
                print(f"Node.js 서버 응답: {response.json()}")

        except requests.exceptions.RequestException as e:
            print(f"Node.js 서버로 결과 전송 실패: {e}")
        except IOError as e:
            print(f"이미지 파일({test_image_path})을 열 수 없습니다: {e}")
        
        if is_defective and annotated_image is not None:
            base_name = os.path.basename(test_image_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_filename = f"result/diff_bbox_{file_name_without_ext}.png"
            cv2.imwrite(output_filename, annotated_image)
            print(f"결과 이미지 저장됨: {output_filename}")
        
        print("\n")
        time.sleep(3) # 필요하다면 대기 시간 조절

    print("모든 테스트 완료.")

@app.route('/start', methods=['POST'])
def start_processing():
    """
    Node.js 서버로부터 작업 시작 신호를 받는 API 엔드포인트.
    """
    print("'/start' 요청 수신. 감지 프로세스를 백그라운드에서 시작합니다.")
    # 이미 처리 중인 작업이 있는지 확인할 수 있는 로직을 추가할 수도 있습니다.
    thread = threading.Thread(target=run_detection_process)
    thread.start()
    return jsonify({"message": "Detection process started in the background."}), 202

if __name__ == "__main__":
    # 0.0.0.0으로 서버를 실행하여 외부 네트워크에서 접근할 수 있도록 합니다.
    # 라즈베리파이와 Node.js 서버가 동일한 네트워크에 있어야 합니다.
    # 포트는 다른 서비스와 겹치지 않는 번호로 설정합니다 (예: 5001).
    app.run(host='0.0.0.0', port=5001)
