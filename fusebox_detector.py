import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def detect_components(image):
    """
    이미지에서 HSV 색상 기반으로 주요 부품(파란 몸체, 노란 퓨즈, 빨간 퓨즈)을 탐지합니다.
    노이즈 감소를 위해 가우시안 블러와 형태학적 변환을 적용합니다.
    """
    # 노이즈 감소를 위해 가우시안 블러 적용
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    components = {}

    # 각 부품의 HSV 색상 범위 정의
    # 참고: 이 값들은 조명 환경에 따라 미세 조정이 필요할 수 있습니다.
    color_ranges = {
        "blue_body": ([90, 80, 80], [130, 255, 255]),
        "yellow_fuse": ([20, 100, 100], [30, 255, 255]),
        "red_fuse": ([0, 120, 120], [10, 255, 255]) # 빨간색은 0과 180 주변에 걸쳐있을 수 있음
    }

    for name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # 빨간색의 경우, 색상환(Hue)의 양 끝에 걸쳐있을 수 있어 두 범위를 합쳐야 할 수 있음
        if name == "red_fuse":
            lower2 = np.array([170, 120, 120], dtype="uint8")
            upper2 = np.array([180, 255, 255], dtype="uint8")
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)

        # 형태학적 변환으로 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # 작은 노이즈 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 객체 내의 작은 구멍 메우기

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100: # 최소 면적 필터링
                components[name] = cv2.boundingRect(largest_contour)
    
    return components

def compare_fuse_boxes(data_img_path, target_img_path, ssim_threshold=0.8, position_tolerance=20, size_tolerance_ratio=0.5):
    # 1. 이미지 로드
    data_img = cv2.imread(data_img_path)
    target_img = cv2.imread(target_img_path)

    if data_img is None or target_img is None:
        return "ERROR: Image not found", None

    # 2. 각 이미지에서 부품 탐지
    data_components = detect_components(data_img)
    target_components = detect_components(target_img)

    annotated_image = target_img.copy()
    defects = []
    processed_defects = set() # 중복된 불량 표시를 피하기 위함

    # 3. 기준 부품과 대상 부품 비교
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

        # 위치 비교
        if abs(dx - tx) > position_tolerance or abs(dy - ty) > position_tolerance:
            if (name, 'misplaced') not in processed_defects:
                defects.append(f"{name} 위치 불량")
                cv2.rectangle(annotated_image, (tx, ty), (tx + tw, ty + th), (0, 165, 255), 2) # 주황색
                cv2.putText(annotated_image, f"{name} Misplaced", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                processed_defects.add((name, 'misplaced'))

        # 크기 비교 (너비와 높이가 각각 허용 오차를 벗어나는지 확인)
        width_diff = abs(dw - tw) / float(dw)
        height_diff = abs(dh - th) / float(dh)
        if width_diff > size_tolerance_ratio or height_diff > size_tolerance_ratio:
            if (name, 'size') not in processed_defects:
                defects.append(f"{name} 크기 불량")
                cv2.rectangle(annotated_image, (tx, ty), (tx + tw, ty + th), (255, 0, 255), 2) # 보라색
                cv2.putText(annotated_image, f"{name} Size Mismatch", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                processed_defects.add((name, 'size'))

        # 모양(SSIM) 비교
        data_roi = data_img[dy:dy+dh, dx:dx+dw]
        target_roi = target_img[ty:ty+th, tx:tx+tw]
        
        # SSIM 비교를 위해 크기 통일
        target_roi_resized = cv2.resize(target_roi, (dw, dh))

        data_gray_roi = cv2.cvtColor(data_roi, cv2.COLOR_BGR2GRAY)
        target_gray_roi = cv2.cvtColor(target_roi_resized, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(data_gray_roi, target_gray_roi, full=True)

        if score < ssim_threshold:
            if (name, 'shape') not in processed_defects:
                defects.append(f"{name} 모양 불량 (SSIM: {score:.2f})")
                cv2.rectangle(annotated_image, (tx, ty), (tx + tw, ty + th), (0, 255, 255), 2) # 노란색
                cv2.putText(annotated_image, f"{name} Shape Mismatch", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                processed_defects.add((name, 'shape'))

    # 4. 추가된 부품 검사
    for name, target_bbox in target_components.items():
        if name not in data_components:
            if (name, 'extra') not in processed_defects:
                defects.append(f"{name} 추가됨")
                tx, ty, tw, th = target_bbox
                cv2.rectangle(annotated_image, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2) # 초록색
                cv2.putText(annotated_image, f"{name} Extra", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                processed_defects.add((name, 'extra'))

    # 5. 최종 결과 생성
    if not defects:
        result_string = "정상품"
    else:
        # 중복 제거 후 최종 결과 문자열 생성
        result_string = "불량: " + ", ".join(sorted(list(set(defects))))

    print(f"결과: {result_string}")
    return result_string, annotated_image


if __name__ == "__main__":
    normal_image = "normal_fusebox.jpg"
    test_image_good = "test_fusebox_good.jpg"
    test_image_bad_shape = "test_fusebox_missing.jpg"
    test_image_bad_color = "test_fusebox_wrong_color.jpg"

    print("--- 테스트 1: 정상품 비교 ---")
    result1, annotated_image1 = compare_fuse_boxes(normal_image, test_image_good)
    print(f"최종 판정: {result1}")
    if "불량" in result1:
        cv2.imwrite("diff_bbox_test1.png", annotated_image1)
        print("결과 이미지 저장됨: diff_bbox_test1.png")

    print("\n--- 테스트 2: 부품 누락 불량품 비교 ---")
    result2, annotated_image2 = compare_fuse_boxes(normal_image, test_image_bad_shape)
    print(f"최종 판정: {result2}")
    if "불량" in result2:
        cv2.imwrite("diff_bbox_test2.png", annotated_image2)
        print("결과 이미지 저장됨: diff_bbox_test2.png")

    print("\n--- 테스트 3: 색상 다른 불량품 비교 ---")
    result3, annotated_image3 = compare_fuse_boxes(normal_image, test_image_bad_color)
    print(f"최종 판정: {result3}")
    if "불량" in result3:
        cv2.imwrite("diff_bbox_test3.png", annotated_image3)
        print("결과 이미지 저장됨: diff_bbox_test3.png")
