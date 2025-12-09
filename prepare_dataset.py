import os
import glob
import random
import shutil

random.seed(42)

# 1. 원본 데이터가 있는 폴더 (지금 네 zip 풀린 구조 기준)
SRC_ROOT = "Training"   # 같은 폴더에 Training/Apple, Training/Banana ... 있다고 가정

# 2. 우리가 최종으로 쓰고 싶은 클래스들
#    -> 지금 데이터에는 Apple / Banana / Strawberry는 확실히 있음
#    -> Grape가 없다면 일단 주석 처리해두고 3종만 학습해도 됨.
TARGET_CLASSES = ["Apple", "Banana", "Strawberry"]
CANONICAL_NAMES = {  # 폴더 이름 -> 우리가 쓸 라벨 이름
    "Apple": "apple",
    "Banana": "banana",
    "Strawberry": "strawberry",
    # "Grape": "grape",  # 나중에 Grape 폴더가 생기면 추가하면 됨
}

DEST_ROOT = "data"      # 결과가 저장될 루트
TRAIN_RATIO = 0.8       # 학습:검증 비율 8:2
MAX_PER_CLASS = None    # 과일당 최대 이미지 수 제한 (None이면 전부 사용)


def main():
    # 0. 폴더 존재 확인
    if not os.path.isdir(SRC_ROOT):
        raise FileNotFoundError(f"원본 폴더 {SRC_ROOT} 를 찾을 수 없습니다. 위치를 다시 확인하세요.")

    # 1. 폴더 구조 만들기
    for split in ["train", "val"]:
        for src_name in TARGET_CLASSES:
            cname = CANONICAL_NAMES[src_name]
            out_dir = os.path.join(DEST_ROOT, split, cname)
            os.makedirs(out_dir, exist_ok=True)
            print(f"[폴더 생성] {out_dir}")

    # 2. 각 클래스별 이미지 모으기
    files_by_class = {}

    for src_name in TARGET_CLASSES:
        src_dir = os.path.join(SRC_ROOT, src_name)
        if not os.path.isdir(src_dir):
            print(f"[경고] {src_dir} 폴더가 없습니다. 건너뜁니다.")
            continue

        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            imgs.extend(glob.glob(os.path.join(src_dir, ext)))

        if not imgs:
            print(f"[경고] {src_dir} 안에 이미지 파일이 없습니다.")
            continue

        files_by_class[src_name] = imgs
        print(f"[수집] {src_name}: {len(imgs)}장 발견")

    # 3. train / val 나누고 복사
    for src_name, paths in files_by_class.items():
        cname = CANONICAL_NAMES[src_name]
        print(f"\n=== {src_name} -> {cname} ===")
        print(f"총 {len(paths)}장")

        # 필요하면 개수 제한
        if MAX_PER_CLASS is not None and len(paths) > MAX_PER_CLASS:
            paths = random.sample(paths, MAX_PER_CLASS)
            print(f"MAX_PER_CLASS={MAX_PER_CLASS} 적용 -> {len(paths)}장만 사용")

        random.shuffle(paths)

        n_train = int(len(paths) * TRAIN_RATIO)
        train_paths = paths[:n_train]
        val_paths = paths[n_train:]

        print(f" -> train: {len(train_paths)}장, val: {len(val_paths)}장")

        # 파일 복사
        for src in train_paths:
            dst = os.path.join(DEST_ROOT, "train", cname, os.path.basename(src))
            shutil.copy2(src, dst)

        for src in val_paths:
            dst = os.path.join(DEST_ROOT, "val", cname, os.path.basename(src))
            shutil.copy2(src, dst)

    print("\n✅ 데이터 분리가 완료되었습니다!")
    print(f"- 학습 데이터: {os.path.join(DEST_ROOT, 'train')}")
    print(f"- 검증 데이터: {os.path.join(DEST_ROOT, 'val')}")


if __name__ == "__main__":
    main()
