"""
Generate vest classification dataset from two sources:

  Source A — data/ (Roboflow vest detection):
    - Directly crop each bounding box → vest (class 1) / no_vest (class 0)
    - Directory: {train,valid,test}/{images,labels}/

  Source B — data2/ (Construction-PPE):
    - Crop Person (class 6) boxes; if vest (class 2) overlaps → vest, else no_vest
    - Directory: images/{train,val,test}/ + labels/{train,val,test}/

Output: vest_cls/{train,val,test}/{vest,no_vest}/
  - test split only from data2/test (independent holdout)
  - nc=1 single-class output: P(vest)
"""

import shutil
from pathlib import Path

import cv2

# =========================================================================
# 配置
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
DATA2_ROOT = PROJECT_ROOT / "data2"
OUTPUT_ROOT = PROJECT_ROOT / "vest_cls"

CROP_SIZE = 224
PADDING_RATIO = 0.05  # crop expands 5% outward

# --- data/ class mapping ---
DATA_VEST_CLS = 1
DATA_NOVEST_CLS = 0

# --- data2/ class mapping ---
D2_PERSON_CLS = 6
D2_VEST_CLS = 2


# =========================================================================
# 通用工具
# =========================================================================

def parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> dict[int, list[tuple[int, int, int, int]]]:
    """Read YOLO-format labels → {class_id: [(x1,y1,x2,y2), ...]} in pixel coords."""
    result: dict[int, list[tuple[int, int, int, int]]] = {}
    if not label_path.exists():
        return result

    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])

            x1 = int((xc - bw / 2) * img_w)
            y1 = int((yc - bh / 2) * img_h)
            x2 = int((xc + bw / 2) * img_w)
            y2 = int((yc + bh / 2) * img_h)

            result.setdefault(cls_id, []).append((x1, y1, x2, y2))

    return result


def crop_box(img, box: tuple[int, int, int, int], w: int, h: int):
    """Crop + pad + resize a single box. Returns None if invalid."""
    x1, y1, x2, y2 = box

    # padding
    pad_w = int((x2 - x1) * PADDING_RATIO)
    pad_h = int((y2 - y1) * PADDING_RATIO)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE))


def save_crop(crop, out_dir: Path, stem: str, suffix: str) -> bool:
    """Save a crop image. Returns True on success."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{suffix}.jpg"
    return cv2.imwrite(str(out_path), crop)


# =========================================================================
# 数据源 A: data/  — 直接裁剪标注框
# =========================================================================

def process_data_split(img_dir: Path, lbl_dir: Path, out_split: str) -> tuple[int, int]:
    """
    Crop every bounding box from data/.
    Returns (vest_count, no_vest_count).
    """
    vest_cnt = 0
    novest_cnt = 0
    processed = 0

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        labels = parse_yolo_labels(lbl_dir / (img_path.stem + ".txt"), w, h)

        stem = img_path.stem
        for box in labels.get(DATA_VEST_CLS, []):
            crop = crop_box(img, box, w, h)
            if crop is not None and save_crop(crop, OUTPUT_ROOT / out_split / "vest", stem, f"v{vest_cnt}"):
                vest_cnt += 1

        for box in labels.get(DATA_NOVEST_CLS, []):
            crop = crop_box(img, box, w, h)
            if crop is not None and save_crop(crop, OUTPUT_ROOT / out_split / "no_vest", stem, f"n{novest_cnt}"):
                novest_cnt += 1

        processed += 1

    print(f"  data/ → {out_split}: {processed} images → vest={vest_cnt}  no_vest={novest_cnt}")
    return vest_cnt, novest_cnt


def process_data_source():
    """Process data/ — train/valid go to train/val, test merges into val."""
    print("=" * 60)
    print("Source A: data/  (direct box cropping)")
    print("=" * 60)

    total_v = total_n = 0

    # train
    v, n = process_data_split(
        DATA_ROOT / "train" / "images",
        DATA_ROOT / "train" / "labels",
        "train",
    )
    total_v += v
    total_n += n

    # valid → val
    v, n = process_data_split(
        DATA_ROOT / "valid" / "images",
        DATA_ROOT / "valid" / "labels",
        "val",
    )
    total_v += v
    total_n += n

    # test → val (merge, not big enough to keep separate)
    v, n = process_data_split(
        DATA_ROOT / "test" / "images",
        DATA_ROOT / "test" / "labels",
        "val",
    )
    total_v += v
    total_n += n

    print(f"  data/ total: vest={total_v}  no_vest={total_n}\n")
    return total_v, total_n


# =========================================================================
# 数据源 B: data2/  — Person 框裁剪 + vest 重叠判断
# =========================================================================

def box_center_in_box(inner: tuple, outer: tuple) -> bool:
    """Check if center of `inner` box is inside `outer` box."""
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def process_data2_split(img_dir: Path, lbl_dir: Path, out_split: str) -> tuple[int, int]:
    """
    Crop Person boxes. If any vest box overlaps → vest, else → no_vest.
    Returns (vest_count, no_vest_count).
    """
    vest_cnt = 0
    novest_cnt = 0
    processed = 0

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        # skip orphaned labels (no matching image)
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        labels = parse_yolo_labels(label_path, w, h)
        person_boxes = labels.get(D2_PERSON_CLS, [])
        vest_boxes = labels.get(D2_VEST_CLS, [])

        stem = img_path.stem
        for pi, pbox in enumerate(person_boxes):
            is_vest = any(box_center_in_box(vb, pbox) for vb in vest_boxes)
            class_dir = "vest" if is_vest else "no_vest"

            crop = crop_box(img, pbox, w, h)
            if crop is None:
                continue

            if save_crop(crop, OUTPUT_ROOT / out_split / class_dir, stem, f"p{pi}"):
                if is_vest:
                    vest_cnt += 1
                else:
                    novest_cnt += 1

        processed += 1

    print(f"  data2/ → {out_split}: {processed} images → vest={vest_cnt}  no_vest={novest_cnt}")
    return vest_cnt, novest_cnt


def process_data2_source():
    """Process data2/ — train→train, val→val, test→test (holdout)."""
    print("=" * 60)
    print("Source B: data2/  (Person-crop + vest overlap)")
    print("=" * 60)

    total_v = total_n = 0

    v, n = process_data2_split(
        DATA2_ROOT / "images" / "train",
        DATA2_ROOT / "labels" / "train",
        "train",
    )
    total_v += v
    total_n += n

    v, n = process_data2_split(
        DATA2_ROOT / "images" / "val",
        DATA2_ROOT / "labels" / "val",
        "val",
    )
    total_v += v
    total_n += n

    # test → test (independent holdout)
    v, n = process_data2_split(
        DATA2_ROOT / "images" / "test",
        DATA2_ROOT / "labels" / "test",
        "test",
    )
    total_v += v
    total_n += n

    print(f"  data2/ total: vest={total_v}  no_vest={total_n}\n")
    return total_v, total_n


# =========================================================================
# Main
# =========================================================================

def main():
    # Clean old output
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
        print(f"Removed old: {OUTPUT_ROOT}\n")

    # Process both sources
    dv, dn = process_data_source()
    d2v, d2n = process_data2_source()

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------
    grand_v = dv + d2v
    grand_n = dn + d2n

    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    for split in ("train", "val", "test"):
        for cls_name in ("vest", "no_vest"):
            d = OUTPUT_ROOT / split / cls_name
            count = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"  {split}/{cls_name}: {count}")

    total = grand_v + grand_n
    print(f"\n  Total: {total}  (vest={grand_v}, no_vest={grand_n})")
    if total > 0:
        print(f"  vest ratio: {grand_v / total:.1%}")

    # -----------------------------------------------------------------
    # Write data.yaml (nc=1 single-class: just "vest")
    # -----------------------------------------------------------------
    yaml_content = f"""# Vest binary classification (single-class sigmoid output)
# nc=1 → model outputs P(vest), threshold at inference
path: {OUTPUT_ROOT.as_posix()}
train: train
val: val
test: test
nc: 1
names:
  0: vest
"""
    (OUTPUT_ROOT / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\nConfig written: {OUTPUT_ROOT / 'data.yaml'}")
    print("Done.")


if __name__ == "__main__":
    main()
