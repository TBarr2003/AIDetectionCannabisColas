#!/usr/bin/env python3
"""
Single-class YOLOv8 TP / FP / FN analysis with CSV export.
Designed for Lambda06 + YOLOv8 validation workflows.
"""

from pathlib import Path
import argparse
import csv

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# -------------------------
# Geometry helpers
# -------------------------

def yolo_to_xyxy(x, y, w, h):
    return (x - w/2, y - h/2, x + w/2, y + h/2)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = a_area + b_area - inter
    return 0.0 if denom <= 0 else inter / denom

# -------------------------
# IO helpers
# -------------------------

def read_gt(path: Path):
    """GT format: cls x y w h"""
    boxes = []
    if not path.exists():
        return boxes

    txt = path.read_text().strip()
    if not txt:
        return boxes

    for line in txt.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        x, y, w, h = map(float, p[1:5])
        boxes.append(yolo_to_xyxy(x, y, w, h))
    return boxes

def read_preds(path: Path, conf_thres: float):
    """Pred format: cls x y w h conf (conf optional)"""
    preds = []
    if not path.exists():
        return preds

    txt = path.read_text().strip()
    if not txt:
        return preds

    for line in txt.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        x, y, w, h = map(float, p[1:5])
        conf = float(p[5]) if len(p) >= 6 else 1.0
        if conf >= conf_thres:
            preds.append((conf, yolo_to_xyxy(x, y, w, h)))

    preds.sort(key=lambda t: t[0], reverse=True)
    return preds

# -------------------------
# Matching logic
# -------------------------

def greedy_match(preds, gts, iou_thres):
    used_gt = [False] * len(gts)
    matched_pred = [False] * len(preds)
    matches = []

    for pi, (conf, pbox) in enumerate(preds):
        best_iou, best_gi = 0.0, -1
        for gi, gtbox in enumerate(gts):
            if used_gt[gi]:
                continue
            v = iou(pbox, gtbox)
            if v >= iou_thres and v > best_iou:
                best_iou, best_gi = v, gi

        if best_gi != -1:
            used_gt[best_gi] = True
            matched_pred[pi] = True
            matches.append((pi, best_gi, best_iou))

    tp = sum(matched_pred)
    fp = len(preds) - tp
    fn = len(gts) - sum(used_gt)

    return tp, fp, fn, matched_pred, used_gt, matches

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="tp_fp_fn_results.csv")
    args = ap.parse_args()

    images_dir = Path(args.images)
    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)

    stems = sorted([
        p.stem for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])

    total_tp = total_fp = total_fn = 0
    img_TP = img_FP = img_FN = img_TN = 0
    rows = []

    for stem in stems:
        gts = read_gt(gt_dir / f"{stem}.txt")
        preds = read_preds(pred_dir / f"{stem}.txt", args.conf)

        tp, fp, fn, matched_pred, used_gt, matches = greedy_match(
            preds, gts, args.iou
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # TP rows
        for pi, gi, iou_val in matches:
            conf, box = preds[pi]
            rows.append({
                "image": stem,
                "type": "TP",
                "confidence": conf,
                "iou_gt": iou_val,
                "iou_tp": "",
                "x1": box[0], "y1": box[1],
                "x2": box[2], "y2": box[3],
            })

        tp_boxes = [preds[pi][1] for pi, _, _ in matches]

        # FP rows
        for i, (conf, box) in enumerate(preds):
            if not matched_pred[i]:
                iou_gt = max([iou(box, g) for g in gts], default=0.0)
                iou_tp = max([iou(box, tpb) for tpb in tp_boxes], default=0.0)
                rows.append({
                    "image": stem,
                    "type": "FP",
                    "confidence": conf,
                    "iou_gt": iou_gt,
                    "iou_tp": iou_tp,
                    "x1": box[0], "y1": box[1],
                    "x2": box[2], "y2": box[3],
                })

        # FN rows
        for gi, gtbox in enumerate(gts):
            if not used_gt[gi]:
                rows.append({
                    "image": stem,
                    "type": "FN",
                    "confidence": "",
                    "iou_gt": 0.0,
                    "iou_tp": "",
                    "x1": gtbox[0], "y1": gtbox[1],
                    "x2": gtbox[2], "y2": gtbox[3],
                })

        has_gt = len(gts) > 0
        has_pred = len(preds) > 0

        if not has_gt and not has_pred:
            img_TN += 1
        elif not has_gt and has_pred:
            img_FP += 1
        elif has_gt and tp > 0:
            img_TP += 1
        else:
            img_FN += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "type", "confidence",
                "iou_gt", "iou_tp",
                "x1", "y1", "x2", "y2"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Detection Summary (single class) ===")
    print(f"Images evaluated: {len(stems)}")
    print(f"TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\n=== Image-level counts ===")
    print(f"TP images: {img_TP}")
    print(f"FP images: {img_FP}")
    print(f"FN images: {img_FN}")
    print(f"TN images: {img_TN}")

    print(f"\n📄 CSV written to: {args.out}")

if __name__ == "__main__":
    main()
