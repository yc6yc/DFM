from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import streamlit as st
from PIL import Image, ImageDraw


DEFAULT_TARGET_FRAMES = 10
DEFAULT_IMAGE_EXT = ".jpg"
DEFAULT_ANNOTATION_EXT = ".xml"


@dataclass
class FrameRecord:
    case_id: str
    stem: str
    image_path: Path
    annotation_path: Path
    boxes: List[Tuple[int, int, int, int]]


@dataclass
class ScanReport:
    records_by_case: Dict[str, List[FrameRecord]]
    scan_errors: List[str]


def frame_sort_key(stem: str) -> Tuple[int, str]:
    # 优先按末尾数字排序，例如 200_10 > 200_2。
    tail = stem.split("_")[-1]
    if tail.isdigit():
        return int(tail), stem
    return 10**9, stem


def parse_voc_boxes(xml_path: Path) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = bndbox.findtext("xmin")
        ymin = bndbox.findtext("ymin")
        xmax = bndbox.findtext("xmax")
        ymax = bndbox.findtext("ymax")
        if not all([xmin, ymin, xmax, ymax]):
            continue

        try:
            x1 = int(float(xmin))
            y1 = int(float(ymin))
            x2 = int(float(xmax))
            y2 = int(float(ymax))
        except ValueError:
            continue

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return boxes


def scan_dataset(dataset_root: Path, image_ext: str, annotation_ext: str) -> ScanReport:
    records_by_case: Dict[str, List[FrameRecord]] = {}
    scan_errors: List[str] = []

    if not dataset_root.exists() or not dataset_root.is_dir():
        return ScanReport({}, [f"数据集路径不存在或不是目录: {dataset_root}"])

    case_dirs = sorted(
        [d for d in dataset_root.iterdir() if d.is_dir() and (d / "images").is_dir() and (d / "annotations").is_dir()],
        key=lambda p: p.name,
    )

    for case_dir in case_dirs:
        case_id = case_dir.name
        images_dir = case_dir / "images"
        ann_dir = case_dir / "annotations"

        image_map: Dict[str, Path] = {}
        for img in images_dir.glob(f"*{image_ext}"):
            image_map[img.stem] = img

        case_records: List[FrameRecord] = []
        for ann in ann_dir.glob(f"*{annotation_ext}"):
            stem = ann.stem
            image_path = image_map.get(stem)
            if image_path is None:
                continue

            try:
                boxes = parse_voc_boxes(ann)
            except Exception as exc:
                scan_errors.append(f"{case_id}/{ann.name} 解析失败: {exc}")
                continue

            # 只保留有标注框的帧。
            if not boxes:
                continue

            case_records.append(
                FrameRecord(
                    case_id=case_id,
                    stem=stem,
                    image_path=image_path,
                    annotation_path=ann,
                    boxes=boxes,
                )
            )

        case_records.sort(key=lambda r: frame_sort_key(r.stem))
        if case_records:
            records_by_case[case_id] = case_records

    return ScanReport(records_by_case, scan_errors)


def draw_boxes_on_image(image_path: Path, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=4)
    return image


def ensure_trash_paths(dataset_root: Path, case_id: str) -> Tuple[Path, Path, Path]:
    trash_case_root = dataset_root / ".trash" / case_id
    trash_images = trash_case_root / "images"
    trash_annotations = trash_case_root / "annotations"
    trash_images.mkdir(parents=True, exist_ok=True)
    trash_annotations.mkdir(parents=True, exist_ok=True)
    return trash_case_root, trash_images, trash_annotations


def append_delete_log(dataset_root: Path, row: Dict[str, str]) -> None:
    log_path = dataset_root / ".trash" / "deletion_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "case_id",
        "stem",
        "image_src",
        "annotation_src",
        "image_dst",
        "annotation_dst",
        "status",
        "error",
    ]

    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def move_to_trash(dataset_root: Path, frame: FrameRecord) -> Tuple[bool, str]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    _, trash_images, trash_annotations = ensure_trash_paths(dataset_root, frame.case_id)

    img_dst = trash_images / frame.image_path.name
    ann_dst = trash_annotations / frame.annotation_path.name

    row = {
        "timestamp": timestamp,
        "case_id": frame.case_id,
        "stem": frame.stem,
        "image_src": str(frame.image_path),
        "annotation_src": str(frame.annotation_path),
        "image_dst": str(img_dst),
        "annotation_dst": str(ann_dst),
        "status": "ok",
        "error": "",
    }

    try:
        if not frame.image_path.exists():
            raise FileNotFoundError(f"图片不存在: {frame.image_path}")
        if not frame.annotation_path.exists():
            raise FileNotFoundError(f"标注不存在: {frame.annotation_path}")

        shutil.move(str(frame.image_path), str(img_dst))
        shutil.move(str(frame.annotation_path), str(ann_dst))
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        append_delete_log(dataset_root, row)
        return False, str(exc)

    append_delete_log(dataset_root, row)
    return True, "已移动到 .trash"


def get_three_frame_window(frames: List[FrameRecord], group_idx: int) -> List[FrameRecord]:
    start = group_idx * 3
    end = start + 3
    return frames[start:end]


def reset_delete_state() -> None:
    st.session_state.pop("pending_delete_stem", None)


def refresh_scan_report_with_position(
    dataset_root: Path,
    image_ext: str,
    annotation_ext: str,
    preserve_case_id: Optional[str],
    preserve_group_idx: int,
) -> None:
    report = scan_dataset(dataset_root, image_ext, annotation_ext)
    st.session_state["scan_report"] = report

    case_ids = sorted(report.records_by_case.keys(), key=lambda c: int(c) if c.isdigit() else c)
    if not case_ids:
        st.session_state["selected_case_id"] = None
        st.session_state["group_idx"] = 0
        reset_delete_state()
        return

    selected_case_id = preserve_case_id if preserve_case_id in case_ids else case_ids[0]
    st.session_state["selected_case_id"] = selected_case_id

    frames = report.records_by_case[selected_case_id]
    max_group = max(0, (len(frames) - 1) // 3)
    st.session_state["group_idx"] = min(max(preserve_group_idx, 0), max_group)
    reset_delete_state()


def ui_sidebar() -> Tuple[Optional[Path], int, str, str, bool]:
    st.sidebar.header("数据集设置")

    dataset_root_input = st.sidebar.text_input("数据集根目录", value=str(Path.cwd()))
    target_frames = st.sidebar.number_input("每病例目标保留帧数", min_value=1, max_value=9999, value=DEFAULT_TARGET_FRAMES)
    image_ext = st.sidebar.text_input("图片后缀", value=DEFAULT_IMAGE_EXT).strip() or DEFAULT_IMAGE_EXT
    annotation_ext = st.sidebar.text_input("标注后缀", value=DEFAULT_ANNOTATION_EXT).strip() or DEFAULT_ANNOTATION_EXT

    do_scan = st.sidebar.button("导入/刷新数据集", use_container_width=True)

    dataset_root = None
    if dataset_root_input:
        dataset_root = Path(dataset_root_input).expanduser()

    return dataset_root, int(target_frames), image_ext, annotation_ext, do_scan


def render_frame_card(dataset_root: Optional[Path], frame: FrameRecord, col, op_key_prefix: str) -> None:
    with col:
        st.markdown(f"**{frame.stem}**")

        try:
            image = draw_boxes_on_image(frame.image_path, frame.boxes)
            st.image(image, use_container_width=True)
        except Exception as exc:
            st.error(f"图像加载或绘制失败: {exc}")
            return

        st.caption(f"框数量: {len(frame.boxes)}")

        delete_button_key = f"delete_{op_key_prefix}_{frame.stem}"
        confirm_button_key = f"confirm_{op_key_prefix}_{frame.stem}"
        cancel_button_key = f"cancel_{op_key_prefix}_{frame.stem}"

        pending_stem = st.session_state.get("pending_delete_stem")
        if pending_stem == frame.stem:
            st.warning(f"确认删除 {frame.stem} ?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("确认删除", key=confirm_button_key, use_container_width=True):
                    if dataset_root is None:
                        st.error("数据集根目录无效，无法删除")
                    else:
                        ok, msg = move_to_trash(dataset_root, frame)
                        if ok:
                            st.success(f"{frame.stem}: {msg}")
                            st.session_state["auto_rescan_requested"] = True
                        else:
                            st.error(f"{frame.stem}: 删除失败 - {msg}")
                    reset_delete_state()
                    st.rerun()
            with c2:
                if st.button("取消", key=cancel_button_key, use_container_width=True):
                    reset_delete_state()
                    st.rerun()
        else:
            if st.button("删除此帧", key=delete_button_key, use_container_width=True):
                st.session_state["pending_delete_stem"] = frame.stem
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="病例帧筛选工具", layout="wide")
    st.title("病例帧筛选工具")
    st.caption("同一病例连续三帧对比 | 红色框叠加 | 删除时同步移动图片与 VOC XML 到 .trash")

    dataset_root, target_frames, image_ext, annotation_ext, do_scan = ui_sidebar()

    if "scan_report" not in st.session_state:
        st.session_state["scan_report"] = None
    if "selected_case_id" not in st.session_state:
        st.session_state["selected_case_id"] = None
    if "group_idx" not in st.session_state:
        st.session_state["group_idx"] = 0
    if "auto_rescan_requested" not in st.session_state:
        st.session_state["auto_rescan_requested"] = False

    if do_scan:
        if dataset_root is None:
            st.error("请先输入数据集根目录")
        else:
            refresh_scan_report_with_position(
                dataset_root=dataset_root,
                image_ext=image_ext,
                annotation_ext=annotation_ext,
                preserve_case_id=st.session_state.get("selected_case_id"),
                preserve_group_idx=int(st.session_state.get("group_idx", 0)),
            )
    elif st.session_state.get("auto_rescan_requested"):
        st.session_state["auto_rescan_requested"] = False
        if dataset_root is not None:
            refresh_scan_report_with_position(
                dataset_root=dataset_root,
                image_ext=image_ext,
                annotation_ext=annotation_ext,
                preserve_case_id=st.session_state.get("selected_case_id"),
                preserve_group_idx=int(st.session_state.get("group_idx", 0)),
            )

    report: Optional[ScanReport] = st.session_state.get("scan_report")
    if report is None:
        st.info("请先在左侧点击“导入/刷新数据集”")
        return

    if report.scan_errors:
        with st.expander(f"扫描警告 ({len(report.scan_errors)})", expanded=False):
            for e in report.scan_errors[:200]:
                st.write(f"- {e}")
            if len(report.scan_errors) > 200:
                st.write(f"... 其余 {len(report.scan_errors) - 200} 条未展示")

    case_ids = sorted(report.records_by_case.keys(), key=lambda c: int(c) if c.isdigit() else c)
    if not case_ids:
        st.warning("未找到可用帧：请确认数据结构正确，且帧存在可解析的标注框")
        return

    if st.session_state["selected_case_id"] not in case_ids:
        st.session_state["selected_case_id"] = case_ids[0]

    selected_case = st.selectbox(
        "选择病例",
        options=case_ids,
        index=case_ids.index(st.session_state["selected_case_id"]),
        key="case_selector",
    )
    st.session_state["selected_case_id"] = selected_case

    current_case_index = case_ids.index(selected_case)
    has_next_case = current_case_index < len(case_ids) - 1
    if st.button("下一病例首帧", use_container_width=False, disabled=not has_next_case):
        st.session_state["selected_case_id"] = case_ids[current_case_index + 1]
        st.session_state["group_idx"] = 0
        reset_delete_state()
        st.rerun()

    frames = report.records_by_case[selected_case]
    remaining = len(frames)
    gap = max(0, remaining - target_frames)

    c_meta1, c_meta2, c_meta3 = st.columns(3)
    with c_meta1:
        st.metric("当前病例有效帧", remaining)
    with c_meta2:
        st.metric("目标保留帧", target_frames)
    with c_meta3:
        st.metric("需继续删除", gap)

    max_group = max(0, (remaining - 1) // 3)
    if st.session_state["group_idx"] > max_group:
        st.session_state["group_idx"] = max_group

    nav1, nav2, nav3 = st.columns([1, 1, 4])
    with nav1:
        if st.button("上一组", use_container_width=True, disabled=st.session_state["group_idx"] <= 0):
            st.session_state["group_idx"] -= 1
            reset_delete_state()
            st.rerun()
    with nav2:
        if st.button("下一组", use_container_width=True, disabled=st.session_state["group_idx"] >= max_group):
            st.session_state["group_idx"] += 1
            reset_delete_state()
            st.rerun()
    with nav3:
        st.write(f"当前组: {st.session_state['group_idx'] + 1} / {max_group + 1}")

    window = get_three_frame_window(frames, st.session_state["group_idx"])
    cols = st.columns(3)
    for idx, col in enumerate(cols):
        if idx < len(window):
            render_frame_card(dataset_root, window[idx], col, op_key_prefix=f"g{st.session_state['group_idx']}")
        else:
            with col:
                st.info("该位置无帧")

    if dataset_root is not None:
        st.caption(f"日志文件: {dataset_root / '.trash' / 'deletion_log.csv'}")


if __name__ == "__main__":
    main()
