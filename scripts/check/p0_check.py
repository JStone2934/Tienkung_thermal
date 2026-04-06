#!/usr/bin/env python3
"""P0 实机核对：仅检查「离线读仓库无法确定」的接口与话题数据。

请在机器人侧已 source ROS2 工作空间、且 `/leg/status` 有数据时运行。
代码仓库内的映射、插件逻辑等请在开发机用文档与审阅完成，不在此脚本中重复。

检查内容概览：
  - `ros2 interface show bodyctrl_msgs/msg/MotorStatusMsg`：字段类型与变长/标量假设
  - `ros2 topic echo /leg/status`：单帧能否解析 12 路腿电机并映射到 T_leg 顺序
  - 短时连续 echo：消息 header 时间间隔（可选与 `--dt` 对比）

用法::

    python scripts/check/p0_check.py
    python scripts/check/p0_check.py --dt 0.0025
    python scripts/check/p0_check.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

import yaml

PASS = "PASS"
FAIL = "FAIL"
PENDING = "PENDING"

# 与 `configs/leg_index_mapping.yaml` 中 motor_names 一致，用于实机样本 → T_leg[i] 核对
T_LEG_MOTOR_NAMES = [
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "hip_pitch_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
]

# Deploy 侧 CAN id → 名称（与 bodyIdMap / 消息中 name 字段一致）
DEPLOY_LEG_IDS = {
    51: "l_hip_roll",
    52: "l_hip_pitch",
    53: "l_hip_yaw",
    54: "l_knee",
    55: "l_ankle_pitch",
    56: "l_ankle_roll",
    61: "r_hip_roll",
    62: "r_hip_pitch",
    63: "r_hip_yaw",
    64: "r_knee",
    65: "r_ankle_pitch",
    66: "r_ankle_roll",
}

DEPLOY_TO_LAB_NAME = {
    "l_hip_roll": "hip_roll_l_joint",
    "l_hip_pitch": "hip_pitch_l_joint",
    "l_hip_yaw": "hip_yaw_l_joint",
    "l_knee": "knee_pitch_l_joint",
    "l_ankle_pitch": "ankle_pitch_l_joint",
    "l_ankle_roll": "ankle_roll_l_joint",
    "r_hip_roll": "hip_roll_r_joint",
    "r_hip_pitch": "hip_pitch_r_joint",
    "r_hip_yaw": "hip_yaw_r_joint",
    "r_knee": "knee_pitch_r_joint",
    "r_ankle_pitch": "ankle_pitch_r_joint",
    "r_ankle_roll": "ankle_roll_r_joint",
}


@dataclass
class CheckResult:
    check_id: str
    status: str
    summary: str
    details: list[str] = field(default_factory=list)


def inspect_motor_status_interface_text(text: str) -> dict[str, Any]:
    status_type = _extract_declared_field_type(text, "status")
    name_type = _extract_declared_field_type(text, "name")
    current_type = _extract_declared_field_type(text, "current")
    temperature_type = _extract_declared_field_type(text, "temperature")
    header_type = _extract_declared_field_type(text, "header")

    return {
        "status_type": status_type,
        "status_is_array": status_type is not None and "[" in status_type,
        "header_type": header_type,
        "has_header_or_stamp": header_type is not None or bool(re.search(r"\bstamp\b", text)),
        "name_type": name_type,
        "name_is_integer_like": name_type is not None and bool(re.search(r"(u?int|byte)", name_type)),
        "current_type": current_type,
        "current_is_array": current_type is not None and "[" in current_type,
        "temperature_type": temperature_type,
        "temperature_is_array": temperature_type is not None and "[" in temperature_type,
    }


def summarize_timing_payload(payload: dict[str, Any]) -> dict[str, float]:
    """供单元测试使用：从摘要字典恢复统计量。"""
    deltas = _extract_timing_deltas(payload)
    if deltas:
        mean_dt = statistics.fmean(deltas)
        return {
            "sample_count": float(len(deltas)),
            "mean_dt": mean_dt,
            "median_dt": statistics.median(deltas),
            "min_dt": min(deltas),
            "max_dt": max(deltas),
            "avg_hz": 1.0 / mean_dt if mean_dt > 0 else 0.0,
        }

    sample_count = float(payload.get("sample_count", 0))
    mean_dt = _coerce_float(payload.get("mean_dt") or payload.get("avg_dt"))
    median_dt = _coerce_float(payload.get("median_dt"))
    min_dt = _coerce_float(payload.get("min_dt"))
    max_dt = _coerce_float(payload.get("max_dt"))
    avg_hz = _coerce_float(payload.get("avg_hz"))

    if mean_dt is None and avg_hz and avg_hz > 0:
        mean_dt = 1.0 / avg_hz
    if avg_hz is None and mean_dt and mean_dt > 0:
        avg_hz = 1.0 / mean_dt
    if median_dt is None:
        median_dt = mean_dt
    if min_dt is None:
        min_dt = median_dt
    if max_dt is None:
        max_dt = median_dt

    if mean_dt is None or median_dt is None or min_dt is None or max_dt is None or avg_hz is None:
        raise ValueError("timing payload has no deltas or summary keys")

    return {
        "sample_count": sample_count,
        "mean_dt": mean_dt,
        "median_dt": median_dt,
        "min_dt": min_dt,
        "max_dt": max_dt,
        "avg_hz": avg_hz,
    }


def _run_ros2(args: list[str], timeout: float | None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return -1, "", "executable not found"
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (getattr(e, "output", None) or "")
        return -2, out, (e.stderr or str(e))


def try_ros2_interface_show() -> tuple[str | None, str]:
    code, out, err = _run_ros2(
        ["ros2", "interface", "show", "bodyctrl_msgs/msg/MotorStatusMsg"],
        15.0,
    )
    if code != 0:
        return None, (err or out or f"exit {code}").strip()
    return out, ""


def try_ros2_topic_echo_once(topic: str) -> tuple[str | None, str]:
    code, out, err = _run_ros2(["ros2", "topic", "echo", topic, "--once"], 20.0)
    if code != 0:
        return None, (err or out or f"exit {code}").strip()
    return out, ""


def try_ros2_topic_echo_multi(topic: str, timeout_sec: float) -> tuple[str | None, str]:
    code, out, err = _run_ros2(["ros2", "topic", "echo", topic], timeout_sec)
    if not (out or "").strip():
        return None, err or f"exit {code}"
    return out, err or ""


def split_echo_documents(text: str) -> list[str]:
    parts = re.split(r"^---\s*$", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def extract_stamp_sec(block: str) -> float | None:
    m = re.search(
        r"stamp:\s*\n\s*sec:\s*(\d+)\s*\n\s*nanosec:\s*(\d+)",
        block,
    )
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2)) * 1e-9


def inspect_leg_status_sample_yaml(text: str, lab_motor_names: list[str]) -> dict[str, Any]:
    status_items: list[dict[str, Any]] = []

    def collect_from_doc(doc: Any) -> None:
        if not isinstance(doc, dict):
            return
        st = doc.get("status")
        if isinstance(st, list):
            for item in st:
                if isinstance(item, dict):
                    status_items.append(item)

    for part in split_echo_documents(text):
        try:
            collect_from_doc(yaml.safe_load(part))
        except yaml.YAMLError:
            continue
    if not status_items and text.strip():
        try:
            collect_from_doc(yaml.safe_load(text))
        except yaml.YAMLError:
            pass

    mapped = []
    unknown_names = []
    non_scalar_temperature = False
    non_numeric_current = False
    for item in status_items:
        deploy_name = _normalize_sample_name(item.get("name"))
        if deploy_name is None or deploy_name not in DEPLOY_TO_LAB_NAME:
            unknown_names.append(item.get("name"))
            continue
        temperature = item.get("temperature")
        current = item.get("current")
        if isinstance(temperature, (list, dict)):
            non_scalar_temperature = True
        if not isinstance(current, (int, float)):
            non_numeric_current = True
        lab_name = DEPLOY_TO_LAB_NAME[deploy_name]
        mapped.append(
            {
                "deploy_name": deploy_name,
                "lab_name": lab_name,
                "lab_index": lab_motor_names.index(lab_name),
            }
        )

    mapped.sort(key=lambda x: x["lab_index"])
    unique_idx = sorted({m["lab_index"] for m in mapped})
    return {
        "raw_status_count": len(status_items),
        "leg_entry_count": len(mapped),
        "unique_leg_index_count": len(unique_idx),
        "mapped": mapped,
        "unknown_names": unknown_names,
        "non_scalar_temperature": non_scalar_temperature,
        "non_numeric_current": non_numeric_current,
    }


def run_on_robot(
    topic: str,
    cfg_dt: float | None,
    dt_tolerance_ratio: float,
    echo_multi_timeout: float,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    if not shutil.which("ros2"):
        results.append(
            CheckResult(
                check_id="p0.ros2_available",
                status=FAIL,
                summary="未找到 ros2 可执行文件，请在实机 source 安装/overlay 后再运行",
                details=[],
            )
        )
        results.append(
            CheckResult(
                check_id="p0.manual.current_amp_sign",
                status=PENDING,
                summary="电流安培与符号：需单关节运动对照 pos/speed（本脚本不判物理单位）",
                details=[],
            )
        )
        results.append(
            CheckResult(
                check_id="p0.manual.temperature_physical",
                status=PENDING,
                summary="工程基线已约定 temperature 为 float32、°C（见 plan.md §0、TienKung_ROS MotorStatus.msg）；实机可再对照驱动器显示或外测温",
                details=[],
            )
        )
        return results

    results.append(
        CheckResult(
            check_id="p0.ros2_available",
            status=PASS,
            summary="已检测到 ros2",
            details=[],
        )
    )

    iface_text, iface_err = try_ros2_interface_show()
    if iface_text:
        info = inspect_motor_status_interface_text(iface_text)
        ok_fields = all(
            info[k] is not None
            for k in ("status_type", "name_type", "current_type", "temperature_type")
        )
        results.append(
            CheckResult(
                check_id="p0.interface.motor_status_msg",
                status=PASS if ok_fields else FAIL,
                summary="MotorStatusMsg 接口可解析（interface show）",
                details=[f"{k}: {v}" for k, v in info.items()],
            )
        )
        results.append(
            CheckResult(
                check_id="p0.interface.temperature_scalar",
                status=FAIL if info["temperature_is_array"] else PASS,
                summary="temperature 在接口中为标量（基线假设）"
                if not info["temperature_is_array"]
                else "temperature 为数组，需调整监督/网络",
                details=[f"type={info['temperature_type']}"],
            )
        )
        results.append(
            CheckResult(
                check_id="p0.interface.current_name",
                status=PASS
                if not info["current_is_array"] and info["name_is_integer_like"]
                else FAIL,
                summary="current 为标量、name 为整型 id（与 getIndexById 假设一致）"
                if not info["current_is_array"] and info["name_is_integer_like"]
                else "current 或 name 与常见插件用法不一致",
                details=[f"current={info['current_type']}", f"name={info['name_type']}"],
            )
        )
    else:
        results.append(
            CheckResult(
                check_id="p0.interface.motor_status_msg",
                status=FAIL,
                summary="ros2 interface show 失败（是否已安装 bodyctrl_msgs 并 source？）",
                details=[iface_err[:600]],
            )
        )

    echo_text, echo_err = try_ros2_topic_echo_once(topic)
    if echo_text:
        sample_info = inspect_leg_status_sample_yaml(echo_text, T_LEG_MOTOR_NAMES)
        ok_sample = (
            sample_info["unique_leg_index_count"] == 12
            and not sample_info["non_scalar_temperature"]
            and not sample_info["non_numeric_current"]
        )
        results.append(
            CheckResult(
                check_id="p0.topic.leg_status_sample",
                status=PASS if ok_sample else FAIL,
                summary=f"单帧 {topic} 可映射满 12 路 T_leg"
                if ok_sample
                else "单帧腿电机不完整或字段异常",
                details=[
                    f"status条数={sample_info['raw_status_count']}",
                    f"可映射条数={sample_info['leg_entry_count']}",
                    f"unknown_name样本={sample_info['unknown_names'][:8]}",
                ],
            )
        )
    else:
        results.append(
            CheckResult(
                check_id="p0.topic.leg_status_sample",
                status=FAIL,
                summary=f"ros2 topic echo {topic} --once 失败（话题是否存在、是否有数据？）",
                details=[echo_err[:600]],
            )
        )

    multi, multi_err = try_ros2_topic_echo_multi(topic, echo_multi_timeout)
    timing_done = False
    if multi:
        docs = split_echo_documents(multi)
        stamps = [extract_stamp_sec(d) for d in docs]
        stamps = [s for s in stamps if s is not None]
        if len(stamps) >= 2:
            deltas = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
            median_dt = statistics.median(deltas)
            mean_dt = statistics.fmean(deltas)
            lines = [
                f"样本间隔数={len(deltas)}",
                f"median_dt={median_dt:.6f}s",
                f"mean_dt={mean_dt:.6f}s",
                f"min={min(deltas):.6f}s max={max(deltas):.6f}s",
            ]
            if cfg_dt is not None and cfg_dt > 0:
                rel = abs(median_dt - cfg_dt) / cfg_dt
                ok = rel <= dt_tolerance_ratio
                lines.append(f"对比 cfg_dt={cfg_dt}: 相对偏差 {rel:.1%} (容差 {dt_tolerance_ratio:.0%})")
                results.append(
                    CheckResult(
                        check_id="p0.topic.stamp_interval_vs_dt",
                        status=PASS if ok else FAIL,
                        summary="消息 stamp 间隔与给定 --dt 在容差内"
                        if ok
                        else "消息间隔与 --dt 偏差过大（请确认控制周期与话题是否同源）",
                        details=lines,
                    )
                )
            else:
                lines.append("未传 --dt，仅记录观测间隔；与 tg22_config 对比请自行核对")
                results.append(
                    CheckResult(
                        check_id="p0.topic.stamp_interval",
                        status=PASS,
                        summary="已观测 /leg/status 相邻消息 header 间隔",
                        details=lines,
                    )
                )
            timing_done = True
    if not timing_done:
        results.append(
            CheckResult(
                check_id="p0.topic.stamp_interval",
                status=PENDING,
                summary="未能从连续 echo 得到至少两帧有效 header（可增大 --echo-multi-timeout 或检查话题）",
                details=[multi_err[:300] if multi_err else ""],
            )
        )

    results.append(
        CheckResult(
            check_id="p0.manual.current_amp_sign",
            status=PENDING,
            summary="电流安培与符号：需单关节运动对照 pos/speed（本脚本不判物理单位）",
            details=[],
        )
    )
    results.append(
        CheckResult(
            check_id="p0.manual.temperature_physical",
            status=PENDING,
            summary="温度 °C / 滤波：需驱动器或外测温对照（本脚本仅看接口类型）",
            details=[],
        )
    )

    return results


def print_report(results: list[CheckResult]) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print("TienKung Thermal — P0 实机确认")
    print("=" * 40)
    print("统计: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print()
    for r in results:
        print(f"[{r.status}] {r.check_id}")
        print(f"  {r.summary}")
        for d in r.details:
            if d:
                print(f"  - {d}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0：实机 ROS 接口与 /leg/status 数据确认（不读代码仓库）")
    parser.add_argument(
        "--topic",
        default="/leg/status",
        help="腿部状态话题，默认 /leg/status",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="与 Deploy tg22_config.yaml 中 dt 一致时传入，用于与消息 stamp 间隔对比；省略则只打印观测间隔",
    )
    parser.add_argument("--dt-tolerance-ratio", type=float, default=0.2)
    parser.add_argument("--echo-multi-timeout", type=float, default=8.0, help="连续 echo 采集时长（秒）")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results = run_on_robot(
        topic=args.topic,
        cfg_dt=args.dt,
        dt_tolerance_ratio=args.dt_tolerance_ratio,
        echo_multi_timeout=args.echo_multi_timeout,
    )
    if args.json:
        print(json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2))
    else:
        print_report(results)

    if any(r.status == FAIL for r in results):
        return 1
    return 0


def _extract_declared_field_type(text: str, field_name: str) -> str | None:
    pattern = re.compile(rf"^(.+?)\s+{re.escape(field_name)}$")
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def _extract_timing_deltas(payload: dict[str, Any]) -> list[float]:
    for key in ("stamp_deltas_sec", "intervals_sec", "deltas_sec", "dts", "dt_samples"):
        values = payload.get(key)
        if isinstance(values, list):
            floats = [_coerce_float(value) for value in values]
            return [value for value in floats if value is not None]
    return []


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _normalize_sample_name(value: Any) -> str | None:
    if isinstance(value, int):
        return DEPLOY_LEG_IDS.get(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return DEPLOY_LEG_IDS.get(int(stripped))
        return stripped
    return None


if __name__ == "__main__":
    sys.exit(main())
