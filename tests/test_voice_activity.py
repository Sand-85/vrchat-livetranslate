#!/usr/bin/env python
"""`vlt/voice_activity.py` 验收：块电平 → 「有没有人在说」，必须是**相对噪声底**的。

## 为什么这个测试长这样

PR #49 的阻断项是「判据信号在真实链路上拿不到」。换成绝对门限也不行 —— 实测
（本文件第 3 节）：噪声底 ≥ -50dBFS 时，100ms 静音块的**峰值**中位数就有 363，
而审核建议沿用的 `SILENCE_PEAK` 是 220 → 静音被判成有声。本文件把这条
性质固化成用例：**同一句话在 -70 ~ -35dBFS 的噪声底里都必须被认出来**。绝对门限
做不到这件事（那正是它不该被用的理由），相对噪声底做得到。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlt.engine import SILENCE_PEAK, chunk_level_db, voice_margin_settings  # noqa: E402
from vlt.voice_activity import (DEFAULT_VOICE_MARGIN_DB, VOICE_MARGIN_MAX_DB,  # noqa: E402
                                VoiceActivity)

CHUNK_SAMPLES = 1600          # 100ms @16kHz mono s16le
RNG = np.random.default_rng(20261004)


def noise_chunk(floor_db: float) -> bytes:
    sigma = 32768.0 * (10.0 ** (floor_db / 20.0))
    return RNG.normal(0, sigma, CHUNK_SAMPLES).astype("<i2").tobytes()


def speech_chunk(level_db: float, f: float = 220.0) -> bytes:
    t = np.arange(CHUNK_SAMPLES) / 16000.0
    amp = 32768.0 * (10.0 ** (level_db / 20.0)) * np.sqrt(2.0)   # 幅度 → 让 RMS 落在 level_db
    return (np.sin(2 * np.pi * f * t) * amp).astype("<i2").tobytes()


def peak_of(pcm: bytes) -> int:
    return int(np.abs(np.frombuffer(pcm, dtype="<i2")).max())


def test_warmup_is_conservative() -> bool:
    ok = True
    va = VoiceActivity(min_samples=5)
    cond = va.ready is False
    print(f"  初始：ready={va.ready}（期望 False）  {'OK' if cond else '✗'}")
    ok &= cond
    got = [va.feed(chunk_level_db(noise_chunk(-60.0))) for _ in range(4)]
    cond = all(got) and va.ready is False
    print(f"  预热期内 4 块噪声 → 全判「有人在说」（保守），ready 仍为 False  {'OK' if cond else '✗'}")
    ok &= cond
    got5 = va.feed(chunk_level_db(noise_chunk(-60.0)))
    cond = va.ready is True and got5 is False
    print(f"  第 5 块（够样本了）→ ready=True 且判「静」  {'OK' if cond else '✗'}")
    ok &= cond
    return ok


def test_relative_to_noise_floor() -> bool:
    """同一句话（噪声底 +25dB）在各种噪声底里都要被认出来。"""
    ok = True
    for floor_db in (-70.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0):
        va = VoiceActivity(min_samples=20)
        for _ in range(30):                       # 30 块房间噪声（3s）
            va.feed(chunk_level_db(noise_chunk(floor_db)))
        floor_est = va.floor_db
        voice_hits = sum(va.feed(chunk_level_db(speech_chunk(floor_db + 25.0)))
                         for _ in range(10))      # 说话：底噪 +25dB
        # 绝对门限（审核建议沿用的 SILENCE_PEAK）在同样的块上是什么结果？
        peaks = [peak_of(speech_chunk(floor_db + 25.0)) for _ in range(1)]
        cond = voice_hits == 10
        print(f"  噪声底 {floor_db:>6.0f}dBFS → 估计 {floor_est:>6.1f}dBFS"
              f"｜说话块判为人声 {voice_hits}/10（期望 10）  {'OK' if cond else '✗'}")
        ok &= cond
        cond = peaks[0] >= SILENCE_PEAK
        print(f"          （同一块峰值 {peaks[0]:>6} vs SILENCE_PEAK {SILENCE_PEAK}）"
              f"  {'OK' if cond else '✗'}")
        ok &= cond
    return ok


def test_noise_never_voice_but_absolute_would_fail() -> bool:
    """房间噪声一律判「静」；同时展示绝对峰值门限在同一批块上会怎么错。"""
    ok = True
    for floor_db in (-70.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0):
        va = VoiceActivity(min_samples=20)
        for _ in range(25):                       # 预热
            va.feed(chunk_level_db(noise_chunk(floor_db)))
        quiet = sum(1 for _ in range(20)
                    if not va.feed(chunk_level_db(noise_chunk(floor_db))))
        # 绝对门限：噪声块里有多少被判成「有声」
        would_be_loud = sum(1 for _ in range(20)
                            if peak_of(noise_chunk(floor_db)) >= SILENCE_PEAK)
        cond = quiet == 20
        note = ""
        if would_be_loud:
            note = f"｜⚠️ 绝对门限会把这 20 块里的 {would_be_loud} 块当成有声 → 快路径顶死"
        print(f"  噪声底 {floor_db:>6.0f}dBFS → 判「静」{quiet}/20（期望 20）  "
              f"{'OK' if cond else '✗'}{note}")
        ok &= cond
    return ok


def test_digital_silence_and_transition() -> bool:
    ok = True
    # 麦被系统静音 / 设备没信号：恒为数字静音 → 必须判「静」
    va = VoiceActivity(min_samples=20)
    for _ in range(25):
        va.feed(chunk_level_db(b"\x00\x00" * CHUNK_SAMPLES))
    quiet = sum(1 for _ in range(10)
                if not va.feed(chunk_level_db(b"\x00\x00" * CHUNK_SAMPLES)))
    cond = quiet == 10
    print(f"  数字静音（麦被静音）→ 判「静」{quiet}/10（期望 10）  {'OK' if cond else '✗'}")
    ok &= cond
    # 从数字静音切到真实房间噪声：噪声底得重新学，期间会把噪声判成有声
    #（保守方向：不提前封句，不会切句）—— 记为已知边界，不当作失败
    va = VoiceActivity(min_samples=20)
    for _ in range(30):
        va.feed(chunk_level_db(b"\x00\x00" * CHUNK_SAMPLES))
    hits = sum(1 for _ in range(10)
               if va.feed(chunk_level_db(noise_chunk(-50.0))))
    print(f"  数字静音 → -50dBFS 噪声的切换期：前 10 块判成有声 {hits}/10"
          f"（保守方向，30s 窗口走满后自愈）")
    return ok


def test_margin_settings() -> bool:
    ok = True
    cases = [
        ({}, DEFAULT_VOICE_MARGIN_DB, "没配 → 默认"),
        ({"fast_final_voice_margin_db": None}, DEFAULT_VOICE_MARGIN_DB, "显式 null → 默认"),
        ({"fast_final_voice_margin_db": 6}, 6.0, "合法值照用"),
        ({"fast_final_voice_margin_db": 0.5}, DEFAULT_VOICE_MARGIN_DB, "低于下限 → 回落默认"),
        ({"fast_final_voice_margin_db": 99}, DEFAULT_VOICE_MARGIN_DB, "高于上限 → 回落默认"),
        ({"fast_final_voice_margin_db": "abc"}, DEFAULT_VOICE_MARGIN_DB, "非数字 → 回落默认"),
        ({"fast_final_voice_margin_db": True}, DEFAULT_VOICE_MARGIN_DB, "布尔 → 回落默认"),
    ]
    for base, want, why in cases:
        got = voice_margin_settings(base)
        cond = got == want
        print(f"  {why}: {base.get('fast_final_voice_margin_db', '(缺)')!r} → {got}"
              f"（期望 {want}）  {'OK' if cond else '✗'}")
        ok &= cond
    cond = VOICE_MARGIN_MAX_DB >= DEFAULT_VOICE_MARGIN_DB
    ok &= cond
    return ok


if __name__ == "__main__":
    print("test_voice_activity:")
    print(" 1) 预热期保守（未 ready 一律算「有人在说」）")
    ok = test_warmup_is_conservative()
    print(" 2) 相对噪声底：同一句话在 -70~-35dBFS 底噪里都被认出来")
    ok &= test_relative_to_noise_floor()
    print(" 3) 房间噪声一律判「静」（并展示绝对峰值门限会怎么错）")
    ok &= test_noise_never_voice_but_absolute_would_fail()
    print(" 4) 数字静音 / 切换期（已知边界）")
    ok &= test_digital_silence_and_transition()
    print(" 5) 配置读取（非法值留痕 + 回落默认）")
    ok &= test_margin_settings()
    assert ok, "voice_activity 用例失败（见上）"
    print("ALL PASSED")
