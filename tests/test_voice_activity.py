#!/usr/bin/env python
"""`vlt/voice_activity.py` 验收：块电平 → 「有没有人在说」，必须是**相对噪声底**的。

## 为什么这个测试长这样

快封句的信号换过两次：#49 用「距上次上送音频的间隔」（真链路恒 ~0.1s，永不触发）；
#57 换成 `peak >= SILENCE_PEAK`（220 ≈ -43.5dBFS 峰值）。后者仍然不行 —— 那个门限是
给 30s 长静音闸门定的，低到把**房间噪声**也算成「有人在说」。本文件把这条性质固化成
用例：**同一句话在 -70 ~ -35dBFS 的噪声底里都必须被认出来，且房间噪声一律不许被判成
「有人在说」**；第 3 节同时打印固定峰值门限在同一批噪声块上的错判数（-50dBFS 起 20/20
全错 → `user_quiet_s()` 会被噪声持续刷新、快路径静默失效）。

绝对门限做不到这件事（那正是它不该被用的理由：底噪 + 麦克风增益在用户之间能差 20~30dB），
相对噪声底做得到。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlt.engine import SILENCE_PEAK, chunk_level_db          # noqa: E402
from vlt.voice_activity import VoiceActivity                 # noqa: E402

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
    print(f"  预热期内 4 块噪声 → 全判「有人在说」（保守占位），ready 仍为 False  "
          f"{'OK' if cond else '✗'}")
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
        cond = voice_hits == 10
        print(f"  噪声底 {floor_db:>6.0f}dBFS → 估计 {floor_est:>6.1f}dBFS"
              f"｜说话块判为人声 {voice_hits}/10（期望 10）  {'OK' if cond else '✗'}")
        ok &= cond
    return ok


def test_noise_never_voice_but_absolute_threshold_fails() -> bool:
    """房间噪声一律判「静」；同时展示固定峰值门限在同一批块上会怎么错。"""
    ok = True
    for floor_db in (-70.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0):
        va = VoiceActivity(min_samples=20)
        for _ in range(25):                       # 预热
            va.feed(chunk_level_db(noise_chunk(floor_db)))
        quiet = sum(1 for _ in range(20)
                    if not va.feed(chunk_level_db(noise_chunk(floor_db))))
        # 固定峰值门限：噪声块里有多少会被判成「有人在说」
        would_be_loud = sum(1 for _ in range(20)
                            if peak_of(noise_chunk(floor_db)) >= SILENCE_PEAK)
        cond = quiet == 20
        note = ""
        if would_be_loud:
            note = f"｜⚠️ 固定峰值门限会把 20 块里的 {would_be_loud} 块当成「有人在说」→ 快路径顶死"
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


if __name__ == "__main__":
    print("test_voice_activity:")
    print(" 1) 预热期保守（未 ready 一律算「有人在说」，但不算「听到人声」）")
    ok = test_warmup_is_conservative()
    print(" 2) 相对噪声底：同一句话在 -70~-35dBFS 底噪里都被认出来")
    ok &= test_relative_to_noise_floor()
    print(" 3) 房间噪声一律判「静」（并展示固定峰值门限会怎么错）")
    ok &= test_noise_never_voice_but_absolute_threshold_fails()
    print(" 4) 数字静音 / 切换期（已知边界）")
    ok &= test_digital_silence_and_transition()
    assert ok, "voice_activity 用例失败（见上）"
    print("ALL PASSED")
