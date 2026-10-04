#!/usr/bin/env python
"""「快封句」验收：上游也没人说话了 → 不白等那 3s（但绝不在人还在说时抢跑）。

## 为什么要有它（以及它跟被回滚的那版差在哪）

服务端在人停止说话后 **8s 内一条事件都不发**（既无 `response.text.done` 也无
`response.done`）→ 没有语义完成信号可用，只能靠定时器；而累计译文在「说完前 0.74~0.89s」
就不再增长 → 那 3s 全是白等。所以给静默兜底加一条**快路径**：文字静默 ≥1.1s 且
**上游也没有人在说话** ≥0.5s → 立刻封句。

⚠️ 「上游没有人在说话」这个信号**必须用「电平」**，且必须**相对噪声底**：

1. 不能用「距上次上送音频的间隔」—— 麦克风腿没有闸门（`_SilenceGate` 只挂在环回腿上、
   阈值默认 30s），人说话时静音块照样每 ~0.1s 上送一次 → 那个间隔恒为 ~0.1s，
   快路径永远不触发（#49 的阻断项，见 §「代理」那条用例）。
2. 也不能用固定峰值门限（`peak >= SILENCE_PEAK`，220 ≈ -43.5dBFS 峰值）—— 那个门限是给
   30s 长静音闸门定的，低到把**房间噪声**也算成「有人在说」：实测噪声底 ≥ -50dBFS 时，
   静音块的峰值中位数就有 363 > 220 → `user_quiet_s()` 被噪声持续刷新、恒 ~0.1s，
   快路径在真麦克风上**静默失效**（同型故障，换了个死法）。

**为什么以前测不出来**：喂给采集泵的「静音」是**全零**（数字静音，峰值恒为 0），
而真实麦克风在没人说话时送的是**带噪声底的房间声**。所以本文件里的静音一律用
**噪声**（默认 -50dBFS，接近本机实测用户那台 USB 麦的底噪 -48.9dBFS），并让判据先
暖一段噪声底再喂「说话」——这两件事正是把上面第 2 条钉死的钉子。

判据本体在 `vlt/voice_activity.py`，单元用例在 `tests/test_voice_activity.py`。
"""
from __future__ import annotations

import asyncio
import math
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlt.engine import SILENCE_PEAK                                # noqa: E402
from vlt.session.base import (DEFAULT_FAST_FINAL_SILENCE_S,       # noqa: E402
                              DEFAULT_FAST_FINAL_USER_QUIET_S,
                              DEFAULT_FINAL_SILENCE_S, SessionConfig, should_finalize)
from vlt.session.qwen38 import QwenLiveTranslateSession           # noqa: E402

CHUNK_SAMPLES = 1600                       # 100ms @16k 单声道
QUIET_CHUNKS = 20                          # 开场 2.0s 房间噪声（暖噪声底 = voice_activity 的预热）
SPEECH_CHUNKS = 10                         # 1.0s 说话
SILENCE_CHUNKS = 30                        # 之后 3.0s 房间噪声
NOISE_DB = -50.0                           # 房间噪声底（本机实测用户那台麦 ≈ -48.9dBFS）
SPEECH_DB = -25.0                          # 说话电平（比底噪高 25dB）
_RNG = np.random.default_rng(20261004)
_TONE = b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / 16000)))
                 for i in range(CHUNK_SAMPLES))


def _noise(floor_db: float = NOISE_DB) -> bytes:
    sigma = 32768.0 * (10.0 ** (floor_db / 20.0))
    return _RNG.normal(0, sigma, CHUNK_SAMPLES).astype("<i2").tobytes()


def _speech(level_db: float = SPEECH_DB) -> bytes:
    t = np.arange(CHUNK_SAMPLES) / 16000.0
    amp = 32768.0 * (10.0 ** (level_db / 20.0)) * math.sqrt(2.0)
    return (np.sin(2 * math.pi * 220.0 * t) * amp).astype("<i2").tobytes()


def _peak_of(pcm: bytes) -> int:
    return int(np.abs(np.frombuffer(pcm, dtype="<i2")).max())


def _fmt(sec: float | None) -> str:
    return "None（没收到过信号）" if sec is None else f"{sec:.2f}s"


def test_pure_table() -> bool:
    ok = True
    cases = [
        # (文字静默, 上游静默, 期望, 说明)
        (3.10, 0.30, True,  "慢路径：人还在说，但文字静默过了 3s → 照旧封"),
        (2.30, 0.20, False, "**不抢跑**：说话中间那个 2.3s 大间隔（上游还在说）"),
        (1.20, 1.00, True,  "快路径：上游静了 1s + 文字静默 1.2s → 封（省 ~1.8s）"),
        (1.20, 0.20, False, "**不抢跑**：文字静默够了但人还在说"),
        (0.90, 5.00, False, "上游静很久了，但文字才静 0.9s（服务端可能还在追）"),
        (1.10, 0.50, True,  "正好卡在阈值上（≥ 即封）"),
        (1.09, 0.50, False, "差一点就不封"),
    ]
    for text_q, user_q, want, why in cases:
        got = should_finalize(text_quiet_s=text_q, user_quiet_s=user_q)
        cond = got is want
        print(f"  文字静默={text_q:.2f}s 上游静默={user_q:.2f}s → {got}"
              f"（期望 {want}）{'OK' if cond else '✗'}  {why}")
        ok &= cond
    return ok


def test_pure_edges() -> bool:
    ok = True
    # 从没收到过「有人说话」的信号（没声音 / 信号没接上）→ **保守走慢路径**，绝不早封：
    # 抢跑会把半句当最终版；而「从没说过话」时本来也没文本可封。
    got = should_finalize(text_quiet_s=1.2, user_quiet_s=None)
    cond = got is False
    print(f"  从没收到「有人说话」信号 + 文字静默 1.2s → {got}（期望 False：保守走慢路径）"
          f"  {'OK' if cond else '✗'}")
    ok &= cond
    # 关掉快路径（fast_final_silence_s=None）→ 退回纯 3.0s 行为
    got = should_finalize(text_quiet_s=1.2, user_quiet_s=9.9, fast_silence_s=None)
    cond = got is False
    print(f"  关掉快路径 + 文字静默 1.2s → {got}（期望 False）  {'OK' if cond else '✗'}")
    ok &= cond
    got = should_finalize(text_quiet_s=3.1, user_quiet_s=9.9, fast_silence_s=None)
    cond = got is True
    print(f"  关掉快路径 + 文字静默 3.1s → {got}（期望 True，慢路径还在）  {'OK' if cond else '✗'}")
    ok &= cond
    # 阈值可调
    got = should_finalize(text_quiet_s=0.6, user_quiet_s=1.0, fast_silence_s=0.55)
    cond = got is True
    print(f"  自定义快阈值 0.55s + 文字静默 0.6s → {got}（期望 True）  {'OK' if cond else '✗'}")
    ok &= cond
    # 默认值一致性（config 默认 1.1 / 0.5，别被悄悄改大）
    cond = (DEFAULT_FAST_FINAL_SILENCE_S == 1.1 and DEFAULT_FAST_FINAL_USER_QUIET_S == 0.5
            and DEFAULT_FINAL_SILENCE_S == 3.0)
    print(f"  默认阈值：快={DEFAULT_FAST_FINAL_SILENCE_S}s/上游={DEFAULT_FAST_FINAL_USER_QUIET_S}s"
          f"，慢={DEFAULT_FINAL_SILENCE_S}s  {'OK' if cond else '✗'}")
    ok &= cond
    return ok


class _FakeWS:
    async def send(self, _data) -> None:
        return None


class _FakeCfg:
    session_base = {"silence_gate_enabled": True, "silence_gate_after_s": 30.0,
                    "silence_gate_preroll_s": 1.0}


class _FakeEngine:
    """喂给真实 `_SessionProxy`：只提供它要读的字段。"""

    _cfg = _FakeCfg()
    _session = None
    _audio_in_chunks = 0
    _silent_chunks = 0
    _last_loud_ts = 0.0
    _silence_gate_cfg = None


def test_proxy_reports_voice_by_level() -> bool:
    """★ 核心回归：信号必须来自**相对噪声底的电平**。

    真实 `_SessionProxy` + **真实房间噪声**（不是全零）：
      - 预热期（前 2s）不算「听到人声」→ `user_quiet_s()` 仍是 `None`（保守）；
      - 喂一串「说话」→ `user_quiet_s()` 很小（信号到了会话）；
      - 接着喂**噪声**（真实麦克风腿就是这样：没人说话也每 100ms 送一块）→
        `user_quiet_s()` 必须**继续变大**。若哪天有人把判据改回固定峰值门限
        （`peak >= SILENCE_PEAK`），这些噪声块会被当成「有人在说」，
        这个值就会恒 ~0.0s（用例立刻红）。
    """
    from vlt.engine import _SessionProxy

    async def run() -> bool:
        ok = True
        sess = QwenLiveTranslateSession(SessionConfig(api_key="x"))
        sess._ws = _FakeWS()
        eng = _FakeEngine()
        eng._session = sess
        proxy = _SessionProxy(eng)

        # ① 从未说过话 → None（按「已静」看待，判据那边走慢路径）
        cond = sess.user_quiet_s() is None
        print(f"  没说过话时 user_quiet_s() = {sess.user_quiet_s()}（期望 None）  {'OK' if cond else '✗'}")
        ok &= cond

        # ② 预热期：20 块房间噪声（2.0s）。噪声底还没学出来 → 不许上报「有人说话」
        for _ in range(QUIET_CHUNKS):
            await proxy.send_audio(_noise())
        cond = sess.user_quiet_s() is None
        print(f"  预热期喂 {QUIET_CHUNKS} 块房间噪声（{QUIET_CHUNKS / 10:.1f}s）→ "
              f"user_quiet_s()={_fmt(sess.user_quiet_s())}（期望 None：占位不算听到人声）"
              f"  {'OK' if cond else '✗'}")
        ok &= cond

        # ③ 一层「说话」——底噪 +25dB
        for _ in range(5):
            await proxy.send_audio(_speech())
        q = sess.user_quiet_s()
        cond = q is not None and q < 0.2
        print(f"  喂 5 块说话（底噪 +25dB）→ user_quiet_s()={_fmt(q)}（期望 <0.2s）  {'OK' if cond else '✗'}")
        ok &= cond

        # ④ 接着连喂**噪声**块（真实麦克风腿就是这样：不说话也每 100ms 送一块）
        for _ in range(6):
            await proxy.send_audio(_noise())
            await asyncio.sleep(0.1)
        q = sess.user_quiet_s()
        cond = q is not None and q >= 0.4
        print(f"  之后 0.6s 全是房间噪声（仍在往上送）→ user_quiet_s()={_fmt(q)}"
              f"（期望 ≥0.4s：噪声就该算「静」）  {'OK' if cond else '✗'}")
        ok &= cond

        # ⑤ 钉住「为什么不能用固定峰值门限」：同一批噪声块在 220 门限下全是「有人在说」
        noise_peaks = [_peak_of(_noise()) for _ in range(20)]
        loud = sum(1 for p in noise_peaks if p >= SILENCE_PEAK)
        cond = loud >= 15
        print(f"  同一批房间噪声块：峰值中位 {sorted(noise_peaks)[10]}，"
              f"其中 {loud}/20 块 ≥ SILENCE_PEAK({SILENCE_PEAK}) → 固定峰值门限会把噪声")
        print(f"    全当成「有人在说」→ user_quiet_s() 恒 ~0、快路径静默失效"
              f"（这就是必须相对噪声底的原因）  {'OK' if cond else '✗'}")
        ok &= cond

        # ⑥ 再喂一块说话 → 重新计时
        await proxy.send_audio(_speech())
        q = sess.user_quiet_s()
        cond = q is not None and q < 0.2
        print(f"  再喂一块说话的 → user_quiet_s()={_fmt(q)}（期望 <0.2s）  {'OK' if cond else '✗'}")
        ok &= cond
        return ok

    return asyncio.run(run())


class _FakeSource:
    """按实时节奏吐块（100ms 一块）：先暖噪声底，再说话，之后一直是房间噪声。"""

    rate = 16000
    channels = 1

    def __init__(self) -> None:
        self.seq = ([_noise() for _ in range(QUIET_CHUNKS)]
                    + [_speech() for _ in range(SPEECH_CHUNKS)]
                    + [_noise() for _ in range(SILENCE_CHUNKS)])
        self.i = 0
        self.speech_end_at: float | None = None

    async def read(self, timeout: float = 1.0):
        await asyncio.sleep(0.1)
        if self.i >= len(self.seq):
            return _noise()                 # 采集流不会停：一直有房间噪声（真实麦克风就是这样）
        chunk = self.seq[self.i]
        self.i += 1
        if self.i == QUIET_CHUNKS + SPEECH_CHUNKS:      # 刚送完最后一块「说话」
            self.speech_end_at = time.perf_counter()
        return chunk

    def close(self) -> None:
        return None


E2E_WATCHDOG_S = 7.0


def _run_e2e(fast_silence_s: float | None) -> tuple[float, float, float | None] | None:
    """真链路复刻：`_pump_capture` 喂「2s 噪声 + 1s 说话 + 噪声」+ 真实 `tick()`。

    返回 (说完 → 封句的秒数, 封句瞬间文字静默, 封句瞬间上游静默)。
    """
    from vlt.engine import _SessionProxy, _pump_capture

    async def run() -> tuple[float, float, float | None] | None:
        sess = QwenLiveTranslateSession(
            SessionConfig(api_key="x", fast_final_silence_s=fast_silence_s))
        sess._ws = _FakeWS()
        eng = _FakeEngine()
        eng._session = sess
        proxy = _SessionProxy(eng)

        finals: list[tuple[float, float, float | None]] = []

        def on_text(d) -> None:
            if d.is_final:
                now = time.perf_counter()
                finals.append((now, now - sess._last_text_at, sess.user_quiet_s(now)))

        sess.on_text = on_text
        # 模拟「服务端已经把译文吐完了」：文本已累计，但**时间戳留到说话结束再打**
        # （真机时间线：最后一条增量在说完前 0.74~0.89s 到齐；若在这里就打时间戳，
        #   慢路径会先于快路径触发，测的就不是快路径了）。
        sess._buf = ["你好，世界。"]

        source = _FakeSource()
        stop = threading.Event()

        async def ticker() -> None:
            stamped = False
            while not stop.is_set():
                if source.speech_end_at is not None and not stamped:
                    # 真机日志：最后一片译文在说完**前 0.74~0.89s** 就到齐 → 这里按 0.8s 复刻
                    sess._last_text_at = source.speech_end_at - 0.8
                    stamped = True
                sess.tick()
                await asyncio.sleep(0.1)

        async def watchdog() -> None:
            await asyncio.sleep(E2E_WATCHDOG_S)
            stop.set()

        tk = asyncio.create_task(ticker())
        wd = asyncio.create_task(watchdog())
        await _pump_capture(proxy, None, 0.0, stop, "mic", lambda: source, None)
        stop.set()
        wd.cancel()
        await tk
        if source.speech_end_at is None or not finals:
            return None
        at, tq, uq = finals[0]
        return (at - source.speech_end_at, tq, uq)

    return asyncio.run(run())


def test_end_to_end_fast_path() -> bool:
    """★ 端到端：走真实采集泵（房间噪声持续上送）时，快路径**真的**要生效。"""
    ok = True
    got = _run_e2e(DEFAULT_FAST_FINAL_SILENCE_S)
    if got is None:
        print("  ✗ 没拿到完整时间线（无法判定）")
        return False
    fast, tq, uq = got
    cond = fast < 2.0
    print(f"  默认开快路径：说完 → 封句 +{fast:.2f}s（封句瞬间：文字静默 {tq:.2f}s、"
          f"上游静默 {_fmt(uq)}）  期望 <2.0s  {'OK' if cond else '✗'}")
    ok &= cond

    got = _run_e2e(None)
    if got is None:
        print("  ✗ 关掉快路径那次没拿到时间线")
        return False
    slow, tq2, _ = got
    # 慢路径 = 文字静默满 3.0s；而最后一片译文在说完前 0.8s 就到了 → 实测 ~+2.2s
    cond = slow >= 2.0
    print(f"  关掉快路径：说完 → 封句 +{slow:.2f}s（文字静默 {tq2:.2f}s）  期望 ≥2.0s  "
          f"{'OK' if cond else '✗'}")
    ok &= cond
    cond = slow - fast >= 1.0
    print(f"  快路径的收益 = {slow - fast:.2f}s（期望 ≥1.0s；真机日志里省 ~1.8s）  "
          f"{'OK' if cond else '✗'}")
    ok &= cond
    return ok


def test_config_parsing() -> bool:
    """配置口径：非法值**留痕 + 回落默认值**（不许静默关掉）；`null` 才是明确关掉。"""
    ok = True
    import contextlib
    import io

    from vlt.config import _float_or_default, _opt_float
    from vlt.engine import voice_margin_settings
    from vlt.voice_activity import DEFAULT_VOICE_MARGIN_DB

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = _opt_float("1,1", DEFAULT_FAST_FINAL_SILENCE_S, key="session.fast_final_silence_s")
    log = buf.getvalue()
    cond = got == DEFAULT_FAST_FINAL_SILENCE_S and "⚠️" in log and "fast_final_silence_s" in log
    print(f"  fast_final_silence_s='1,1' → {got}（期望回落 {DEFAULT_FAST_FINAL_SILENCE_S}）+ 留痕  "
          f"{'OK' if cond else '✗'}")
    ok &= cond

    cond = _opt_float(None, DEFAULT_FAST_FINAL_SILENCE_S, key="k") is None
    cond &= _opt_float("", DEFAULT_FAST_FINAL_SILENCE_S, key="k") is None
    print(f"  null / 空串 → None（明确关掉快路径）  {'OK' if cond else '✗'}")
    ok &= cond

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = _float_or_default("abc", DEFAULT_FAST_FINAL_USER_QUIET_S,
                                key="session.fast_final_user_quiet_s")
    cond = got == DEFAULT_FAST_FINAL_USER_QUIET_S and "⚠️" in buf.getvalue()
    print(f"  fast_final_user_quiet_s='abc' → {got}（期望回落默认值）+ 留痕  {'OK' if cond else '✗'}")
    ok &= cond

    # 判据余量：缺省/null 用默认；越界或非数字留痕 + 回落默认（不许静默带病运行）
    cond = voice_margin_settings({}) == DEFAULT_VOICE_MARGIN_DB
    cond &= voice_margin_settings({"fast_final_voice_margin_db": None}) == DEFAULT_VOICE_MARGIN_DB
    cond &= voice_margin_settings({"fast_final_voice_margin_db": 6}) == 6.0
    print(f"  fast_final_voice_margin_db：缺省/null → 默认 {DEFAULT_VOICE_MARGIN_DB}，6 → 6.0  "
          f"{'OK' if cond else '✗'}")
    ok &= cond
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bad = voice_margin_settings({"fast_final_voice_margin_db": 99})
    cond = bad == DEFAULT_VOICE_MARGIN_DB and "⚠️" in buf.getvalue()
    print(f"  fast_final_voice_margin_db=99 → {bad}（越界回落默认）+ 留痕  {'OK' if cond else '✗'}")
    ok &= cond
    return ok


if __name__ == "__main__":
    print("test_fast_finalize:")
    results = [
        ("纯函数判据表", test_pure_table()),
        ("纯函数边界", test_pure_edges()),
        ("代理按电平上报（真实 _SessionProxy + 真实房间噪声）", test_proxy_reports_voice_by_level()),
        ("端到端（真实采集泵 + tick）", test_end_to_end_fast_path()),
        ("配置解析", test_config_parsing()),
    ]
    bad = [name for name, ok in results if not ok]
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    if bad:
        print(f"快封句用例失败（见上）：{', '.join(bad)}")
        sys.exit(1)
    print("ALL PASSED")
