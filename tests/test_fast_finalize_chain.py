#!/usr/bin/env python
"""链路级验收：**真实采集泵 → 真实代理 → 真实会话**，判据信号到底拿不拿得到。

这是 PR #49 阻断项（审核打回）的回归测试。原实现用「距上次 `send_audio` 的间隔」
当「麦克风静了多久」，而采集腿每块都发 → 那个量恒为 ~0.1s，快路径从未触发。
本文件不做打桩推理，只把**声卡设备**换成按真实节奏吐块的脚本源，其余全是真代码：

    _pump_capture → _send_gated(gate=None) → _SessionProxy.send_audio
      → VoiceActivity（相对噪声底）→ session.note_voice()
      → QwenLiveTranslateSession.tick() → should_finalize()

三个断言：
  A) 旧量（上送间隔）在静音期**依然**是 ~0.1s —— 证明它本来就做不到这件事；
  B) 新量（音频静默）在静音期能长过阈值，且**封句真的提前发生**（快路径生效）；
  C) 全程只有房间噪声（从没听到人声）时**不提前封句** —— 保守闸生效，不抢跑。
"""
from __future__ import annotations

import asyncio
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlt.engine import CHUNK_BYTES, _pump_capture, _SessionProxy     # noqa: E402
from vlt.session.base import SessionConfig                           # noqa: E402
from vlt.session.qwen38 import QwenLiveTranslateSession              # noqa: E402

CHUNK_S = CHUNK_BYTES / 2 / 16000.0          # 0.1s
QUIET, SPEECH = "quiet", "speech"


# ---------------------------------------------------------------- 边界替身

class FakeWS:
    """只替掉网络：会话内部逻辑全是真的。"""

    def __init__(self) -> None:
        self.sent = 0

    async def send(self, data: str) -> None:
        self.sent += 1

    async def close(self) -> None:
        pass


class StubEngine:
    """`_SessionProxy` 需要的引擎属性（真实代理类，只替掉 Engine 本体）。"""

    def __init__(self, session_base: dict | None = None) -> None:
        self._cfg = types.SimpleNamespace(session_base=session_base or {})
        self._audio_in_chunks = 0
        self._silent_chunks = 0
        self._last_loud_ts = 0.0
        self._session = None


class TimedSession(QwenLiveTranslateSession):
    """记录每次 `send_audio` 的时刻 —— 用来复现「旧判据为什么不行」。"""

    def __init__(self, cfg: SessionConfig) -> None:
        super().__init__(cfg)
        self.send_times: list[float] = []
        self.texts: list[tuple[str, bool, float]] = []

    async def send_audio(self, pcm: bytes) -> None:
        self.send_times.append(time.perf_counter())
        await super().send_audio(pcm)

    def on_text_recorder(self, delta) -> None:      # type: ignore[no-untyped-def]
        self.texts.append((delta.display, bool(delta.is_final), time.perf_counter()))

    def max_send_gap_after(self, t: float) -> float:
        """t 之后相邻两次上送的**最大间隔**（= 原 PR 那个「麦克风静默」量）。"""
        ts = [x for x in self.send_times if x >= t]
        return max((b - a for a, b in zip(ts, ts[1:])), default=0.0)


class ScriptedSource:
    """按脚本吐块的采集源：phase 序列 = [(QUIET|SPEECH, 秒), ...]，按真实节奏。"""

    rate = 16000
    channels = 1

    def __init__(self, phases: list[tuple[str, float]], noise_db: float = -50.0,
                 seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        t = np.arange(CHUNK_BYTES // 2) / 16000.0
        speech = (np.sin(2 * math.pi * 220.0 * t) * 8000).astype("<i2").tobytes()
        sigma = 32768.0 * (10.0 ** (noise_db / 20.0))          # 真实房间噪声（不是全零！）
        quiet = rng.normal(0, sigma, CHUNK_BYTES // 2).astype("<i2").tobytes()
        self._chunks: list[tuple[str, bytes]] = []
        for kind, secs in phases:
            self._chunks += [(kind, speech if kind == SPEECH else quiet)] * int(round(secs / CHUNK_S))
        self.total_s = len(self._chunks) * CHUNK_S
        self._i = 0
        self.speech_end_t: float | None = None       # 第一段「说话」结束的时刻

    async def read(self, timeout: float = 1.0) -> bytes | None:
        if self._i >= len(self._chunks):
            await asyncio.sleep(CHUNK_S)
            return None
        await asyncio.sleep(CHUNK_S)
        kind, chunk = self._chunks[self._i]
        self._i += 1
        if kind == SPEECH and (self._i >= len(self._chunks)
                               or self._chunks[self._i][0] != SPEECH):
            if self.speech_end_t is None:
                self.speech_end_t = time.perf_counter()
        return chunk

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- 场景

async def run_chain(*, phases: list[tuple[str, float]], noise_db: float,
                    fast_silence: float, fast_quiet: float,
                    inject_at: str, inject_lead: float,
                    ) -> tuple[TimedSession, ScriptedSource, dict]:
    """跑一次真实链路，返回 (会话, 源, 观测)。"""
    eng = StubEngine()
    src = ScriptedSource(phases, noise_db=noise_db)
    session = TimedSession(SessionConfig(
        api_key="test-key", final_silence_s=3.0,
        fast_final_silence_s=fast_silence, fast_final_audio_quiet_s=fast_quiet))
    session.on_text = session.on_text_recorder
    session._ws = FakeWS()
    eng._session = session
    proxy = _SessionProxy(eng)

    obs = {"injected": False, "finalize_at": None, "audio_quiet_at_final": None,
           "voiced_at_final": None, "max_audio_quiet": 0.0, "t0": time.perf_counter()}
    task = asyncio.create_task(_pump_capture(proxy, None, src.total_s, None, "mic",
                                             lambda: src, None))
    while not task.done():
        await asyncio.sleep(0.02)
        now = time.perf_counter()
        if session._last_voice_at:
            obs["max_audio_quiet"] = max(obs["max_audio_quiet"], now - session._last_voice_at)
        if not obs["injected"] and src.speech_end_t is not None:
            session._buf = ["你好世界。"]                                  # 模拟累计译文
            session._last_text_at = time.perf_counter() - inject_lead     # 说完前 lead 秒定稿
            obs["injected"] = True
        session.tick()
        if obs["finalize_at"] is None and session.texts and session.texts[-1][1]:
            obs["finalize_at"] = session.texts[-1][2]
            obs["audio_quiet_at_final"] = (
                now - session._last_voice_at) if session._last_voice_at else None
            obs["voiced_at_final"] = session._voiced_in_utterance
    await task
    return session, src, obs


async def scenario_fast_path() -> bool:
    """A + B：静音期旧量恒 ~0.1s（做不到），新量能长起来且封句真的提前。"""
    ok = True
    session, src, obs = await run_chain(
        phases=[(QUIET, 2.0), (SPEECH, 2.0), (QUIET, 4.0)], noise_db=-50.0,
        fast_silence=0.6, fast_quiet=0.6, inject_at="speech_end", inject_lead=0.8)
    t_end = src.speech_end_t
    old_gap = session.max_send_gap_after(t_end)
    cond = old_gap < 0.35
    print(f"  A) 说完后「相邻上送最大间隔」（原判据）= {old_gap:.3f}s"
          f"（阈值 0.6s）→ {'拿不到，正是阻断项' if cond else '✗ 与审核结论不符'}"
          f"  {'OK' if cond else '✗'}")
    ok &= cond
    cond = obs["max_audio_quiet"] >= 1.0
    print(f"  B1) 新判据「音频静默」峰值 = {obs['max_audio_quiet']:.3f}s（期望 ≥ 1.0s）"
          f"  {'OK' if cond else '✗'}")
    ok &= cond
    if obs["finalize_at"] is None:
        print("  B2) 没封句 ✗")
        return False
    after = obs["finalize_at"] - t_end
    cond = after < 1.5
    print(f"  B2) 封句发生在说完后 +{after:.2f}s（快路径期望 < 1.5s；慢路径会是 +2.2s）"
          f"｜封句瞬间音频静默 {obs['audio_quiet_at_final']:.2f}s"
          f"  {'OK' if cond else '✗'}")
    ok &= cond
    return ok


async def scenario_no_voice() -> bool:
    """C：全程只有房间噪声（判据从没听到人声）→ 只走慢路径，绝不抢跑。"""
    ok = True
    session, _src, obs = await run_chain(
        phases=[(QUIET, 4.5)], noise_db=-50.0,
        fast_silence=0.6, fast_quiet=0.6, inject_at="speech_end", inject_lead=0.0)
    # 这段没有「说话」，手工注入一次文本（时间最近一次文本增量在 1.0s 前）
    cond = not session.texts
    print(f"  未听到人声（只有 -50dBFS 房间噪声）→ 是否提前封句："
          f"{'否' if cond else '是 ✗'}（快路径阈值 0.6s；慢路径 3.0s 未到）  {'OK' if cond else '✗'}")
    ok &= cond
    cond = session._voiced_in_utterance is False
    print(f"  voiced 标记 = {session._voiced_in_utterance}（期望 False：从没判到人声）"
          f"  {'OK' if cond else '✗'}")
    ok &= cond
    return ok


async def scenario_no_voice_with_text() -> bool:
    """C': 有文本、音频侧「静」了，但从没听到人声 → 仍不许快封。"""
    ok = True
    eng = StubEngine()
    src = ScriptedSource([(QUIET, 4.5)], noise_db=-50.0)
    session = TimedSession(SessionConfig(
        api_key="test-key", final_silence_s=3.0,
        fast_final_silence_s=0.6, fast_final_audio_quiet_s=0.6))
    session.on_text = session.on_text_recorder
    session._ws = FakeWS()
    eng._session = session
    proxy = _SessionProxy(eng)
    task = asyncio.create_task(_pump_capture(proxy, None, src.total_s, None, "mic",
                                             lambda: src, None))
    injected = False
    t0 = time.perf_counter()
    while not task.done():
        await asyncio.sleep(0.02)
        # 等噪声底预热完（2s）再注入译文，避免「预热期占位」污染本用例
        if not injected and time.perf_counter() - t0 > 2.5:
            session._buf = ["你好世界。"]
            session._last_text_at = time.perf_counter()
            injected = True
        session.tick()
    await task
    quiet = (time.perf_counter() - session._last_voice_at) if session._last_voice_at else None
    cond = not session.texts
    print(f"  有译文 + 音频侧静了 {'∞' if quiet is None else f'{quiet:.1f}s'}"
          f"（快阈值 0.6s）但从没听到人声 → 是否封句："
          f"{'否（保守闸生效）' if cond else '是 ✗ 抢跑了'}  {'OK' if cond else '✗'}")
    ok &= cond
    cond = session._voiced_in_utterance is False
    print(f"  voiced 标记 = {session._voiced_in_utterance}（期望 False）  {'OK' if cond else '✗'}")
    ok &= cond
    return ok


async def main() -> None:
    print("test_fast_finalize_chain（真实 _pump_capture → _SessionProxy → 真实会话）:")
    print(" 1) 快路径：静音期旧量做不到、新量做得到，且封句真的提前")
    ok = await scenario_fast_path()
    print(" 2) 保守闸：从没听到人声时不提前封句")
    ok &= await scenario_no_voice()
    ok &= await scenario_no_voice_with_text()
    assert ok, "链路级用例失败（见上）"
    print("ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
