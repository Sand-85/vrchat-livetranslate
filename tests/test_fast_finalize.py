#!/usr/bin/env python
"""「快封句」验收：上行音频也静了 → 不白等那 3s（但绝不在人还在说时抢跑）。

## 实测依据（2026-10-01 真链路 2 句）

| 指标 | 短句(2.0s) | 长句(4.8s) |
|---|---|---|
| 首条译文**上屏** | 开口后 +1.26s | 开口后 +3.29s |
| 累计译文**停止增长** | 说完前 0.89s | 说完前 0.74s |
| 终版（我们自己的 tick 兜底发的） | 说完后 +2.14s | 说完后 +2.44s |

服务端在用户停止说话后的 **8s 内一条事件都不发**（既无 `response.text.done` 也无
`response.done`）→ 没有语义完成信号可用，只能靠定时器；而译文早在说完前就不长了
→ 那 3s 全是白等。故加**双条件**：上行音频静 ≥ `fast_final_audio_quiet_s` 时，
文字静默 `fast_final_silence_s` 就封。

⚠️ 为什么快路径**必须**带「音频也静了」：实测连续说话时相邻 delta 可间隔 **2.3s**
（那是句子**中间**的停顿）—— 只看文字静默会把半句当最终版发出去（抢跑）。

⚠️ 2026-10-04 修正（审核打回）：音频侧判据原先用「距上次 `send_audio` 的间隔」，
而采集腿每块都发 → 那个量恒为 ~0.1s，快路径从未触发。现在它由采集侧按电平相对
噪声底判定（`vlt/voice_activity.py`），本文件的「接线」一节验的就是这条链。
链路级（真实采集泵 + 真实代理 + 真实会话）的验收在 `test_fast_finalize_chain.py`。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlt.session.base import (DEFAULT_FAST_FINAL_AUDIO_QUIET_S,   # noqa: E402
                              DEFAULT_FAST_FINAL_SILENCE_S,
                              DEFAULT_FINAL_SILENCE_S, SessionConfig, should_finalize)
from vlt.session.qwen38 import QwenLiveTranslateSession          # noqa: E402


def test_pure_table() -> bool:
    ok = True
    # (文字静默, 音频静默, 期望, 说明)
    cases: list[tuple[float, float | None, bool, str]] = [
        (3.10, 0.30, True,  "慢路径：人还在说，但文字静默过了 3s → 照旧封"),
        (2.30, 0.20, False, "**不抢跑**：连续说话时那个 2.3s 大间隔（音频侧没静）"),
        (1.60, 1.60, True,  "快路径：音频静了 1.6s + 文字静默 1.6s → 封（省 ~1.5s）"),
        (0.90, 1.60, False, "**不抢跑**：音频静了很久，但文字才静 0.9s（服务端可能还在追）"),
        (1.60, 1.40, False, "**不抢跑**：文字静了 1.6s，但音频还没静够（1.40 < 1.5）"),
        (1.60, 1.50, True,  "正好卡在音频阈值上（≥ 即封）"),
        (1.09, 1.60, False, "文字侧差一点（1.09 < 1.1）就不封"),
    ]
    for text_q, audio_q, want, why in cases:
        got = should_finalize(text_quiet_s=text_q, audio_quiet_s=audio_q)
        cond = got is want
        print(f"  文字静默={text_q:.2f}s 音频静默={audio_q!s:>5} → {got}"
              f"（期望 {want}）{'OK' if cond else '✗'}  {why}")
        ok &= cond
    return ok


def test_pure_edges() -> bool:
    ok = True
    # 从未上送过音频（没有音频在流 = 没人在说话）→ 音频侧按「已静」处理
    got = should_finalize(text_quiet_s=1.6, audio_quiet_s=None)
    cond = got is True
    print(f"  从未上送音频 + 文字静默 1.6s → {got}（期望 True）  {'OK' if cond else '✗'}")
    ok &= cond
    # 关掉快路径（fast_final_silence_s=None）→ 退回纯 3.0s 行为
    got = should_finalize(text_quiet_s=1.6, audio_quiet_s=9.9, fast_silence_s=None)
    cond = got is False
    print(f"  关掉快路径 + 文字静默 1.6s → {got}（期望 False）  {'OK' if cond else '✗'}")
    ok &= cond
    got = should_finalize(text_quiet_s=3.1, audio_quiet_s=9.9, fast_silence_s=None)
    cond = got is True
    print(f"  关掉快路径 + 文字静默 3.1s → {got}（期望 True，慢路径还在）  {'OK' if cond else '✗'}")
    ok &= cond
    # 本段没听到过人声（麦被静音/增益过低/太小声）→ 只走慢路径，绝不抢跑
    got = should_finalize(text_quiet_s=1.6, audio_quiet_s=9.9, voiced=False)
    cond = got is False
    print(f"  本段没听到人声 + 文字静默 1.6s → {got}（期望 False，只走慢路径）  {'OK' if cond else '✗'}")
    ok &= cond
    got = should_finalize(text_quiet_s=3.1, audio_quiet_s=9.9, voiced=False)
    cond = got is True
    print(f"  本段没听到人声 + 文字静默 3.1s → {got}（期望 True，慢路径不受影响）  {'OK' if cond else '✗'}")
    ok &= cond
    # 阈值可调
    got = should_finalize(text_quiet_s=0.6, audio_quiet_s=1.0, fast_silence_s=0.55,
                          fast_quiet_s=0.5)
    cond = got is True
    print(f"  自定义阈值（文字 0.55s / 音频 0.5s）→ {got}（期望 True）  {'OK' if cond else '✗'}")
    ok &= cond
    # 默认值一致性（别被悄悄改小：音频侧必须 > 文字侧，否则真正起作用的是文字侧那 1.1s）
    cond = (DEFAULT_FAST_FINAL_SILENCE_S == 1.1
            and DEFAULT_FAST_FINAL_AUDIO_QUIET_S > DEFAULT_FAST_FINAL_SILENCE_S
            and DEFAULT_FINAL_SILENCE_S == 3.0)
    print(f"  默认阈值：文字快={DEFAULT_FAST_FINAL_SILENCE_S}s"
          f"音频={DEFAULT_FAST_FINAL_AUDIO_QUIET_S}s（须 > 文字快），慢={DEFAULT_FINAL_SILENCE_S}s"
          f"  {'OK' if cond else '✗'}")
    ok &= cond
    return ok


def test_tick_wiring() -> bool:
    """真实 `QwenLiveTranslateSession.tick()` 接线：两条路径都要按判据行动。"""
    ok = True

    def sealed(*, text_quiet: float, audio_quiet: float | None, voiced: bool = True,
               cfg: SessionConfig | None = None) -> bool:
        s = QwenLiveTranslateSession(cfg or SessionConfig(api_key="x"))
        s._buf = ["こんにちは。"]                     # 有一段累计译文
        now = time.perf_counter()
        s._last_text_at = now - text_quiet
        s._last_voice_at = (now - audio_quiet) if audio_quiet is not None else 0.0
        s._voiced_in_utterance = voiced
        out: list[bool] = []
        s._emit = lambda **kw: out.append(bool(kw.get("is_final")))   # type: ignore[method-assign]
        s.tick()
        return bool(out and out[0])

    cond = sealed(text_quiet=1.6, audio_quiet=1.6) is True
    print(f"  真实 tick：文字静默 1.6s + 音频静 1.6s → 封句  {'OK' if cond else '✗'}")
    ok &= cond
    cond = sealed(text_quiet=1.6, audio_quiet=0.2) is False
    print(f"  真实 tick：文字静默 1.6s + 人还在说 → 不封  {'OK' if cond else '✗'}")
    ok &= cond
    cond = sealed(text_quiet=3.1, audio_quiet=0.2) is True
    print(f"  真实 tick：文字静默 3.1s（人还在说）→ 慢路径封  {'OK' if cond else '✗'}")
    ok &= cond
    cond = sealed(text_quiet=1.6, audio_quiet=1.6, voiced=False) is False
    print(f"  真实 tick：本段没听到人声 → 不封（哪怕音频「静」着）  {'OK' if cond else '✗'}")
    ok &= cond
    cond = sealed(text_quiet=1.6, audio_quiet=1.6,
                  cfg=SessionConfig(api_key="x", fast_final_silence_s=None)) is False
    print(f"  真实 tick：关掉快路径 → 不封  {'OK' if cond else '✗'}")
    ok &= cond
    # note_voice() 是采集侧唯一的入口：调它 = 「刚听到人声」
    s = QwenLiveTranslateSession(SessionConfig(api_key="x"))
    cond = s._voiced_in_utterance is False and s._last_voice_at == 0.0
    print(f"  新会话初始：未听到人声、无时间戳  {'OK' if cond else '✗'}")
    ok &= cond
    s.note_voice()
    cond = s._voiced_in_utterance is True and s._last_voice_at > 0.0
    print(f"  note_voice() 后：标记为「听到过」并记时间  {'OK' if cond else '✗'}")
    ok &= cond
    # 新的一段（response.created）→ 本段的「听到过吗」重新计
    s._handle_event({"type": "response.created"})
    cond = s._voiced_in_utterance is False and s._last_voice_at > 0.0
    print(f"  response.created 后：本段标记重置，但时间戳保留  {'OK' if cond else '✗'}")
    ok &= cond
    # 已经封过同一段文本 → 不重复发（既有行为，别被改坏）
    s = QwenLiveTranslateSession(SessionConfig(api_key="x"))
    s._buf = ["こんにちは。"]
    now = time.perf_counter()
    s._last_text_at, s._last_voice_at = now - 5.0, now - 5.0
    s._voiced_in_utterance = True
    s._last_final_text = "こんにちは。"
    out: list[bool] = []
    s._emit = lambda **kw: out.append(True)          # type: ignore[method-assign]
    s.tick()
    cond = not out
    print(f"  真实 tick：同一段文本已封过 → 不重复发  {'OK' if cond else '✗'}")
    ok &= cond
    return ok


if __name__ == "__main__":
    print("test_fast_finalize:")
    print(" 1) 纯函数判据表（含三条「不抢跑」）")
    ok = test_pure_table()
    print(" 2) 纯函数边界（从未上送/关掉快路径/没听到人声/可调阈值/默认值）")
    ok &= test_pure_edges()
    print(" 3) 真实 tick 接线 + note_voice 入口 + response.created 重置")
    ok &= test_tick_wiring()
    assert ok, "快封句用例失败（见上）"
    print("ALL PASSED")
