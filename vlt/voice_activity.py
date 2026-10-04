"""从上行音频的块电平里判断「这条采集腿上现在有没有人在说」。

## 治什么：#49 与 #57 都还没解决的最后一环

快封句（`session.fast_final_*`）要在「说话的人确实说完了」时提前封句，省掉那 3s 白等。
信号来源换过两次，第二次（#57）换成了 `_SessionProxy.send_audio` 里现成的
`loud = peak >= SILENCE_PEAK` → `session.note_voice()`。

那条信号解决了「距上次上送音频的间隔恒为 ~0.1s」的问题，但**门限本身不能用**：
`SILENCE_PEAK = 220` 是给「30s 长静音闸门」定的，约等于 -43.5dBFS **峰值** —— 低到把
**房间噪声**也算成「有人在说」。实测（`tests/test_voice_activity.py` 第 3 节，
100ms 块的峰值 vs 220）：

| 房间噪声底(RMS) | 静音块峰值中位数 | `peak>=220` 判成有声？ |
|---|---|---|
| -60 dBFS | 115 | 否（能判出静） |
| -55 dBFS | 206 | 勉强（20 块里有 7 块误判） |
| -50 dBFS | 363 | **是：20/20 全误判** |
| -45 dBFS | 647 | 是 → 顶死 |
| -40 dBFS | 1142 | 是 → 顶死 |

后果与 #49 那个 bug **同型**：`note_voice()` 被噪声持续刷新 → `user_quiet_s()` 恒 ~0.1s
→ 快路径在真麦克风上永远不触发，功能静默失效。而这只在**真实房间噪声**下发生 ——
现有 e2e 用例喂的是**全零**（数字静音），峰值恒为 0，所以两边都测不出来。

不是"极端环境"：本机实测用户那台 USB 麦（关麦状态）底噪就有 **-48.9dBFS**，正在失效区。

## 所以：相对噪声底

估计**噪声底**（最近窗口内块电平的低分位），判据是「当前块 ≥ 噪声底 + margin」。
麦克风增益、房间噪声、不同设备全被自动吸收 —— 上游是给**不认识的用户群**用的，
任何绝对门限都只是换个数字赌运气（底噪 + 增益在用户之间能差 20~30dB）。

## 两条纪律（宁可慢，不可抢跑）

- **预热期一律判「有人在说」**：窗口没攒够样本时噪声底不可信 → 压住「上游已静」的计时，
  退化成慢路径（不省时间，但绝不会把半句当终版发出去）。
- **预热期那个 `True` 不当作「听到了人声」**：它只是占位。会话侧「从没收到过信号」
  （`user_quiet_s() is None`）本来就是保守走慢路径，靠 `self.ready` 区分（见
  `engine._SessionProxy.send_audio`）。
"""
from __future__ import annotations

import math
from collections import deque

# 噪声底窗口：30s @ 100ms 一块。窗口太短会被「连续说话」拉高（说话时窗口里没有静音
# 样本），太长则换了房间要等很久才自适应。30s 是折中：连续说话超过 ~27s 才会开始拉高，
# 而那时本来也不需要「提前封句」。
VOICE_WINDOW_CHUNKS = 300

# 预热：窗口里至少这么多块才敢信噪声底（2s @100ms）。
# 代价：一段会话**开场 2s 内**不走快路径（保守方向，只是慢一点）。
VOICE_MIN_SAMPLES = 20

# 噪声底取窗口内的低分位（不是最小值：最小值会被单块的异常低值带跑）。
VOICE_FLOOR_PERCENTILE = 10.0

# 「有人说话」要高出噪声底多少 dB。10dB 是保守值：真人说话通常高出 15~30dB。
DEFAULT_VOICE_MARGIN_DB = 10.0
VOICE_MARGIN_MIN_DB = 3.0
VOICE_MARGIN_MAX_DB = 30.0


class VoiceActivity:
    """按块电平判断「有人在说」（相对噪声底，自动适应增益与房间噪声）。

    `feed()` 每次喂一个 100ms 块的电平（dBFS，来自 `engine.chunk_level_db`），
    返回「这一块算不算有人在说」。用法见 `engine._SessionProxy.send_audio`。
    """

    def __init__(self, margin_db: float = DEFAULT_VOICE_MARGIN_DB,
                 window_chunks: int = VOICE_WINDOW_CHUNKS,
                 min_samples: int = VOICE_MIN_SAMPLES) -> None:
        self.margin_db = float(margin_db)
        self.window_chunks = max(2, int(window_chunks))
        self.min_samples = max(1, int(min_samples))
        self._levels: deque[float] = deque(maxlen=self.window_chunks)
        self.floor_db: float | None = None      # 噪声底估计（None = 还没攒够样本）
        self.chunks = 0                         # 累计喂进来的块数
        self.voice_chunks = 0                   # 累计判为「有人在说」的块数
        self.last_level_db: float | None = None

    def feed(self, level_db: float) -> bool:
        """喂一块电平，返回「这块算不算有人在说」。

        ⚠️ **预热期的 `True` 不算数**：那是「不知道，先按有人在说保守处理」，用来压住
        「上游已静」的计时；它**不能**当作「本段听到了人声」的证据 —— 后者只认
        `ready` 之后的判定，调用方用 `self.ready` 区分（见 `engine._SessionProxy`）。
        """
        level = float(level_db)
        self.last_level_db = level
        self.chunks += 1
        self._levels.append(level)

        if not self.ready:
            self.floor_db = None
            self.voice_chunks += 1
            return True                          # 预热期：保守按「有人在说」

        self.floor_db = self._percentile(self._levels)
        voice = level >= self.floor_db + self.margin_db
        if voice:
            self.voice_chunks += 1
        return voice

    @property
    def ready(self) -> bool:
        """噪声底是否可信（窗口攒够了样本）。未 ready 时判据一律按保守方向走。"""
        return len(self._levels) >= self.min_samples

    def reset(self) -> None:
        """清空估计（换设备/重开采集时用；噪声底必须重新学习）。"""
        self._levels.clear()
        self.floor_db = None
        self.chunks = 0
        self.voice_chunks = 0
        self.last_level_db = None

    def _percentile(self, values: deque[float]) -> float:
        """低分位（不引 numpy：窗口只有几百个数，排序比 import 便宜且离线可测）。"""
        ordered = sorted(values)
        n = len(ordered)
        if n == 1:
            return ordered[0]
        # 线性插值分位（与 numpy.percentile 默认口径一致），避免只取第 k 个造成台阶
        pos = (VOICE_FLOOR_PERCENTILE / 100.0) * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac
