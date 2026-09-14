"""Deterministic Updater（文档 §31）：LLM 输出不是最终状态。

程序执行 C_{t+1} = Apply(C_t, LLMUpdates, R_t, J_t)，检查：
强度范围、更新幅度、Judgment 方向、Peripheral 限制、Intention 支持、数量限制，
以及收紧后的两条：新增核心节点强度上限（所有 Route 一致）、
冲突衰减计入 victim 本轮变化预算（不得绕过 RJ 限幅）。
所有被约束的动作都会写进 notes，保证过程可审计。
"""
from __future__ import annotations

from simulator.cognitive.rj_contract import limits_for
from simulator.state.schema import (
    BDIItem,
    DEACTIVATE_THRESHOLD,
    MAX_ITEMS,
    POLARITIES,
    STRENGTH_MAX,
    STRENGTH_MIN,
    UserState,
)
from simulator.utils import clamp

INTENTION_EPSILON = 0.5   # 文档 §31 的 epsilon（文档未给具体值，取 0.5）
SUPPORT_STRENGTH = 2.0    # “相关强 Belief/Desire”的强度门槛


def constrained_apply(
    state: UserState,
    proposal: dict,
    route: str,
    judgment: str,
    related_belief_ids: list[str],
) -> list[str]:
    """就地应用 proposal 到 state，返回约束审计 notes。"""
    notes: list[str] = []
    limits = limits_for(route, judgment)
    applied_deltas: dict[str, float] = {}

    # ---- 1. 更新已有节点 ----
    # v1.0.1-fix（修复提案 05 问题 2/3a/3b/5）：
    # - 元素级类型守卫（问题 5）；
    # - 轮级限幅：以轮初强度为基准核算净变化，重复更新合计不得突破 limits（问题 2）；
    # - 激活判定移到幅度裁剪之后（问题 3a）；
    # - 重新激活同样遵守容量上限，超限拒绝并记 note（问题 3b，拒绝方案）。
    turn_start: dict[str, float] = {}
    for u in proposal.get("bdi_updates", []):
        if not isinstance(u, dict):
            notes.append("[robustness] bdi_updates 含非对象元素，忽略")
            continue
        item_id = str(u.get("id", ""))
        item = state.find(item_id)
        if item is None:
            notes.append(f"跳过不存在节点 {item_id}")
            continue
        try:
            new_s = clamp(u.get("new_strength", item.strength), STRENGTH_MIN, STRENGTH_MAX)
        except (TypeError, ValueError):
            notes.append(f"[robustness] {item_id} new_strength 数值非法，忽略该更新")
            continue
        start = turn_start.setdefault(item_id, item.strength)
        cap = limits[item.type]
        net = new_s - start
        if abs(net) > cap:
            net = cap * (1 if net > 0 else -1)
            notes.append(f"{item_id} 本轮净变化 {new_s - start:+.2f} 超限 {cap}，截断为 {net:+.2f}")
        final_s = start + net

        if not item.active:
            # 已退役节点只允许被重新激活：裁剪后的强度需回升到阈值以上（问题 3a）
            if final_s >= DEACTIVATE_THRESHOLD + 0.2:
                active_count = sum(1 for i in state.items(item.type) if i.active)
                if active_count >= MAX_ITEMS[item.type]:
                    notes.append(f"{item_id} 重新激活将使 {item.type} 活跃数超过上限 "
                                 f"{MAX_ITEMS[item.type]}，拒绝激活")
                    continue
                item.active = True
                notes.append(f"{item_id} 强度回升至 {final_s:.2f}，重新激活")
            else:
                notes.append(f"跳过未激活节点 {item_id}（裁剪后强度 {final_s:.2f} 不足以重新激活）")
                continue

        # Judgment=Reject 方向限制（文档 §31）：被拒命题相关 core Belief 不允许正向更新
        if judgment == "reject" and item.type == "belief" and item.id in related_belief_ids and final_s > item.strength:
            notes.append(f"{item_id} 与被拒命题相关，禁止正向更新（{final_s - item.strength:+.2f} -> 0）")
            final_s = item.strength

        # Peripheral 核心限制（文档 §31）：核心 Desire 不允许大幅改变
        if route == "peripheral" and item.type == "desire" and judgment != "accept" and final_s != item.strength:
            notes.append(f"Peripheral+{judgment} 不允许修改核心 Desire {item_id}")
            final_s = item.strength

        applied_deltas[item.id] = applied_deltas.get(item.id, 0.0) + (final_s - item.strength)
        item.strength = final_s

    # ---- 2. 新增节点 ----
    new_intention_ids: set[str] = set()
    for n in proposal.get("new_items", []):
        if not isinstance(n, dict):
            notes.append("[robustness] new_items 含非对象元素，忽略")
            continue
        type_ = str(n.get("type", "")).strip().lower()
        if type_ not in MAX_ITEMS:
            notes.append(f"未知节点类型 {type_!r}，拒绝新增")
            continue
        content = str(n.get("content", "")).strip()
        if not content:
            notes.append("空内容节点，拒绝新增")
            continue
        try:
            strength = clamp(n.get("strength", 1.0), STRENGTH_MIN, STRENGTH_MAX)
        except (TypeError, ValueError):
            notes.append("[robustness] 新增节点 strength 数值非法，取默认 1.0")
            strength = 1.0
        is_cue = bool(n.get("cue", False))
        is_core = bool(n.get("core", True))
        polarity = str(n.get("polarity", "approach")).strip().lower()
        if type_ == "belief":
            polarity = "approach"   # Belief 不使用极性字段
        elif polarity not in POLARITIES:
            notes.append(f"非法 polarity {polarity!r}，默认 approach")
            polarity = "approach"

        # 新增节点强度受 RJ 幅度约束（§31 收紧）：
        # 一律以本回合 limits[type_] 为强度上限，不因 core=False 而豁免
        # （旧版 core=False/cue=True 即可绕过限幅，Central+Noncommit 下
        #  曾插入 2.2 的强新信念）。唯一豁免：Peripheral+Accept 下的 cue 信念
        # （文档 §25：trust / authority / social norm 等 cue 信念允许增强）。
        # Reject 下新增 Intention 的上限由此一并覆盖（limits["intention"]）。
        exempt_cue = (route == "peripheral" and judgment == "accept"
                      and type_ == "belief" and is_cue)
        cap = STRENGTH_MAX if exempt_cue else limits[type_]
        if cap == 0.0:
            notes.append(f"{route}+{judgment} 禁止新增 {type_}（限制为 0），拒绝")
            continue
        if strength > cap:
            notes.append(f"新增 {type_} 强度 {strength} 超限 {cap}，截断为 {cap}")
            strength = cap

        new_item = BDIItem(
            id=f"{type_[0].upper()}{_next_id(state, type_)}",
            type=type_,
            content=content,
            strength=strength,
            core=is_core,
            active=True,
            polarity=polarity,
        )
        # 弱非核心节点直接以退役状态入库，不占用配额（扩展规则，README 偏差第 6 条）
        if not is_core and strength < DEACTIVATE_THRESHOLD:
            new_item.active = False
            notes.append(f"新增 {new_item.id} [{type_}] strength={strength:.2f} 过弱，直接退役")
        lst = state.items(type_)
        if sum(1 for i in lst if i.active) >= MAX_ITEMS[type_]:
            evicted = _evict(state, type_, new_item)
            if evicted == "incoming":
                notes.append(f"{type_} 数量达上限且新节点更弱，拒绝新增")
            else:
                notes.append(f"{type_} 数量达上限 {MAX_ITEMS[type_]}，淘汰 {evicted}，新增 {new_item.id}")
        else:
            lst.append(new_item)
            if new_item.active:
                notes.append(f"新增 {new_item.id} [{type_}] strength={strength:.2f}"
                             + (f" polarity={polarity}" if type_ != "belief" else ""))
        if type_ == "intention" and new_item in state.intentions:
            new_intention_ids.add(new_item.id)

        # Belief 冲突衰减（扩展规则，§31 收紧）：
        # 新增 Belief 与已有 Belief 冲突时，已有 Belief 按新节点强度衰减，
        # 避免状态同时强持有 P 和 ¬P。但该衰减是对 victim 的强度修改，
        # 必须计入 victim 本轮总变化并受本回合 belief 限幅约束
        # （旧版按新节点全强度扣减，Central+Noncommit 限幅 0.4 时
        # 一轮仍可跌 2.5，等于绕过了 RJ 幅度约束）。
        if type_ == "belief" and new_item in state.beliefs:
            conf_ids = [str(c) for c in n.get("conflicts_with", []) or []]
            if not new_item.active and conf_ids:
                # 新信念过弱已退役 = 用户几乎没接收该命题，不应产生冲突衰减压力
                notes.append(f"新 Belief {new_item.id} 未激活（过弱退役），"
                             f"对 {conf_ids} 不产生冲突衰减")
                continue
            for cid in conf_ids:
                victim = state.find(cid)
                if victim is None or victim.type != "belief" or not victim.active:
                    continue
                already = applied_deltas.get(victim.id, 0.0)
                # |already - decay| <= limits["belief"] -> 剩余可衰减额度
                budget = limits["belief"] + already   # already 可能为负
                decay = min(strength, max(0.0, budget))
                if decay <= 1e-9:
                    notes.append(f"新 Belief {new_item.id} 与 {cid} 冲突，但 {cid} 本轮变化已达 "
                                 f"belief 限幅 {limits['belief']}，暂不衰减")
                    continue
                victim.strength = clamp(victim.strength - decay, STRENGTH_MIN, STRENGTH_MAX)
                applied_deltas[victim.id] = already - decay
                notes.append(f"新 Belief {new_item.id} 与 {cid} 冲突，{cid} 按限幅 {limits['belief']} "
                             f"衰减 {decay:.2f} → {victim.strength:.2f}")

    # ---- 3. Intention 支持检查（文档 §31）----
    # |ΔI| > ε 时，必须有：当轮同方向 B/D 更新，或现存强 Belief/Desire 可解释。
    for item in state.intentions:
        # 新增 Intention 节点同样要检查：strength > ε 需有支持，否则截断。
        if item.id in new_intention_ids:
            delta = item.strength  # 从 0 新增，视为全量变化
        else:
            delta = applied_deltas.get(item.id, 0.0)
        if abs(delta) <= INTENTION_EPSILON:
            continue
        bd_same_direction = any(
            d * delta > 0 and abs(d) >= 0.2
            for iid, d in applied_deltas.items()
            if state.find(iid) is not None and state.find(iid).type in ("belief", "desire")
        )
        strong_bd = any(
            i.active and i.strength >= SUPPORT_STRENGTH
            for i in state.beliefs + state.desires
        )
        if not (bd_same_direction or strong_bd):
            capped = item.strength - delta + INTENTION_EPSILON * (1 if delta > 0 else -1)
            notes.append(f"{item.id} 大额 Intention 变化 {delta:+.2f} 无 Belief/Desire 支持，截断到 {capped:.2f}")
            item.strength = clamp(capped, STRENGTH_MIN, STRENGTH_MAX)

    # ---- 4. 退役扫描（扩展规则）：非核心弱节点 deactivate ----
    for lst in (state.beliefs, state.desires, state.intentions):
        for it in lst:
            if it.active and not it.core and it.strength < DEACTIVATE_THRESHOLD:
                it.active = False
                notes.append(f"{it.id} [{it.type}] 强度 {it.strength:.2f} < {DEACTIVATE_THRESHOLD}，退役（inactive）")

    return notes


def _next_id(state: UserState, type_: str) -> int:
    nums = []
    for it in state.items(type_):
        digits = "".join(ch for ch in it.id if ch.isdigit())
        nums.append(int(digits) if digits else 0)
    return max(nums, default=0) + 1


def _evict(state: UserState, type_: str, incoming: BDIItem) -> str:
    """数量超限时淘汰一个节点（文档 §31），返回被淘汰的 id。

    淘汰优先级：退役节点 > 非核心弱节点。
    若 incoming 比所有现存活跃节点都弱且非核心，则拒绝新增（返回 "incoming"）。
    """
    lst = state.items(type_)
    cands = sorted(lst, key=lambda i: (i.active, i.core, i.strength))  # v1.0.1-fix（问题 4）：最弱优先
    victim = cands[0]
    if (not victim.active) or ((not victim.core) and incoming.strength >= victim.strength):
        lst.remove(victim)
        lst.append(incoming)
        return victim.id
    return "incoming"
