"""端到端演示：三种任务的样例对话（文档 §41 实例）。

用法:
    cd Cog-Sim && python -m simulator.run_sim --task bargain --turns 3
    cd Cog-Sim && python -m simulator.run_sim --task support --turns 2
    cd Cog-Sim && python -m simulator.run_sim --task donation --turns 2
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from simulator.llm import LLMClient
from simulator.profile.cognitive_profile import build_habit_card
from simulator.simulator import UserSimulator
from simulator.state.schema import BDIItem, CognitiveProfile, Emotion, UserState

# ---- 三种任务的初始场景（文档 §41）----
SCENARIOS = {
    "support": {
        "persona": (
            "你是一名研二学生，最近论文进展不顺利，导师催得紧，你感到压力很大、"
            "有些焦虑。你平时说话比较克制，不太主动求助。"
        ),
        "profile": CognitiveProfile(eta_R=0.70, tau_A=0.35, tau_R=0.75),
        "beliefs": [BDIItem("B1", "belief", "这件事我完全解决不了", 3.5)],
        "desires": [BDIItem("D1", "desire", "我希望恢复一些控制感", 3.2)],
        "intentions": [BDIItem("I1", "intention", "我暂时不想采取行动", 2.8, polarity="avoid")],
        "emotion": Emotion(valence=-0.45, arousal=0.65, category="anxiety"),
        "script": [
            "你不需要一次解决所有问题，可以先明天只给导师发一条消息。",
            "这听起来是个可行的第一步，先从最小的行动开始。",
            "你今天最担心的是什么？",
        ],
    },
    "donation": {
        "persona": (
            "你是一个普通上班族，收入稳定但最近刚买了房，日常开销比较在意。"
            "你对公益有好感，但担心钱花得不值。"
        ),
        "profile": CognitiveProfile(eta_R=0.65, tau_A=0.40, tau_R=0.80),
        "beliefs": [BDIItem("B1", "belief", "小额捐赠作用很有限", 3.0)],
        "desires": [
            BDIItem("D1", "desire", "我希望帮助别人", 2.5),
            BDIItem("D2", "desire", "我想控制自己的支出", 3.5),
        ],
        "intentions": [BDIItem("I1", "intention", "目前不准备捐款", 2.8, polarity="avoid")],
        "emotion": Emotion(valence=0.0, arousal=0.3, category="neutral"),
        "script": [
            "其实小额捐赠的边际价值很高：一个公益项目里，5元就能让一个孩子吃上一顿热饭，积少成多，效果是实打实的。",
            "很多人都是从小额开始的，而且公益平台现在都能查到每一笔钱的去向。",
        ],
    },
    "bargain": {
        "persona": (
            "你是一个想买二手自行车的学生，预算有限，但明天开学要用车，"
            "所以比较着急。你砍价比较直接。"
        ),
        "profile": CognitiveProfile(eta_R=0.75, tau_A=0.45, tau_R=0.80),
        "beliefs": [BDIItem("B1", "belief", "卖家最终可能接受80元", 3.0)],
        "desires": [
            BDIItem("D1", "desire", "希望价格尽可能低", 3.8),
            BDIItem("D2", "desire", "希望今天成交", 3.1),
        ],
        "intentions": [BDIItem("I1", "intention", "坚持80元", 3.0)],
        "emotion": Emotion(valence=0.1, arousal=0.4, category="neutral"),
        "script": [
            "85是最低价，我今天已经拒绝几个80的报价。",
            "这车成色很新，85已经比市场价低了，你今晚就能骑走。",
            "行，那今天先聊到这，你考虑好了随时联系我。",
        ],
    },
}


def build_state(sc: dict) -> UserState:
    # v1.0.1-fix（修复提案 05 问题 1）：深拷贝场景，避免多次构造的状态之间
    # 以及状态与模块级 SCENARIOS 之间共享 BDIItem 对象
    sc = copy.deepcopy(sc)
    state = UserState(
        persona=sc["persona"],
        profile=sc["profile"],
        beliefs=list(sc["beliefs"]),
        desires=list(sc["desires"]),
        intentions=list(sc["intentions"]),
        emotion=sc["emotion"],
        history=[],
    )
    state.habit_card = build_habit_card(
        state.profile.eta_R, state.profile.tau_A, state.profile.tau_R
    )
    return state


def fmt_bdi(state: UserState) -> str:
    lines = []
    for lst in (state.beliefs, state.desires, state.intentions):
        for it in lst:
            tag = it.type[:1].upper()
            if it.polarity == "avoid" and it.type != "belief":
                tag += "/avoid"
            if not it.active:
                tag += "/inactive"
            lines.append(f"  {it.id} [{tag}] ({it.strength:.2f}): {it.content}")
    return "\n".join(lines) or "  (空)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(SCENARIOS), default="bargain")
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--model", default=None, help="默认 deepseek-flash")
    ap.add_argument("--route-mode", choices=["deterministic", "bernoulli"],
                    default="deterministic", help="Route 采样模式（文档 §17、§17.1）")
    ap.add_argument("--no-debug", action="store_true",
                    help="production 模式：不请求纯审计字段（reason/dominant_cues），不打印链路追踪")
    ap.add_argument("--out", default=None, help="日志输出路径")
    args = ap.parse_args()

    sc = SCENARIOS[args.task]
    sim = UserSimulator(build_state(sc), LLMClient(model=args.model),
                        route_mode=args.route_mode, debug=not args.no_debug)
    print(f"===== 任务: {args.task} | 模型: {sim.llm.model} | route_mode: {args.route_mode} =====")
    print(f"Persona: {sc['persona']}")
    print(f"Theta: eta_R={sc['profile'].eta_R}, tau_A={sc['profile'].tau_A}, tau_R={sc['profile'].tau_R}")
    print("初始状态:")
    print(fmt_bdi(sim.state))
    print(f"Emotion: {sim.state.emotion}")
    print()

    for i, agent_reply in enumerate(sc["script"][: args.turns], 1):
        print(f"--- 第 {i} 轮 ---")
        print(f"[assistant] {agent_reply}")
        user_reply = sim.simulate_turn(agent_reply)
        log = sim.logs[-1]
        print(f"  mode={log.mode} ({log.mode_reason or '—'})")
        if log.route:
            print(f"  route={log.route}  judgment={log.judgment}  p_C={log.p_central}  "
                  f"d_t={log.discrepancy}  rel={log.relevance}  arg={log.argument_strength}  "
                  f"cue={log.cue_strength}  pressure={log.interaction_pressure}"
                  f"{'  cues=' + str(log.dominant_cues) if log.dominant_cues else ''}")
        elif log.interaction_pressure is not None:
            print(f"  pressure={log.interaction_pressure}")
        if log.target:
            print(f"  target: {log.target}")
        if log.relevant_state_ids:
            print(f"  relevant_state: {log.relevant_state_ids}")
        print(f"  appraisal: GC={log.appraisal['goal_congruence']:.2f} "
              f"CP={log.appraisal['coping_potential']:.2f} FE={log.appraisal['future_expectancy']:.2f}")
        print(f"  emotion: {log.emotion_before['category']}({log.emotion_before['valence']:.2f},{log.emotion_before['arousal']:.2f})"
              f" -> {log.emotion_after['category']}({log.emotion_after['valence']:.2f},{log.emotion_after['arousal']:.2f})")
        if log.reaction_plan:
            print(f"  reaction_plan: {log.reaction_plan}")
        for n in log.update_notes:
            print(f"  [updater] {n}")
        if sim.debug:
            print(f"  [trace] {sim.chain_trace(log)}")
        print(f"[user] {user_reply}")
        if log.user_done:
            print(f"  user_done=True ({log.done_reason})")
        if sim.conversation_ended:
            print("[driver] 用户已结束对话，停止追问")
            break
        print()

    out = args.out or Path(__file__).resolve().parent.parent / "runs" / f"{args.task}_log.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    sim.dump_logs(out)
    print("===== 状态轨迹 =====")
    print(fmt_bdi(sim.state))
    print(f"Emotion: {sim.state.emotion}")
    print(f"\n日志已写入 {out}")


if __name__ == "__main__":
    main()
