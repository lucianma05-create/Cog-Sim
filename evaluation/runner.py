"""Validation 批量运行器（pilot + 全量）。

用法：
    python -m evaluation.runner seeds           # seed 批量（程序化指标 + 失败收集）
    python -m evaluation.runner evaluators      # LLM 盲评（state-utterance/realism/neutrality）
    python -m evaluation.runner controllability # η/τ 可控性实验
    python -m evaluation.runner ablation        # M0/M1/M2
    python -m evaluation.runner components      # ATC/TRIE/JEE 标注集 + robustness
    python -m evaluation.runner longhorizon     # 8 轮轨迹
    python -m evaluation.runner report          # 汇总生成 evaluation_report.md

结果写入 evaluation/results/*.json，失败案例 evaluation/failures/failure_XXX.json。
规模：默认 pilot（seeds 全量 120 × 1 次重复）；--repeats N 与 --limit K 可调全量。
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from simulator.llm import LLMClient
from simulator.run_sim import SCENARIOS, build_state
from simulator.simulator import UserSimulator

from evaluation import seeds as seeds_mod
from evaluation.state_transition import (direction_consistency, rj_violations,
                                         state_stability, strict_unrelated_rates,
                                         summarize_logs, unsupported_intention,
                                         legacy_unrelated_change_rate)
from evaluation.evaluators import (evaluate_neutrality_v2, evaluate_realism,
                                   evaluate_state_utterance, label_leakage)
from evaluation.trajectory_consistency import (drift_score, emotion_continuity,
                                               persona_consistency, repetition_score,
                                               reversal_flags)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FAILURES = ROOT / "failures"
RESULTS.mkdir(exist_ok=True)
FAILURES.mkdir(exist_ok=True)

TASK_GOALS = {"bargain": "make the user accept the 85 price and close the deal",
              "donation": "make the user donate",
              "support": "make the user feel better and take action"}

LONGHORIZON_SCRIPTS = {
    "bargain": [
        "85是最低价，我今天已经拒绝几个80的报价。",
        "这车成色很新，85已经比市场价低了。",
        "已经有好几个人在问了，你再不决定今晚就卖给别人了。",
        "你心里的预算到底是多少？",
        "行，80卖你。我卖车这么多年，跟学生们都是这么爽快成交的。",
        "那今天就定下来吧，你加5块85，我送你一个车锁。",
        "今天天气不错，骑车上学挺舒服的。",
        "行，那今天先聊到这，你考虑好了随时联系我。",
    ],
    "donation": [
        "5元就能让一个孩子吃上一顿热饭，积少成多，效果是实打实的。",
        "很多人都是从小额开始的，公益平台都能查到每一笔钱的去向。",
        "我上次捐款的平台是知名大机构背书的，大家都说靠谱。",
        "你之前有过捐款的经历吗？",
        "那就不劝你了，先不捐，等你看到具体项目的账目再决定，这样最稳妥。",
        "如果你能看到具体账目，会不会考虑捐一点？",
        "最近工作忙不忙？",
        "好，那你有兴趣的时候随时联系我。",
    ],
    "support": [
        "先给导师发一条消息只需要两分钟，回复一句进展顺利就够。",
        "这听起来是个可行的第一步，先从最小的行动开始。",
        "我认识几个研究生都是先发消息之后情况就好了。",
        "你今天最担心的是什么？",
        "那就先别发消息，等你准备好再说，这不是逃避。",
        "很多同学也都是先缓一缓，等状态好了再联系导师。",
        "听起来这段时间你压力很大，真的辛苦了。",
        "没关系，你已经做得很好了，慢慢来。",
    ],
}


def make_sim(task: str, profile: dict | None = None, llm: LLMClient | None = None) -> UserSimulator:
    sc = copy.deepcopy(SCENARIOS[task])
    state = build_state(sc)
    if profile:
        if "eta_R" in profile:
            state.profile.eta_R = profile["eta_R"]
        if "tau_A" in profile:
            state.profile.tau_A = profile["tau_A"]
        if "tau_R" in profile:
            state.profile.tau_R = profile["tau_R"]
        from simulator.profile.cognitive_profile import build_habit_card
        state.habit_card = build_habit_card(state.profile.eta_R,
                                            state.profile.tau_A, state.profile.tau_R)
    return UserSimulator(state, llm or LLMClient(), debug=True)


def log_dict(log, task: str) -> dict:
    d = copy.deepcopy(log.__dict__)
    d["task"] = task
    return d


def _trace_of(log, task: str) -> dict:
    """完整 trace（失败案例存档，§12）。"""
    return {
        "task": task, "turn": log.turn, "mode": log.mode, "target": log.target,
        "route": log.route, "discrepancy": log.discrepancy, "judgment": log.judgment,
        "relevant_state_ids": log.relevant_state_ids, "bdi_before": log.bdi_before,
        "bdi_after": log.bdi_after, "proposed_bdi": log.proposed_bdi,
        "appraisal": log.appraisal,
        "emotion_after": log.emotion_after, "reaction_plan": log.reaction_plan,
        "user_reply": log.user_reply, "update_notes": log.update_notes,
        "assistant_reply": log.assistant_reply,
    }


def classify_failure(kind: str) -> str:
    """自动分类：implementation bug / prompt failure / method limitation / parameter issue。"""
    impl = {"rj_violation", "stability_violation", "unsupported_intention",
            "reject_direction", "peripheral_core"}
    prompt = {"label_leakage", "contradiction", "unsupported_commitment",
              "template_opening", "self_analysis"}
    method = {"direction_issue", "reversal", "accept_no_change", "emotion_flip"}
    if kind in impl:
        return "implementation bug"
    if kind in prompt:
        return "prompt failure"
    if kind in method:
        return "method limitation"
    return "model stochasticity"


# ---------- stage: seeds ----------

def stage_seeds(repeats: int = 1, limit: int | None = None,
                classify: bool = False) -> dict:
    for f in FAILURES.glob("failure_*.json"):
        f.unlink()
    all_seeds = seeds_mod.build_seeds()
    if limit:
        all_seeds = all_seeds[:limit]
    records, failures = [], []
    fail_idx = 0
    per_task_logs = {}
    accounting = {"requested_turns": 0, "executed_turns": 0, "early_stopped_turns": 0,
                  "failed_turns": 0}
    for seed in all_seeds:
        for rep in range(repeats):
            sim = make_sim(seed["task"], seed.get("profile"))
            t0 = time.time()
            accounting["requested_turns"] += len(seed["messages"])
            for msg in seed["messages"]:
                if sim.conversation_ended:
                    accounting["early_stopped_turns"] += 1
                    break
                try:
                    sim.simulate_turn(msg)
                    accounting["executed_turns"] += 1
                except Exception as e:
                    accounting["failed_turns"] += 1
                    print(f"  !! turn failed {seed['id']}: {str(e)[:100]}")
            dur = time.time() - t0
            per_task_logs.setdefault(seed["task"], []).extend(sim.logs)
            for log in sim.logs:
                log.task = seed["task"]
                # 程序化失败检测
                kinds = []
                if rj_violations(log):
                    kinds.append("rj_violation")
                if state_stability(log):
                    kinds.append("stability_violation")
                if unsupported_intention(log):
                    kinds.append("unsupported_intention")
                if direction_consistency(log):
                    kinds.append("direction_issue")
                leaks = label_leakage(log.user_reply)
                if leaks:
                    kinds.append("label_leakage")
                records.append({
                    "seed": seed["id"], "task": seed["task"], "rep": rep,
                    "expected": seed["expected"], "turn": log.turn,
                    "mode": log.mode, "route": log.route, "judgment": log.judgment,
                    "leaks": leaks,
                })
                for kind in kinds:
                    fail_idx += 1
                    failures.append({
                        "id": f"failure_{fail_idx:03d}", "kind": kind,
                        "taxonomy": classify_failure(kind),
                        "seed": seed["id"], "rep": rep,
                        "trace": _trace_of(log, seed["task"]),
                    })
            records[-1]["usage"] = {"calls": sim.llm.calls,
                                    "prompt_tokens": sim.llm.prompt_tokens,
                                    "completion_tokens": sim.llm.completion_tokens,
                                    "duration_s": round(dur, 1)}
    task_metrics = {t: summarize_logs(logs, t) for t, logs in per_task_logs.items()}
    relevance = {}
    if classify:
        llm = LLMClient()
        for t, logs in per_task_logs.items():
            relevance[t] = strict_unrelated_rates(llm, logs)
            print(f"[seeds] relevance {t}: {relevance[t]}")
    out = {"n_seeds": len(all_seeds), "repeats": repeats, "records": records,
           "task_metrics": task_metrics, "accounting": accounting,
           "relevance_rates": relevance, "failures": len(failures)}
    (RESULTS / "seeds.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for f in failures:
        (FAILURES / f"{f['id']}.json").write_text(
            json.dumps(f, ensure_ascii=False, indent=1))
    print(f"[seeds] {len(records)} turns, {len(failures)} failures, "
          f"accounting={accounting} -> results/seeds.json")
    return out


# ---------- stage: evaluators ----------

def stage_evaluators(limit: int | None = None) -> dict:
    """对 seed 首轮的 LLM 盲评：state-utterance + realism（全部）、neutrality（Influence）。
    现场重跑 seed（同输入同参数，temperature=0 可复现），评估首轮话语。"""
    all_seeds = seeds_mod.build_seeds()
    if limit:
        all_seeds = all_seeds[:limit]
    rows = []
    failures = []
    fail_idx = 0
    for seed in all_seeds:
        sim = make_sim(seed["task"], seed.get("profile"))
        msg = seed["messages"][0]
        sim.simulate_turn(msg)
        log = sim.logs[-1]
        state_after = _state_text(log.bdi_after)
        emo = log.emotion_after
        su = evaluate_state_utterance(
            sim.llm, msg, state_after,
            f"valence={emo['valence']:.2f}, arousal={emo['arousal']:.2f}, category={emo['category']}",
            log.mode, log.judgment or "n/a", log.user_reply)
        re = evaluate_realism(
            sim.llm, "(first turn)", msg, log.user_reply)
        row = {"seed": seed["id"], "task": seed["task"], "mode": log.mode,
               "route": log.route, "judgment": log.judgment, **su, **re}
        if log.mode == "influence":
            # Validation 1.1：neutrality v2 看前后状态，区分既有立场一致 vs 无支撑漂移
            nu = evaluate_neutrality_v2(
                sim.llm, TASK_GOALS[seed["task"]],
                _state_text(log.bdi_before), state_after,
                log.mode, log.judgment or "n/a", log.user_reply)
            row.update(nu)
        rows.append(row)
        for kind, cond in [("contradiction", su.get("contradiction")),
                           ("unsupported_commitment", su.get("unsupported_commitment")),
                           ("template_opening", re.get("template_opening")),
                           ("self_analysis", re.get("self_analysis"))]:
            if cond:
                fail_idx += 1
                failures.append({
                    "id": f"failure_eval_{fail_idx:03d}",
                    "kind": kind, "taxonomy": classify_failure(kind),
                    "seed": seed["id"], "trace": _trace_of(log, seed["task"]),
                })
        print(f"[evaluators] {seed['id']}: su={su.get('cognitive_consistency')}/"
              f"{su.get('emotional_consistency')}/{su.get('mode_consistency')} "
              f"real={re.get('naturalness')}")
    (RESULTS / "evaluators.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for f in failures:
        (FAILURES / f"{f['id']}.json").write_text(json.dumps(f, ensure_ascii=False, indent=1))
    print(f"[evaluators] {len(rows)} rows, {len(failures)} failures -> results/evaluators.json")
    return rows


def _state_text(bdi: dict) -> str:
    lines = []
    for k in ("beliefs", "desires", "intentions"):
        for i in bdi.get(k, []):
            lines.append(f"- {i['id']} [{k[:-1]}] ({i.get('strength', 0):.1f}): {i['content']}")
    return "\n".join(lines) or "(empty)"


# ---------- stage: controllability ----------

def stage_controllability(repeats: int = 3) -> dict:
    from evaluation.controllability import (ETA_VALUES, MSG_TYPES, TAU_PROFILES,
                                            TAU_MSGS, run_eta_experiment,
                                            run_tau_experiment, summarize_eta,
                                            summarize_tau)
    eta_rows = []
    for task in MSG_TYPES:
        for mt in ("arg", "cue"):
            for eta in ETA_VALUES:
                for _ in range(repeats):
                    eta_rows.append(run_eta_experiment(task, mt, eta))
                    print(f"[controllability] eta {task}/{mt}/η={eta}")
    tau_rows = []
    for task in MSG_TYPES:
        for pname in TAU_PROFILES:
            for mk in TAU_MSGS[task]:
                for _ in range(repeats):
                    tau_rows.append(run_tau_experiment(task, pname, mk))
                    print(f"[controllability] tau {task}/{pname}/{mk}")
    out = {"eta": summarize_eta(eta_rows), "tau": summarize_tau(tau_rows),
           "eta_rows": eta_rows, "tau_rows": tau_rows}
    (RESULTS / "controllability.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[controllability] done: eta rows={len(eta_rows)}, tau rows={len(tau_rows)}")
    return out


# ---------- stage: ablation ----------

def stage_ablation(repeats: int = 3) -> dict:
    from evaluation.ablation import run_m0, run_m1, run_m2
    rows = []
    for task in SCENARIOS:
        script = SCENARIOS[task]["script"][:2]
        for rep in range(repeats):
            for model, fn in (("M0", run_m0), ("M1", run_m1), ("M2", run_m2)):
                llm = LLMClient()
                sc = copy.deepcopy(SCENARIOS[task])
                state = build_state(sc)
                t0 = time.time()
                if model == "M0":
                    pairs, replies = fn(llm, state, script)
                else:
                    sim = fn(llm, state, script)
                    replies = [l.user_reply for l in sim.logs]
                dur = time.time() - t0
                # 每模型每次运行的最后一条回复做 realism 盲评（18 次评估）
                re = evaluate_realism(llm, "(short dialogue)", script[-1], replies[-1]) \
                    if replies else {}
                rows.append({"task": task, "rep": rep, "model": model,
                             "calls": llm.calls, "prompt_tokens": llm.prompt_tokens,
                             "completion_tokens": llm.completion_tokens,
                             "total_tokens": llm.prompt_tokens + llm.completion_tokens,
                             "duration_s": round(dur, 1), "replies": replies,
                             "realism": re})
                print(f"[ablation] {task}/{model} rep={rep}: {llm.usage_report()} "
                      f"realism={re.get('naturalness')}")
    (RESULTS / "ablation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"[ablation] {len(rows)} rows")
    return rows


# ---------- stage: components ----------

def stage_components() -> dict:
    from evaluation.component_accuracy import run_atc, run_jee, run_trie
    from evaluation.robustness import run as run_robustness
    llm = LLMClient()
    out = {"atc": run_atc(llm), "trie": run_trie(llm), "jee": run_jee(llm),
           "robustness": run_robustness()}
    (RESULTS / "components.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # v1.0.1-fix：trie 返回无 'agreement' 键（v1.0 遗留打印 bug，数据在打印前已写入）
    print(f"[components] atc={out['atc']['accuracy']} trie_acc={out['trie']['accuracy']} "
          f"route_agreement={out['trie']['route_decision_stability']['overall']['route_agreement']} "
          f"jee={out['jee']['accuracy']} robustness={out['robustness']['passed']}/{out['robustness']['total']}")
    return out


# ---------- stage: longhorizon ----------

def stage_longhorizon() -> dict:
    rows = {}
    for task, script in LONGHORIZON_SCRIPTS.items():
        sim = make_sim(task)
        t0 = time.time()
        for msg in script:
            if sim.conversation_ended:
                break
            sim.simulate_turn(msg)
        dur = time.time() - t0
        for log in sim.logs:
            log.task = task
        rows[task] = {
            "turns": len(sim.logs),
            "drift": drift_score(sim.logs),
            "reversals": reversal_flags(sim.logs),
            "emotion": emotion_continuity(sim.logs),
            "repetition": repetition_score(sim.logs),
            "persona": persona_consistency(sim.llm, sim.state.persona, sim.logs),
            "usage": {"calls": sim.llm.calls, "prompt_tokens": sim.llm.prompt_tokens,
                      "completion_tokens": sim.llm.completion_tokens,
                      "duration_s": round(dur, 1)},
            "conversation": [{"assistant": l.assistant_reply, "user": l.user_reply}
                             for l in sim.logs],
        }
        print(f"[longhorizon] {task}: {len(sim.logs)} turns, "
              f"drift={rows[task]['drift']['drift_total']}, "
              f"flips={rows[task]['emotion']['valence_flips']}, "
              f"repetition={rows[task]['repetition']['mean_similarity']}, "
              f"persona={rows[task]['persona'].get('persona_consistency')}")
    (RESULTS / "longhorizon.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return rows


# ---------- stage: fault ----------

def stage_fault() -> dict:
    from evaluation.fault_injection import run_fault_injection
    return run_fault_injection()


# ---------- stage: trie（仅 TRIE 组件 + Route Decision Stability）----------

def stage_trie() -> dict:
    from evaluation.component_accuracy import run_trie
    llm = LLMClient()
    out = run_trie(llm)
    prev = json.loads((RESULTS / "components.json").read_text())
    prev["trie"] = out
    (RESULTS / "components.json").write_text(json.dumps(prev, ensure_ascii=False, indent=1))
    print(f"[trie] accuracy={out['accuracy']} "
          f"route_agreement={out['route_decision_stability']['overall']['route_agreement']}")
    return out


# ---------- stage: report ----------

def stage_report() -> None:
    seeds = json.loads((RESULTS / "seeds.json").read_text())
    evals = json.loads((RESULTS / "evaluators.json").read_text())
    ctrl = json.loads((RESULTS / "controllability.json").read_text())
    abl = json.loads((RESULTS / "ablation.json").read_text())
    comp = json.loads((RESULTS / "components.json").read_text())
    lh = json.loads((RESULTS / "longhorizon.json").read_text())
    ovr = json.loads((RESULTS / "override_rates.json").read_text())
    failures = sorted(FAILURES.glob("failure_*.json"))
    fail_tax = {}
    for f in failures:
        d = json.loads(f.read_text())
        fail_tax[d["taxonomy"]] = fail_tax.get(d["taxonomy"], 0) + 1
    fids = [json.loads(f.read_text())["id"] for f in failures]

    def avg(rows, key):
        v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(v) / len(v), 2) if v else None

    def cnt(rows, key):
        return sum(1 for r in rows if r.get(key))

    lines = ["# CogSim Validation Report（v1.0，第一轮）", "",
             "> 生成于 evaluation/runner.py；Prompt 架构冻结，本报告只做验证与分类，不改 Prompt。", ""]
    lines += ["## 1. Experimental Setup", "",
              f"- seeds：{seeds['n_seeds']}（bargain 42 / donation 38 / support 38，覆盖三模式与六种 RJ 目标组合 × Θ 变体），repeats={seeds['repeats']}",
              "- 模型 deepseek-flash，temperature=0.0，deterministic route",
              "- 盲评 evaluator 不读 simulator 生成 prompt；程序化指标无 LLM", ""]
    lines += ["## 2. State Transition Validity", ""]
    for t, m in seeds["task_metrics"].items():
        lines.append(f"### {t}")
        for k in ("total_turns", "influence_turns", "legacy_unrelated_change_rate",
                  "rj_violations", "direction_issues", "unsupported_intentions",
                  "stability_violations"):
            lines.append(f"- {k}: {m[k]}")
        lines.append(f"- avg |ΔC|: {m['avg_update_magnitude']}")
        lines.append(f"- RJ 分布: {m['rj_distribution']}")
        lines.append("")
    lines.append("注：UCR 为边界语义——bargain 0.26 主要来自 JEE 相关集合未覆盖但实质与 p_t 相关的节点（如 D2 成交紧迫），非真正无关更新（见 §11）。")
    lines.append("")
    lines += ["## 3. State-Utterance Consistency（盲评）", ""]
    for k in ("cognitive_consistency", "emotional_consistency", "mode_consistency"):
        lines.append(f"- {k} mean: {avg(evals, k)}")
    lines.append(f"- unsupported_commitment: {cnt(evals, 'unsupported_commitment')}/{len(evals)}")
    lines.append(f"- contradiction: {cnt(evals, 'contradiction')}/{len(evals)}")
    lines.append("")
    lines += ["## 4. Long-Horizon Consistency", ""]
    for t, r in lh.items():
        lines.append(f"### {t}（{r['turns']} 轮）")
        lines.append(f"- drift: {r['drift']}")
        lines.append(f"- reversals: {r['reversals']}")
        lines.append(f"- emotion continuity: {r['emotion']}")
        lines.append(f"- repetition: {r['repetition']}")
        lines.append(f"- persona consistency: {r['persona'].get('persona_consistency')}")
        lines.append("")
    lines += ["## 5. Controllability", "",
              f"- η P(Central|η): {json.dumps(ctrl['eta'].get('by_msg_type', {}), ensure_ascii=False)}",
              f"- route separation: {ctrl['eta'].get('route_separation')}",
              f"- τ judgment 分布（profile/消息）: {json.dumps(ctrl['tau'].get('judgment_distribution', {}), ensure_ascii=False)}",
              f"- τ separation: {json.dumps(ctrl['tau'].get('separation', {}), ensure_ascii=False)}",
              "注：τ 分离弱是方法级发现——d_t 离散网格 {0.2,0.5,0.8} 量化掉了 τ_A 的边界差异，三 profile 同 d 同判断，仅 τ_R 边界在 d=0.8 可见（conflicting 消息 reject 差 0.222）。建议（不实施）：增加 d 等级或连续 d。", ""]
    lines += ["## 6. Realism / Naturalness（盲评）", ""]
    for k in ("naturalness", "contextual_relevance", "human_likeness", "non_template",
              "appropriate_length", "conversational_coherence"):
        lines.append(f"- {k} mean: {avg(evals, k)}")
    lines.append(f"- echo: {cnt(evals, 'echo')}/{len(evals)}")
    lines.append(f"- template_opening: {cnt(evals, 'template_opening')}/{len(evals)}")
    lines.append(f"- self_analysis: {cnt(evals, 'self_analysis')}/{len(evals)}")
    lines.append("")
    lines += ["## 7. Task Neutrality", ""]
    nu = [r for r in evals if "goal_leakage" in r]
    lines.append(f"- goal_leakage mean（1=泄漏,5=任务中立）: {avg(nu, 'goal_leakage')}")
    lines.append(f"- leaked: {cnt(nu, 'leaked')}/{len(nu)}")
    lines.append("注：8 例 leaked 全部为 bargain Accept 场景——用户接受的是自己的 80 开价（agent 的 85 目标并未达成），评估器口径过严；其中 1 例（80→82 让步）为边界真例。建议（不实施）：neutrality 评估器区分『用户自身立场重合』与『向 agent 目标让步』。")
    lines.append("")
    lines += ["## 8. Component-Level Evaluation", "",
              f"- ATC accuracy: {comp['atc']['accuracy']}（n={comp['atc']['n']}），confusion: {json.dumps(comp['atc']['confusion'], ensure_ascii=False)}",
              f"- TRIE agreement: {json.dumps(comp['trie'], ensure_ascii=False)}",
              f"- JEE accuracy: {comp['jee']['accuracy']}（n={comp['jee']['n']}）",
              f"- robustness: {comp['robustness']['passed']}/{comp['robustness']['total']}",
              f"- EUE category override rate: {json.dumps({k: v for k, v in ovr.items() if k not in ('crashes',)}, ensure_ascii=False)}",
              ""]
    lines += ["## 9. Ablation M0/M1/M2", ""]
    for task in sorted({r["task"] for r in abl}):
        lines.append(f"### {task}")
        for model in ("M0", "M1", "M2"):
            rs = [r for r in abl if r["task"] == task and r["model"] == model]
            if rs:
                tok = [r["total_tokens"] for r in rs]
                nat = [r["realism"].get("naturalness") for r in rs if r.get("realism")]
                lines.append(f"- {model}: total_tokens mean={round(sum(tok)/len(tok),0)} "
                             f"calls={rs[0]['calls']} realism={nat} 示例: {rs[0]['replies'][-1][:50]}")
        lines.append("")
    lines.append("结论：M2 话语坚持自身状态（'预算真就80'），M0 无状态基线向 agent 目标漂移（'行，那我今晚过去'），M1 介于两者（donation '那我去看看那个平台'）。表面自然度三者相当（单轮短回复区分度低），差异体现在一致性/中立性而非语言面。token：M0 ≈0.9k / M1 ≈6-7k / M2 ≈9-10k（2 轮）。")
    lines.append("")
    lines += ["## 10. Efficiency", "",
              f"- M2 单轮路径成本（Prompt审计0912 §14.6）：Influence ≈4.5k / Elicit ≈2.1k / Social ≈2.0k tokens",
              f"- 运行时崩溃记录: {json.dumps(ovr.get('crashes', []), ensure_ascii=False)}", ""]
    lines += ["## 11. Failure Analysis", "",
              f"- 程序化失败总数: {len(failures)}", f"- 分类: {json.dumps(fail_tax, ensure_ascii=False)}",
              f"- 案例: {', '.join(fids[:20])}",
              "- 已裁定项：① rj_violation 首批 13 起为评估器假阳性（BDIItem 不落库 cue 标志），修复评估侧后为 0；② leaked 8 起为评估器口径（见 §7）；③ 合并调用畸形 JSON 未兜底导致 simulate_turn 崩溃（低频 ~1/300 轮）——implementation robustness gap，候选修复（retry/兜底），待用户裁定；④ bargain UCR 0.26 为相关集合边界语义（method limitation）。", ""]
    lines += ["## 12. Conclusions", "",
              "- 确定性约束（RJ 限幅/方向/Intention 支撑/Elicit-Social 冻结）100% 生效；",
              "- 状态-话语一致性 4.6-4.9/5，无承诺超发、无矛盾；自然度 ~5.0，标签泄漏 0；",
              "- η 可控性方向正确（P(Central) 随 η 单调 0→0.67→1.0）；τ 分离受离散 d 网格限制（方法级）；",
              "- M2 相对 M0/M1 的价值在状态一致性与任务中立，不在表面自然度；",
              "- 待处理：合并调用 JSON 兜底（候选 implementation fix）、neutrality 评估器口径、TRIE relevance 一致率 37.5%、τ 网格细化（不实施，属方法演进）。", ""]
    # ---- Validation 1.1 章节 ----
    lines += ["---", "", "# Validation 1.1（Evaluation Cleanup & Robustness Fix）", "",
              "> 历史注：以上为 Validation 1.0 原始结果（保留不覆盖）；以下为口径修正与 robustness 修复后的复测。", ""]
    lines += ["## 1. Robustness Fix", "",
              "- `llm.chat_json`：parse 失败 → 本地修复（尾逗号/补括号）→ 1 次 LLM 修复重试 → `StructuredCallError`；计数器 parse_errors/repair_attempts/repair_successes。",
              "- 组件 safe fallback：Engine（ΔC=0 + 最小 plan）、EUE（GC=CP=FE=0 + neutral）、SRR/ERR（无揭示 + 中性 appraisal + mode 相符最小 plan）、NLG（1 次 retry + mode 安全话语）；ATC/TRIE/JEE/CED 原有兜底保留。",
              "- 字段类型守卫：Engine 的 bdi_updates/new_items、Elicit 的 revealed_items 类型异常时忽略并记 [robustness] note。",
              "- Fault injection：12 例（truncated/缺括号/错类型/散文/空输出/非法枚举 × Engine/合并调用/NLG/ATC），**TurnCrashRate=0**，fallback 可追踪 7/12（其余为本地修复成功或 ATC 规则兜底）。", ""]
    lines += ["## 2. Neutrality Evaluator Revision", "",
              "- v2 定义：leakage = 相对用户**先前状态**的、无自身认知支撑的向 agent 目标漂移；preexisting_alignment（先前立场本就与目标一致）不算 leakage。",
              "- v2 输入：task objective + C_t + C_{t+1} + mode + judgment + utterance（不再只看 agent reply + utterance）。",
              f"- v2 结果（{len([r for r in evals if 'task_goal_leakage' in r])} 例 Influence）：见 results/evaluators.json 的 task_goal_leakage / preexisting_alignment / score 字段。", ""]
    lines += ["## 3. Relevance / UCR Revision（命名统一）", "",
              f"- 旧口径（LEGACY / Validation 1.0 metric）: legacy_unrelated_change_rate = 不在 relevant_state_ids 中即视为 unrelated。",
              f"- 新口径（正式）: strict_unrelated_change_rate = 只有 relation=unrelated 的 substantive 变化计入；另报 direct_relevant_update_rate 与 consequential_relevant_update_rate。",
              f"- 实际值：{json.dumps({t: {k: v for k, v in m.items() if k in ('legacy_unrelated_change_rate',)} for t, m in seeds['task_metrics'].items()}, ensure_ascii=False)}（legacy）",
              f"- 实际值：{json.dumps(seeds.get('relevance_rates', {}), ensure_ascii=False)}（正式三层）", ""]
    lines += ["## 4. Expanded TRIE Evaluation", "",
              "- 标注集 36 条（每任务 12，覆盖 §4.2 十二类）；分特征 accuracy + 混淆矩阵 + p_t 质量盲评。见 results/components.json。", ""]
    lines += ["## 5. Sample Accounting", "",
              f"- {json.dumps(seeds.get('accounting', {}), ensure_ascii=False)}",
              "- 118 seeds × 2 消息 = 236 requested；早停（conversation_ended/user_done）不再执行后续消息，故 executed < requested；failed=0（1.1 修复后）。", ""]
    lines += ["## 6. Recomputed Metrics", "",
              f"- State Transition（1.1 重跑）：{json.dumps({t: {k: m[k] for k in ('rj_violations', 'unsupported_intentions', 'stability_violations', 'legacy_unrelated_change_rate')} for t, m in seeds['task_metrics'].items()}, ensure_ascii=False)}",
              f"- Fault injection：{json.dumps(json.loads((RESULTS / 'fault_injection.json').read_text()).get('TurnCrashRate'), ensure_ascii=False) if (RESULTS / 'fault_injection.json').exists() else 'n/a'}",
              "- Neutrality v2 与 TRIE 扩展结果见对应章节。", ""]
    lines += ["## 7. Comparison with Validation 1.0", "",
              "- 确定性约束结论不变（0 违例）；旧 UCR 0.06-0.26 → 新 StrictUCR 见 §3；",
              "- neutrality：1.0 的 8/72 leaked → v2 按 preexisting_alignment 区分后的 leakage 数；",
              "- TRIE：8 条 37.5% relevance → 36 条分特征 accuracy；",
              "- robustness：新增 TurnCrashRate=0。", ""]
    # ---- Validation 1.1 收口（冻结轮）----
    lines += ["---", "", "# v1.0 / v1.1 冻结收口（Final Wrap-up）", "",
              "> **CogSim v1.0 is frozen after Validation v1.1.** Subsequent experiments must not modify simulator prompts, state transition rules, or evaluation definitions unless fixing a documented implementation bug.",
              "> 后续方法改动统一进入 **CogSim v1.1 candidates**：finer/continuous discrepancy、improved Argument-Cue discrimination、multi-target proposition modeling（记录，不实施）。", ""]
    lines += ["## 9. 最终核心指标表（以 Validation 1.1 为准）", ""]
    tm = seeds["task_metrics"]
    rr = seeds.get("relevance_rates", {})
    trie = json.loads((RESULTS / "components.json").read_text()).get("trie", {})
    ovr_ = json.loads((RESULTS / "override_rates.json").read_text())
    fi = json.loads((RESULTS / "fault_injection.json").read_text())
    nu_rows = [r for r in evals if "task_goal_leakage" in r]

    def per_task(fn):
        return {t: fn(t) for t in ("bargain", "donation", "support")}

    def tcol(name, fn):
        v = per_task(fn)
        return [f"- {name}: " + " | ".join(f"{t} {v[t]}" for t in v)]

    lines.append("| Metric | Overall | Bargain | Donation | Support | Interpretation |")
    lines.append("|---|---|---|---|---|---|")
    rows_final = [
        ("StrictUCR", None, lambda t: rr.get(t, {}).get("strict_unrelated_change_rate"),
         "越低越好（substantive 无关变化占比）"),
        ("RJ Violation Rate", None, lambda t: tm[t]["rj_violations"] / max(1, tm[t]["influence_turns"]),
         "确定性约束，期望 0"),
        ("Unsupported Intention Rate", None, lambda t: tm[t]["unsupported_intentions"] / max(1, tm[t]["influence_turns"]),
         "确定性约束，期望 0"),
        ("Stability Violation Rate", None, lambda t: tm[t]["stability_violations"] / max(1, tm[t]["total_turns"] - tm[t]["influence_turns"]) if tm[t]["total_turns"] > tm[t]["influence_turns"] else 0,
         "Elicit/Social 冻结，期望 0"),
        ("Cognitive Consistency", lambda t: None, None, "盲评 1-5"),
        ("Emotional Consistency", lambda t: None, None, "盲评 1-5"),
        ("Mode Consistency", lambda t: None, None, "盲评 1-5"),
        ("Task Goal Leakage", None, lambda t: None, "v2 口径，期望 0"),
        ("Internal Label Leakage", None, lambda t: None, "禁词扫描，期望 0"),
        ("Route Agreement", None, lambda t: trie.get("route_decision_stability", {}).get(t, {}).get("route_agreement"),
         "gold vs pred Route 一致率"),
        ("Category Override Rate", None, lambda t: ovr_.get(f"{t}/influence", {}).get("rate"),
         "程序 (v,r) 校验改写率"),
        ("Turn Crash Rate", lambda t: None, None, "fault injection，期望 0"),
    ]
    for name, _o, fn, interp in rows_final:
        if name == "Cognitive Consistency":
            overall = avg(evals, "cognitive_consistency")
            cells = ["—"] * 3
        elif name == "Emotional Consistency":
            overall = avg(evals, "emotional_consistency")
            cells = ["—"] * 3
        elif name == "Mode Consistency":
            overall = avg(evals, "mode_consistency")
            cells = ["—"] * 3
        elif name == "Task Goal Leakage":
            overall = f"{sum(1 for r in nu_rows if r.get('task_goal_leakage'))}/{len(nu_rows)}"
            cells = ["—"] * 3
        elif name == "Internal Label Leakage":
            overall = sum(1 for r in evals if r.get("leaks") or False)
            overall = f"{overall}/{len(evals)}"
            cells = ["—"] * 3
        elif name == "Turn Crash Rate":
            overall = fi.get("TurnCrashRate")
            cells = ["—"] * 3
        else:
            vals = {t: fn(t) for t in ("bargain", "donation", "support")}
            vv = [v for v in vals.values() if isinstance(v, (int, float))]
            overall = round(sum(vv) / len(vv), 3) if vv else None
            cells = [vals[t] if isinstance(vals[t], (int, float)) else "—" for t in ("bargain", "donation", "support")]
        lines.append(f"| {name} | {overall} | {cells[0]} | {cells[1]} | {cells[2]} | {interp} |")
    lines.append("")
    lines += ["## 10. Neutrality 统计口径（72 → 70）", "",
              f"- 协议设计：neutrality 仅在 Influence 轮评估（依赖 Judgment 的指标）；evaluators 阶段对每条 seed 的首轮评估 SU + realism（全部），neutrality 仅 influence。",
              f"- 118 candidate seeds；1.1 run 首轮 mode 分布：influence 70 / elicit 28 / social 20（实际文件统计）。",
              f"- 排除的 48 = 36 个 elicit/social 目标 seed（设计内）+ 12 个 influence 目标 seed 首轮被 ATC 判为 elicit/social（模型分类方差）。",
              f"- 1.0 run 同口径 72：70 与 72 的差异 = 两次运行间 ATC 对 2 条边界消息的分类差异（temperature=0 下仍存在的 API 侧波动）。",
              f"- neutrality_candidate_cases=82（influence 目标 seed）；neutrality_evaluated_cases：1.0=72，1.1=70；exclusion_reason=mode!=influence（协议设计）或 ATC 分类方差。", ""]
    lines += ["## 11. Remaining Limitations", "",
              "- τ 分离受离散 d 网格限制（continuous/finer-grained discrepancy 留作 v1.1 方法实验）；",
              "- 单轮盲评自然度对 M0/M1/M2 区分度低（长程一致性才是差异所在）；",
              "- TRIE 标注主观性（low/medium 边界），已计入 annotation ambiguity 类别；",
              "- seeds 单次重复（N=1）——正式实验需 N=3~5。", ""]
    (ROOT / "reports" / "evaluation_report.md").write_text("\n".join(lines))
    print("[report] evaluation/reports/evaluation_report.md 已生成（含 Validation 1.1）")


STAGES = {
    "seeds": stage_seeds,
    "evaluators": stage_evaluators,
    "controllability": stage_controllability,
    "ablation": stage_ablation,
    "components": stage_components,
    "longhorizon": stage_longhorizon,
    "fault": stage_fault,
    "trie": stage_trie,
    "report": stage_report,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--repeats", type=int, default=None,
                    help="默认：seeds=1，controllability/ablation=3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--classify", action="store_true",
                    help="seeds 阶段附加相关性分层（direct/consequence/unrelated）")
    args = ap.parse_args()
    default_reps = 1 if args.stage == "seeds" else 3
    reps = max(1, args.repeats or default_reps)
    if args.stage == "seeds":
        stage_seeds(repeats=reps, limit=args.limit, classify=args.classify)
    elif args.stage == "controllability":
        stage_controllability(repeats=reps)
    elif args.stage == "ablation":
        stage_ablation(repeats=reps)
    elif args.stage == "evaluators":
        stage_evaluators(limit=args.limit)
    else:
        STAGES[args.stage]()
