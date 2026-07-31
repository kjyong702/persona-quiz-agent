"""측정값에서 임계값을 정한다. 호출 없이 `eval/measurements.json`만 읽는다.

    uv run python -m scripts.analyze
    uv run python -m scripts.analyze --sweep

이 스크립트가 답해야 하는 질문은 넷이다.

1. 세 갈래 분기가 **실제로 세 갈래인가.** 분포가 한쪽에 몰려 있으면 전부 LLM으로
   가거나 전부 임베딩으로 확정되고, 그러면 하이브리드를 만든 이유가 사라진다
2. 임베딩이 **혼자 확정한 판정이 맞았는가.** 여기서 틀리면 LLM이 고칠 기회조차 없다
3. **margin 조건이 발화하는가.** 한 번도 안 걸리면 그 장치는 없는 것과 같다
4. 임계값을 **어디로 옮겨야 하는가.** 정확도와 LLM 호출 비율의 교환이다

세 갈래 판정 규칙은 `app/services/judge_service.py`와 같아야 한다.
여기 로직을 고치면 저쪽도 같이 고쳐야 한다.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from app.core.config import settings

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "eval" / "measurements.json"

EMB_CORRECT = "embedding_correct"
EMB_INCORRECT = "embedding_incorrect"
LLM = "llm"


def route(row: dict[str, Any], upper: float, lower: float, margin: float) -> str:
    """judge_service._is_confident_correct와 같은 규칙."""
    sim = row["similarity"]
    rival = row["rival_similarity"]
    if sim >= upper and (rival is None or (sim - rival) >= margin):
        return EMB_CORRECT
    if sim <= lower:
        return EMB_INCORRECT
    return LLM


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _fmt(v: float) -> str:
    return "  n/a" if v != v else f"{v:.3f}"


def distribution(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("1. 카테고리별 유사도 분포")
    print("=" * 78)
    print(f"{'카테고리':<22}{'n':>4}{'라벨':>6}{'최소':>8}{'25%':>8}{'중앙':>8}{'75%':>8}{'최대':>8}")
    print("-" * 78)
    cats: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    for cat in sorted(cats):
        group = cats[cat]
        sims = [g["similarity"] for g in group]
        labels = {g["label"] for g in group}
        tag = "정답" if labels == {"correct"} else ("오답" if labels == {"incorrect"} else "혼합")
        print(
            f"{cat:<22}{len(group):>4}{tag:>6}"
            f"{_fmt(min(sims)):>8}{_fmt(_pct(sims,0.25)):>8}{_fmt(_pct(sims,0.5)):>8}"
            f"{_fmt(_pct(sims,0.75)):>8}{_fmt(max(sims)):>8}"
        )

    print("\n라벨별 겹침 구간 (여기가 붙어 있으면 어떤 임계값으로도 못 가른다)")
    cor = [r["similarity"] for r in rows if r["label"] == "correct"]
    inc = [r["similarity"] for r in rows if r["label"] == "incorrect"]
    print(f"  정답 {len(cor):>4}건: 최소 {_fmt(min(cor))}  5% {_fmt(_pct(cor,0.05))}  중앙 {_fmt(_pct(cor,0.5))}")
    print(f"  오답 {len(inc):>4}건: 중앙 {_fmt(_pct(inc,0.5))}  95% {_fmt(_pct(inc,0.95))}  최대 {_fmt(max(inc))}")
    overlap = max(inc) - min(cor)
    if overlap > 0:
        print(f"  -> 겹침 폭 {overlap:.3f}. 이 구간의 항목은 임베딩만으로 못 가르므로 LLM이 봐야 한다")
    else:
        print("  -> 겹치지 않는다. 단일 임계값으로 완전 분리가 가능하다")


def margin_check(rows: list[dict[str, Any]], upper: float, margin: float) -> None:
    print("\n" + "=" * 78)
    print("2. margin 조건이 발화하는가")
    print("=" * 78)
    over = [r for r in rows if r["similarity"] >= upper]
    blocked = [r for r in over if r["margin"] is not None and r["margin"] < margin]
    print(f"  상한({upper}) 통과: {len(over)}건")
    print(f"  그중 margin({margin}) 미달로 LLM행: {len(blocked)}건")
    if not over:
        print("  ⚠️  상한을 넘긴 항목이 하나도 없다. 상한이 너무 높다")
    elif not blocked:
        print("  ⚠️  margin이 한 번도 발화하지 않았다. 이 장치는 지금 없는 것과 같다")
    else:
        saved = [b for b in blocked if b["label"] == "incorrect"]
        print(f"  그중 실제 오답: {len(saved)}건  <- margin이 실제로 막아낸 오통과")
        for b in blocked[:8]:
            mark = "막음" if b["label"] == "incorrect" else "헛수고"
            print(f"    [{mark}] s={b['similarity']:.3f} r={b['rival_similarity']:.3f} "
                  f"m={b['margin']:.3f}  {b['answer'][:34]}")


def confusion(rows: list[dict[str, Any]], upper: float, lower: float, margin: float) -> dict[str, int]:
    print("\n" + "=" * 78)
    print(f"3. 현재 임계값 성적  (upper={upper}, lower={lower}, margin={margin})")
    print("=" * 78)
    c = {"fa": 0, "fr": 0, "ok": 0, "llm": 0}
    fa_rows, fr_rows = [], []
    for r in rows:
        rt = route(r, upper, lower, margin)
        if rt == LLM:
            c["llm"] += 1
        elif rt == EMB_CORRECT:
            if r["label"] == "correct":
                c["ok"] += 1
            else:
                c["fa"] += 1
                fa_rows.append(r)
        else:
            if r["label"] == "incorrect":
                c["ok"] += 1
            else:
                c["fr"] += 1
                fr_rows.append(r)

    n = len(rows)
    decided = n - c["llm"]
    print(f"  임베딩 확정 {decided}건 ({decided/n:.1%})  ·  LLM 위임 {c['llm']}건 ({c['llm']/n:.1%})")
    print(f"  임베딩 확정 중 맞음 {c['ok']}건")
    print(f"  ❌ 오답을 정답으로 확정 (false accept): {c['fa']}건   <- 가장 나쁘다. LLM이 고칠 기회가 없다")
    print(f"  ❌ 정답을 오답으로 확정 (false reject): {c['fr']}건")
    for title, group in (("false accept", fa_rows), ("false reject", fr_rows)):
        for r in group[:6]:
            print(f"    [{title}] s={r['similarity']:.3f} {r['category']:<18}{r['answer'][:32]}")

    banded = [r for r in rows if r.get("expected_band") in ("embedding_confident", "llm_required")]
    if banded:
        hit = sum(
            1
            for r in banded
            if (route(r, upper, lower, margin) == LLM) == (r["expected_band"] == "llm_required")
        )
        print(f"\n  위임 판단 정확도: {hit}/{len(banded)} ({hit/len(banded):.1%})")
        print("  (설계 목표와 실제 경로가 일치한 비율. 정확도와 별개로 '누구에게 맡겼는가'가 옳았는지)")
    return c


def sweep(rows: list[dict[str, Any]], margin: float) -> None:
    print("\n" + "=" * 78)
    print("4. 임계값 스윕 — false accept 0건을 지키는 가장 싼 조합")
    print("=" * 78)
    uppers = [round(0.60 + 0.02 * i, 2) for i in range(20)]
    lowers = [round(0.20 + 0.02 * i, 2) for i in range(26)]

    grid = []
    for u in uppers:
        for lo in lowers:
            if lo >= u:
                continue
            fa = fr = ok = llm = 0
            for r in rows:
                rt = route(r, u, lo, margin)
                if rt == LLM:
                    llm += 1
                elif rt == EMB_CORRECT:
                    ok, fa = (ok + 1, fa) if r["label"] == "correct" else (ok, fa + 1)
                else:
                    ok, fr = (ok + 1, fr) if r["label"] == "incorrect" else (ok, fr + 1)
            grid.append({"u": u, "lo": lo, "fa": fa, "fr": fr, "ok": ok, "llm": llm})

    n = len(rows)
    # false accept가 없는 조합 중 LLM을 가장 적게 부르는 것이 우리가 찾는 값이다.
    # 오답을 정답으로 확정하는 것은 LLM 비용으로 살 수 있는 종류의 손해가 아니다
    clean = [g for g in grid if g["fa"] == 0]
    print(f"{'upper':>7}{'lower':>7}{'LLM위임':>9}{'FA':>5}{'FR':>5}{'확정정확':>9}")
    print("-" * 78)
    if not clean:
        print("  false accept 0건인 조합이 없다. 임계값으로는 못 막는다는 뜻이고,")
        print("  정규화나 앵커 구성 또는 라우팅 규칙 자체를 손봐야 한다")
        best_fa = min(grid, key=lambda g: (g["fa"], g["llm"]))
        print(f"  최선: upper={best_fa['u']} lower={best_fa['lo']} FA={best_fa['fa']}건")
    else:
        for g in sorted(clean, key=lambda g: (g["llm"], g["fr"]))[:12]:
            dec = n - g["llm"]
            acc = g["ok"] / dec if dec else 0
            print(f"{g['u']:>7}{g['lo']:>7}{g['llm']/n:>8.1%}{g['fa']:>5}{g['fr']:>5}{acc:>8.1%}")
        print("  (LLM 위임이 적은 순. 위로 갈수록 싸지만 오류를 사는 것일 수 있다)")

        # **정확도를 비용보다 먼저 본다.** LLM 위임을 먼저 최소화하도록 고르면
        # false reject를 비용과 맞바꾸게 된다. 판정 시스템에서 그 교환은 성립하지 않는다.
        # 확정 오류가 0인 지점이 있으면 그중 가장 싼 것, 없으면 오류가 가장 적은 것
        perfect = [g for g in clean if g["fr"] == 0]
        pool = perfect or clean
        best = min(pool, key=lambda g: (g["fr"], g["llm"]) if not perfect else (g["llm"],))
        note = "확정 오류 0건 중 가장 싼 조합" if perfect else "오류 0건 조합이 없어 오류 최소를 고름"
        print(f"\n  추천: upper={best['u']}, lower={best['lo']}   ({note})")
        print(f"        FA {best['fa']}건, FR {best['fr']}건, LLM 위임 {best['llm']/n:.1%}")
        cheapest = min(clean, key=lambda g: g["llm"])
        if cheapest["llm"] < best["llm"]:
            gap = (best["llm"] - cheapest["llm"]) / n
            print(f"  참고: 가장 싼 것은 upper={cheapest['u']}, lower={cheapest['lo']} "
                  f"(LLM {cheapest['llm']/n:.1%})인데 FR {cheapest['fr']}건을 낸다.")
            print(f"        오류 {cheapest['fr']}건을 없애는 값이 LLM 위임 {gap:.1%}p다. "
                  f"이 교환을 받을지는 사람이 정한다")
        cur = next(
            (g for g in grid
             if g["u"] == settings.upper_threshold and g["lo"] == settings.lower_threshold),
            None,
        )
        if cur:
            print(f"  현재값: upper={cur['u']}, lower={cur['lo']}  "
                  f"FA {cur['fa']}건, FR {cur['fr']}건, LLM 위임 {cur['llm']/n:.1%}")
    print("\n  ⚠️  이 추천은 이 데이터셋에 대한 최적값이다. 같은 데이터로 고르고 같은 데이터로")
    print("      성능을 보고하면 과적합이다. 확정 전에 데이터를 나눠 검증할 것")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upper", type=float, default=settings.upper_threshold)
    ap.add_argument("--lower", type=float, default=settings.lower_threshold)
    ap.add_argument("--margin", type=float, default=settings.min_margin)
    ap.add_argument("--sweep", action="store_true", help="임계값 격자 탐색")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"측정값이 없습니다: {SRC}\n먼저 `uv run python -m scripts.measure`")
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    # 정규화 후 빈 문자열이 된 항목은 임베딩 경로에 진입조차 못 한다.
    # 유사도가 없으므로 임계값 계산에서 빼고 따로 센다. 섞으면 분포가 왜곡된다
    rows = [r for r in raw if not r.get("rendered_empty")]
    empty = [r for r in raw if r.get("rendered_empty")]
    print(f"측정값 {len(raw)}건  (모델 {raw[0]['embedding_model']}, 템플릿 {raw[0]['template_version']})")
    if empty:
        print(f"  그중 {len(empty)}건은 정규화 후 빈 문자열이라 임베딩 경로 진입 불가 "
              f"-> 무조건 LLM 폴백. 임계값 계산에서 제외한다")
        print(f"  ({', '.join(repr(e['answer']) for e in empty[:6])})")

    distribution(rows)
    margin_check(rows, args.upper, args.margin)
    confusion(rows, args.upper, args.lower, args.margin)
    if args.sweep:
        sweep(rows, args.margin)
    else:
        print("\n임계값 후보를 보려면: uv run python -m scripts.analyze --sweep")


if __name__ == "__main__":
    main()
