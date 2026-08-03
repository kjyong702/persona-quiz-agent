"""promptfoo 실행 결과를 커밋 가능한 요약으로 줄인다.

    uv run python -m scripts.summarize_promptfoo eval/promptfoo/results-judge.json

promptfoo의 원본 출력은 요청과 응답을 전부 담아 7MB가 넘는다. 리포에 넣을 것은
그게 아니라 **버전별 성적과 어떤 케이스가 틀렸는가**다. 원본은 gitignore하고
이 요약만 커밋한다.

반복 실행(`--repeat`)한 결과를 주면 회차별 편차도 같이 낸다. `temperature=0`이어도
결과가 흔들리므로, 두 버전의 차이가 노이즈보다 큰지 판단하려면 이 값이 필요하다.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/promptfoo/results-judge.json")
    raw = json.loads(src.read_text(encoding="utf-8"))
    rows = raw["results"]["results"] if "results" in raw.get("results", {}) else raw["results"]

    # (프롬프트 버전, 케이스) -> 회차별 통과 여부
    runs: dict[tuple[str, str], list[bool]] = defaultdict(list)
    meta: dict[str, dict] = {}
    for row in rows:
        version = row.get("prompt", {}).get("label", "?")
        case = row["testCase"].get("description") or row["vars"]["answer"]
        runs[(version, case)].append(row["success"])
        # 판정 평가와 페르소나 평가는 vars 모양이 다르다. 공통으로 쓸 수 있게
        # 있는 것만 담고, 깨진 검사는 assert의 metric으로 식별한다
        v = row["vars"]
        broken = [
            c.get("assertion", {}).get("metric")
            for c in (row.get("gradingResult") or {}).get("componentResults", [])
            if not c.get("pass")
        ]
        meta.setdefault(case, {
            "subject": v.get("answer") or f"{v.get('persona_name','')}/{v.get('situation','')}",
            "category": v.get("category") or v.get("situation", ""),
            # 판정 평가는 기대 라벨이 곧 오류 종류(FA/FR)를 가른다. 없으면 빈 값
            "label": (row["testCase"].get("assert") or [{}])[0].get("value", ""),
            "broken": [],
        })
        meta[case]["broken"] = sorted({m for m in (meta[case]["broken"] + broken) if m})

    versions = sorted({v for v, _ in runs})
    cases = sorted({c for _, c in runs})
    summary: dict = {"cases": len(cases), "versions": {}}

    for version in versions:
        per_run = [
            sum(runs[(version, c)][i] for c in cases if i < len(runs[(version, c)]))
            for i in range(min(len(runs[(version, c)]) for c in cases))
        ]
        # 한 번이라도 틀린 것. 흔들리는 케이스는 flaky로 따로 표시한다
        failures = []
        for c in cases:
            results = runs[(version, c)]
            if all(results):
                continue
            m = meta[c]
            failures.append({
                **m,
                "kind": {"incorrect": "FA", "correct": "FR"}.get(m["label"], "규칙위반"),
                "flaky": len(set(results)) > 1,
                "passed": sum(results),
                "runs": len(results),
            })
        summary["versions"][version] = {
            "passed_per_run": per_run,
            "accuracy": round(sum(per_run) / len(per_run) / len(cases), 4) if per_run else None,
            "fa": sum(1 for f in failures if f["kind"] == "FA" and not f["flaky"]),
            "fr": sum(1 for f in failures if f["kind"] == "FR" and not f["flaky"]),
            "flaky": sum(1 for f in failures if f["flaky"]),
            "failures": sorted(failures, key=lambda f: (f["kind"], f["category"])),
        }

    out = src.with_name(src.stem.replace("results", "summary") + ".json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"케이스 {len(cases)}건")
    print(f"\n  {'버전':<12}{'회차별 통과':<22}{'정확도':>8}{'FA':>5}{'FR':>5}{'흔들림':>7}")
    print("  " + "-" * 60)
    for version, s in summary["versions"].items():
        print(f"  {version:<12}{str(s['passed_per_run']):<22}{s['accuracy']:>8.1%}"
              f"{s['fa']:>5}{s['fr']:>5}{s['flaky']:>7}")
    print(f"\n{out} 에 요약을 썼다")


if __name__ == "__main__":
    main()
