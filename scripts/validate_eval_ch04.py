"""评估集自检(make eval-check):内存重建 data/kb 的块(app.kb 同款切分),
不碰 MySQL/Milvus/网络。守:id/query 唯一、五桶各 60、证据关键词命中真实小节、
expect_points 去空白后逐字命中目标小节正文、D 桶不带 ground truth、E 桶 2-3 组。
运行:PYTHONPATH=. uv run python scripts/validate_eval_ch04.py"""
import json
import pathlib
import sys
from collections import Counter

from app.kb import documents
from app.kb.sources import KB_DIR, SOURCE_TYPES

_ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL_FILE = _ROOT / "tests/data/eval_ch04.jsonl"
BUCKETS = ["A_policy", "B_model", "C_colloquial", "D_absent", "E_multi"]
PER_BUCKET = 60


def _norm(s):
    return "".join((s or "").split())   # 与 eval_ch04._norm 同口径:去全部空白


def build_chunks_in_memory():
    chunks = []
    for fname, ctype in SOURCE_TYPES.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        chunks.extend(documents.build_chunks(md, content_type=ctype))
    return chunks


def _matched_chunks(kws, chunks):
    """section_path 去空白后含任一关键词的块(与 _section_match 同口径)。"""
    nks = [_norm(kw) for kw in kws if _norm(kw)]
    return [c for c in chunks if any(k in _norm(c.section_path) for k in nks)]


def _chunk_text(c):
    # 证据文本 = category + questions + answer(dualwrite 同构)
    return _norm(f"{c.category}\n{c.questions}\n{c.answer}")


def validate(rows, chunks):
    errs = []
    seen_ids, seen_queries = {}, {}
    counts = Counter(r.get("bucket") for r in rows)
    for b in BUCKETS:
        if counts.get(b, 0) != PER_BUCKET:
            errs.append(f"[桶数] {b}: {counts.get(b, 0)} 题,应为 {PER_BUCKET}")
    for b in counts:
        if b not in BUCKETS:
            errs.append(f"[桶数] 未知桶 {b}")

    for i, r in enumerate(rows, 1):
        rid = r.get("id", f"line{i}")
        tag = f"[{rid}]"
        if rid in seen_ids:
            errs.append(f"{tag} id 重复(首见于第 {seen_ids[rid]} 行)")
        seen_ids.setdefault(rid, i)
        q = r.get("query", "")
        if not q.strip():
            errs.append(f"{tag} query 为空")
        if q in seen_queries:
            errs.append(f"{tag} query 重复(首见于 {seen_queries[q]})")
        seen_queries.setdefault(q, rid)

        bucket = r.get("bucket")
        section = r.get("expect_section") or []
        groups = r.get("expect_sections_all") or []
        points = r.get("expect_points") or []

        if bucket == "D_absent":
            if section or groups or points:
                errs.append(f"{tag} D 桶不得带 ground truth(expect_section/expect_sections_all/expect_points 须全空)")
            if r.get("should_refuse") is not True:
                errs.append(f"{tag} D 桶 should_refuse 须为 true")
            continue
        if r.get("should_refuse") is not False:
            errs.append(f"{tag} 非 D 桶 should_refuse 须为 false")

        if bucket == "E_multi":
            if section:
                errs.append(f"{tag} E 桶 expect_section 须为空(证据写在 expect_sections_all)")
            if not 2 <= len(groups) <= 3:
                errs.append(f"{tag} E 桶须 2-3 组证据,现 {len(groups)} 组")
            target = []
            for gi, g in enumerate(groups, 1):
                hit = _matched_chunks(g, chunks)
                if not hit:
                    errs.append(f"{tag} 第 {gi} 组别名 {g} 未命中任何小节 section_path")
                target.extend(hit)
        else:
            if not section:
                errs.append(f"{tag} expect_section 为空")
            for kw in section:
                if not _matched_chunks([kw], chunks):
                    errs.append(f"{tag} 关键词「{kw}」未命中任何小节 section_path")
            target = _matched_chunks(section, chunks)

        if target:
            union = "\n".join(_chunk_text(c) for c in target)
            for p in points:
                if _norm(p) not in union:
                    errs.append(f"{tag} 要点「{p}」未在目标小节正文中逐字(去空白)命中")
    return errs, counts


def main():
    rows = [json.loads(ln) for ln in EVAL_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    chunks = build_chunks_in_memory()
    errs, counts = validate(rows, chunks)
    if errs:
        print(f"❌ eval-check:{len(errs)} 处违规")
        for e in errs:
            print("  " + e)
        return 1
    n_points = sum(len(r.get("expect_points") or []) for r in rows)
    print(f"✅ eval-check 通过:{len(rows)} 题 | 内存重建 {len(chunks)} 块 | 校验要点 {n_points} 条")
    for b in BUCKETS:
        print(f"  {b}: {counts[b]} 题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
