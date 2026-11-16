"""CLI(dry-run):预览建库材料的切块结果——各切几块、关键条款几块、触发哪些切块特性。
不写任何库。运行:PYTHONPATH=. uv run python scripts/show_kb.py"""
from app.kb import documents
from app.kb.sources import KB_DIR, SOURCE_TYPES


def main() -> None:
    print(f"{'文件':<26}{'类型':<8}{'块数':>4}{'关键条款':>6}  触发特性")
    total = key_total = 0
    for fname, ctype in SOURCE_TYPES.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        keys = sum(c.is_key_clause for c in chunks)
        total += len(chunks)
        key_total += keys
        feats = "、".join(documents.detect_features(md)) or "—"
        print(f"{fname:<26}{ctype:<8}{len(chunks):>4}{keys:>6}  {feats}")
    print(f"合计:{total} 块,关键条款 {key_total} 块(dry-run,未写库)")


if __name__ == "__main__":
    main()
