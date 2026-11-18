"""建库材料清单的唯一定义处:CLI(build_kb/show_kb)与页面 API 同读这一份。"""
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "kb"

# 文件 → content_type
SOURCE_TYPES = {
    "product-faq.md": "faq",
    "returns-policy.md": "policy",
    "after-sales-manual.md": "manual",
    "product-specs.md": "spec",
}

# 录入允许的 content_type 白名单(mined 留给对话挖知识)
CONTENT_TYPES = {"faq", "policy", "manual", "spec", "mined"}
