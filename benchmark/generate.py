"""
generate — SYNTHETIC, REALISTIC, MULTILINGUAL test sets, all as PDF (via PyMuPDF, HarfBuzz shaping).

- Made-up substances/values => no memorisation leakage; gold is exact by construction.
- Realistic: noise (prose that must NOT be extracted), complex values (RANGE/GMP/prohibited), notes.
- Multilingual: EN, Chinese, Thai (Thai digits ๐-๙), Hindi (Devanagari digits ०-९; Devanagari text
  extracts DEGRADED -> realistic "damaged extraction" case).
- Everything is rendered as **PDF** (HTML -> PyMuPDF Story) then read by Docling: a single realistic path.
Run: uv run python benchmark/generate.py
"""
from __future__ import annotations

import json
import logging
import pathlib

import fitz  # PyMuPDF

HERE = pathlib.Path(__file__).resolve().parent
log = logging.getLogger(__name__)

SUBS = ["Blorbic acid", "Zentanyl", "Quexol", "Fravestan", "Glivorate",
        "Pyrenthol", "Naxidine", "Vorbamate", "Cindralin", "Mophedone"]
CATS = ["Glimmer cakes", "Frosted noodles", "Spiced kelp drinks", "Dried moonfruit", "Cured zalmon",
        "Whipped voidcream", "Pickled starfruit", "Toasted nimbus grains", "Candied dusk pears", "Smoked riverbloom"]
FUNCS = ["preservative", "antioxidant", "stabiliser", "colour", "sweetener", "emulsifier"]

ZH = {
    "sub": dict(zip(SUBS, "布洛比酸 赞坦尼 奎索尔 弗拉维斯坦 格利沃酸盐 派伦托 那昔定 沃巴马特 辛德拉林 莫菲酮".split())),
    "cat": dict(zip(CATS, "微光蛋糕 霜花面条 香料海带饮料 月果干 腌扎鲑鱼 虚空奶油 腌渍星果 烤云粒谷物 蜜饯暮梨 烟熏河花".split())),
    "func": {"preservative": "防腐剂", "antioxidant": "抗氧化剂", "stabiliser": "稳定剂", "colour": "着色剂", "sweetener": "甜味剂", "emulsifier": "乳化剂"},
}
HI = {
    "sub": dict(zip(SUBS, "ब्लोर्बिक_अम्ल ज़ेंटानिल क्वेक्सोल फ्रावेस्टान ग्लिवोरेट पाइरेंथॉल नैक्सिडीन वोरबामेट सिंद्रालिन मोफेडोन".split())),
    "cat": dict(zip(CATS, "झिलमिल_केक तुषार_नूडल्स मसालेदार_केल्प_पेय सूखा_चंद्रफल नमकीन_ज़ालमन फेंटी_शून्यक्रीम अचारी_तारफल भुने_निंबस_अनाज मीठी_नाशपाती धुएँदार_नदीपुष्प".split())),
    "func": {"preservative": "परिरक्षक", "antioxidant": "प्रतिऑक्सीकारक", "stabiliser": "स्थायीकारक", "colour": "रंजक", "sweetener": "मधुरक", "emulsifier": "पायसीकारक"},
}
TH = {
    "sub": dict(zip(SUBS, "กรดบลอร์บิก เซนทานิล เควกซอล ฟราเวสแทน กลิโวเรต ไพเรนทอล แนกซิดีน วอร์บาเมต ซินดราลิน โมเฟโดน".split())),
    "cat": dict(zip(CATS, "เค้กกลิมเมอร์ บะหมี่เคลือบ เครื่องดื่มสาหร่าย ผลจันทร์อบแห้ง ปลาแซลมอนหมัก วอยด์ครีม มะเฟืองดอง ธัญพืชนิมบัส แพร์สนธยา ดอกแม่น้ำรมควัน".split())),
    "func": {"preservative": "สารกันเสีย", "antioxidant": "สารต้านอนุมูล", "stabiliser": "สารคงตัว", "colour": "สี", "sweetener": "สารให้ความหวาน", "emulsifier": "สารอิมัลซิไฟเออร์"},
}
EN = {"sub": {s: s for s in SUBS}, "cat": {c: c for c in CATS}, "func": {f: f for f in FUNCS}}
MAPS = {"en": EN, "zh": ZH, "hi": HI, "th": TH}

# Replace the '_' (separators used by .split()) with spaces
for _M in (ZH, HI, TH):
    for _k in ("sub", "cat"):
        _M[_k] = {e: v.replace("_", " ") for e, v in _M[_k].items()}

NOISE = {
    "en": {"principles": "General principles. Maximum levels are set so that total intake does not exceed the acceptable daily intake (ADI). Unless otherwise stated, levels apply to the food as consumed.",
           "method": "Methodology. Use levels were derived from a dietary exposure assessment; this section is explanatory and establishes no individual limit.",
           "defs": "Definitions. 'Maximum level' means the highest concentration of an additive permitted in a food.",
           "scope": "Scope. This Standard does not apply to foods for infants below 12 months of age."},
    "zh": {"principles": "总则。最大使用量的设定应使总摄入量不超过每日允许摄入量（ADI）。除另有规定外，限量以食用状态的食品计。",
           "method": "方法学。使用量基于膳食暴露评估得出；本节为说明性内容，不设定任何单项限量。",
           "defs": "定义。“最大使用量”指食品中允许的添加剂最高浓度。",
           "scope": "范围。本标准不适用于12月龄以下婴儿食品。"},
    "hi": {"principles": "सामान्य सिद्धांत। अधिकतम मात्राएँ इस प्रकार हैं कि कुल ग्रहण ADI से अधिक न हो। अन्यथा न कहा जाए तो मात्राएँ उपभोग की स्थिति पर लागू।",
           "method": "कार्यप्रणाली। उपयोग मात्राएँ आहार अनावरण आकलन से प्राप्त; यह अनुभाग व्याख्यात्मक है, कोई सीमा निर्धारित नहीं करता।",
           "defs": "परिभाषाएँ। 'अधिकतम मात्रा' का अर्थ खाद्य में अनुमत योज्य की उच्चतम सांद्रता।",
           "scope": "क्षेत्र। यह मानक 12 माह से कम आयु के शिशु खाद्य पर लागू नहीं होता।"},
    "th": {"principles": "หลักการทั่วไป ปริมาณสูงสุดกำหนดให้การได้รับรวมไม่เกินค่าที่ยอมรับได้ต่อวัน (ADI) เว้นแต่ระบุไว้เป็นอย่างอื่น ปริมาณใช้กับอาหารตามสภาพบริโภค",
           "method": "วิธีการ ปริมาณการใช้ได้มาจากการประเมินการได้รับจากอาหาร ส่วนนี้เป็นคำอธิบายและไม่กำหนดขีดจำกัดใด",
           "defs": "นิยาม 'ปริมาณสูงสุด' หมายถึงความเข้มข้นสูงสุดของวัตถุเจือปนที่อนุญาตในอาหาร",
           "scope": "ขอบเขต มาตรฐานนี้ไม่ใช้กับอาหารสำหรับทารกอายุต่ำกว่า 12 เดือน"},
}
LABELS = {
    "en": ("Food Additive Provisions — Testland (synthetic, realistic)", "Food Category", "Maximum Level", "Notes", "Notes"),
    "zh": ("食品添加剂使用规定 — 测试国（合成 · 真实风格）", "食品类别", "最大使用量", "备注", "注"),
    "hi": ("खाद्य योज्य प्रावधान — टेस्टलैंड (कृत्रिम, यथार्थ)", "खाद्य श्रेणी", "अधिकतम मात्रा", "टिप्पणी", "टिप्पणियाँ"),
    "th": ("ข้อกำหนดวัตถุเจือปนอาหาร — เทสต์แลนด์ (สังเคราะห์)", "ประเภทอาหาร", "ปริมาณสูงสุด", "หมายเหตุ", "หมายเหตุ"),
}
_GMP = {"en": "GMP", "zh": "按需适量", "hi": "आवश्यकतानुसार", "th": "ตามจำเป็น"}
_PROH = {"en": "Not permitted", "zh": "不得使用", "hi": "अनुमति नहीं", "th": "ไม่อนุญาต"}
_DIGITS = {"th": str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙"), "hi": str.maketrans("0123456789", "०१२३४५६७८९")}


def _loc(s: str, lang: str) -> str:
    """Localise the DIGITS (Thai/Devanagari); everything else unchanged. Gold stays in Arabic digits."""
    return s.translate(_DIGITS[lang]) if lang in _DIGITS else s


def _level(r: dict, lang: str = "en") -> str:
    t = r["rule_type"]
    if t == "GMP":
        return _GMP[lang]
    if t == "PROHIBITED":
        return _PROH[lang]
    if t == "RANGE":
        return _loc(f"{r['value_min']}–{r['value_max']} {r['unit']}", lang)
    v = r["value_max"]
    v = int(v) if isinstance(v, float) and v.is_integer() else v
    return _loc(f"{v} {r['unit']}", lang)


def _proj(substance, category, rule_type, value_min, value_max, unit) -> dict:
    return {"substance": substance, "category": category, "rule_type": rule_type,
            "value_min": value_min, "value_max": value_max, "unit": unit}


def _data(subs, n_cats, complex_values: bool):
    units = ["mg/kg", "mg/kg", "g/kg", "%"]
    base = []
    for i, sub in enumerate(subs):
        f = FUNCS[i % len(FUNCS)]
        for j in range(n_cats):
            cat = CATS[(i + j) % len(CATS)]
            key = i * 3 + j * 7
            if complex_values and key % 13 == 0:
                r = {"rule_type": "GMP", "value_min": None, "value_max": None, "unit": None}
            elif complex_values and key % 17 == 0:
                r = {"rule_type": "PROHIBITED", "value_min": None, "value_max": None, "unit": None}
            elif complex_values and key % 5 == 0:
                lo = 50 + 10 * (key % 9)
                r = {"rule_type": "RANGE", "value_min": lo, "value_max": lo + 100, "unit": "mg/kg"}
            else:
                u = units[(i + j) % len(units)]
                v = (50 + 20 * (key % 25) if u == "mg/kg" else round(0.5 + 0.1 * (key % 18), 1) if u == "g/kg" else round(0.02 + 0.01 * (key % 6), 2))
                r = {"rule_type": "MAX", "value_min": None, "value_max": v, "unit": u}
            base.append({"substance": sub, "category": cat, "function": f, "notes": [], **r})
    return base


def _add_notes(base):
    notes, exceptions = [], []
    mr = [b for b in base if b["rule_type"] in ("MAX", "RANGE")]
    step = max(1, len(mr) // 10)
    val_bases = mr[::step][:6]
    used = {id(b) for b in val_bases}
    for k, b in enumerate(val_bases):
        nid = str(468 + k)
        b["notes"].append(nid)
        ref = b["value_max"] or 100
        exc = round(ref / 2, 2) if (b["unit"] or "mg/kg") != "mg/kg" else (int(ref) // 2 or 10)
        notes.append({"id": nid, "kind": "value", "substance": b["substance"], "category": b["category"], "exc": exc, "unit": b["unit"] or "mg/kg"})
        exceptions.append({"substance": b["substance"], "category": b["category"], "exc": exc, "unit": b["unit"] or "mg/kg"})
    for k, b in enumerate([b for b in mr if id(b) not in used][:4]):
        nid = f"R{k + 1}"
        b["notes"].append(nid)
        notes.append({"id": nid, "kind": "restrict", "category": b["category"], "function": b["function"]})
    return notes, exceptions


def _scope(cat_en: str, lang: str) -> str:
    return {"en": f"{cat_en} intended for medical use", "zh": f"供医疗用途的{ZH['cat'][cat_en]}",
            "hi": f"चिकित्सा उपयोग हेतु {HI['cat'][cat_en]}", "th": f"{TH['cat'][cat_en]}สำหรับใช้ทางการแพทย์"}[lang]


def _note_text(n: dict, lang: str) -> str:
    m = MAPS[lang]
    if n["kind"] == "value":
        return {"en": f"Except for {_scope(n['category'], 'en')}, the maximum level of {n['substance']} is {n['exc']} {n['unit']}.",
                "zh": f"除{_scope(n['category'], 'zh')}外，{m['sub'][n['substance']]}的最大使用量为 {n['exc']} {n['unit']}。",
                "hi": f"{_scope(n['category'], 'hi')} को छोड़कर, {m['sub'][n['substance']]} की अधिकतम मात्रा {n['exc']} {n['unit']} है।",
                "th": f"ยกเว้น{_scope(n['category'], 'th')} ปริมาณสูงสุดของ{m['sub'][n['substance']]}คือ {n['exc']} {n['unit']}"}[lang]
    return {"en": f"For use in {n['category']} only as a {n['function']}; not permitted in other forms.",
            "zh": f"仅限在{m['cat'][n['category']]}中作为{m['func'][n['function']]}使用；其他形式不得使用。",
            "hi": f"केवल {m['cat'][n['category']]} में {m['func'][n['function']]} के रूप में; अन्य रूपों में अनुमति नहीं।",
            "th": f"ใช้ใน{m['cat'][n['category']]}เป็น{m['func'][n['function']]}เท่านั้น ไม่อนุญาตในรูปแบบอื่น"}[lang]


def _gold(base, exceptions, lang: str):
    m = MAPS[lang]
    g = [_proj(m["sub"][r["substance"]], m["cat"][r["category"]], r["rule_type"], r["value_min"], r["value_max"], r["unit"]) for r in base]
    g += [_proj(m["sub"][e["substance"]], _scope(e["category"], lang), "MAX", None, e["exc"], e["unit"]) for e in exceptions]
    return g


def _html(base, notes, lang: str, noise: bool) -> str:
    m, lab = MAPS[lang], LABELS[lang]
    parts = [f"<h1>{lab[0]}</h1>"]
    if noise:
        parts += [f"<p>{NOISE[lang][k]}</p>" for k in ("principles", "method", "defs")]
    for si, sub in enumerate(dict.fromkeys(r["substance"] for r in base)):
        rows = [r for r in base if r["substance"] == sub]
        parts.append(f"<h2>{m['sub'][sub]} — {m['func'][rows[0]['function']]}</h2>")
        parts.append(f"<table><tr><th>{lab[1]}</th><th>{lab[2]}</th><th>{lab[3]}</th></tr>")
        for r in rows:
            parts.append(f"<tr><td>{m['cat'][r['category']]}</td><td>{_level(r, lang)}</td><td>{', '.join(r['notes']) or '—'}</td></tr>")
        parts.append("</table>")
        if noise and si == 1:
            parts.append(f"<p>{NOISE[lang]['scope']}</p>")
    if notes:
        parts.append(f"<h2>{lab[4]}</h2><ul>")
        parts += [f"<li>{n['id']}: {_loc(_note_text(n, lang), lang)}</li>" for n in notes]
        parts.append("</ul>")
    return "<html><body>" + "".join(parts) + "</body></html>"


def _render_pdf(html: str, path: pathlib.Path, lang: str) -> None:
    # "figo" (FiraGO) is NOT installed (the pymupdf-fonts package is deliberately absent): MuPDF
    # silently falls back to its built-in fonts (Noto Serif Thai / Devanagari, Nimbus Sans).
    # This fallback is exactly what produces the degraded Hindi extraction tested by degraded_hindi —
    # installing pymupdf-fonts would change the regenerated PDFs and break comparability with past F1 scores.
    font = "china-s" if lang == "zh" else "figo"
    css = (f"* {{ font-family: {font}, sans-serif; font-size: 10px; }}"
           " table, th, td { border: 1px solid #777; border-collapse: collapse; padding: 3px; }"
           " h1 { font-size: 15px; } h2 { font-size: 12px; }")
    story = fitz.Story(html=html, user_css=css)
    writer = fitz.DocumentWriter(str(path))
    media = fitz.paper_rect("a4")
    rect = media + (36, 36, -36, -36)
    more = 1
    while more:
        dev = writer.begin_page(media)
        more, _ = story.place(rect)
        story.draw(dev)
        writer.end_page()
    writer.close()


def build() -> dict:
    simple = _data(SUBS[:4], 4, complex_values=False)
    simple[2].update(rule_type="GMP", value_max=None, unit=None)
    base = _data(SUBS, 5, complex_values=True)
    notes, exceptions = _add_notes(base)

    counts = {}
    _render_pdf(_html(simple, [], "en", noise=True), HERE / "inputs" / "noise_en.pdf", "en")
    (HERE / "gold" / "noise_en.json").write_text(json.dumps(_gold(simple, [], "en"), ensure_ascii=False, indent=2), encoding="utf-8")
    counts["noise_en"] = len(simple)
    for lang, fname in {"en": "complex_en", "th": "numerals_thai", "hi": "degraded_hindi"}.items():
        _render_pdf(_html(base, notes, lang, noise=True), HERE / "inputs" / f"{fname}.pdf", lang)
        (HERE / "gold" / f"{fname}.json").write_text(json.dumps(_gold(base, exceptions, lang), ensure_ascii=False, indent=2), encoding="utf-8")
        counts[lang] = len(base) + len(exceptions)
    return counts


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE.parent / "src"))
    import logs
    logs.setup()
    c = build()
    log.info("OK (PDF via PyMuPDF) : noise_en=%d complex_en=%d numerals_thai=%d degraded_hindi=%d",
             c["noise_en"], c["en"], c["th"], c["hi"])
