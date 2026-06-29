import re
import fitz  # PyMuPDF
import numpy as np
import cv2  # For image enhancement
import easyocr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import json
import os
from functools import lru_cache

# ==========================================
# DICTIONARY
# ==========================================
DICT_PATH = "dict.json"

DEFAULT_CUSTOM_TRANSLATIONS = {
    "Meradlá": "Measuring Instruments",
    "Náradie": "Tools",
    "Nástroje": "Tools",
    "Technologická návodka": "Technological Guide",
    "Sled operácií": "Sequence of Operations",
    "Hodnoty technolog. parametrov": "Technological Parameter Values",
    "Hodnoty technol. parametrov": "Technological Parameter Values",
    "Názov": "Name",
    "Typ": "Type",
    "Popis": "Description",
    "Operácia": "Operation",
    "POZNÁMKY": "NOTES",
    "Kontrolovať podľa plánu regulácie": "Check according to regulation plan",
    "Č.programu:": "Program No:",
    "Č.výkresu:": "Drawing No:",
    "HRANY NA HOTOVO": "FINISHED EDGES",
    "Kontrolovať parametre po MO": "Check Parameters After MO",
    "Kontrolovať parametre po TO": "Check Parameters After TO",
    "Merať hriadele": "Measure Shafts",
    "Umývať v ultrazvuku": "Ultrasonic Cleaning",
    "Prať pred TS": "Wash before TS",
    "Brúsiť povrch priebežne": "Grind surface continuously",
    "Stroj: Ručne": "Machine: Manual",
    "TEPELNE SPRACOVAŤ PODĽA": "HEAT TREAT ACCORDING TO",
    "VU TS indukčné kalenie": "VU TS induction hardening",
    "VU indukčné kalenie": "VU induction hardening",
    "Induktívne kaliť": "Inductively harden",
    "Hĺbka prekalenia": "Penetration depth",
    "meradlo priem. ob. dr.": "measuring gauge prim.ob.dr.",
    "odchylkomer": "deviation gauge",
    "posuvné meradlo": "sliding gauge",
    "PRAŤ PODĽA": "WASH ACCORDING TO",
    "SC- ŠPECIÁLNE CHARAKTERISTIKY PROCESU": "SC- SPECIAL PROCESS CHARACTERISTICS",
    "PRACOVAŤ V RUKAVICIACH": "WORK WITH GLOVES",
    "Brúsiť povrch na hotovo": "Grind surface to finish",
    "Brúsiť OD na hotovo": "Grind OD to finish",
    "Superfinišovať OD na hotovo": "Superfinish OD to finish",
    "Rozmerať hriadele": "Measure shafts",
    "Prať v ultrazvuk .Práčke": "Wash in ultrasonic washer",
    "stojan mikrometra": "micrometer stand",
    "Strmeňový mikrometer": "micrometer",
    "univerzál.merací stojan": "universal measuring stand",
    "brúsna lišta": "grinding bar",
    "Stredisko": "Center",
    "Pracovisko": "Workplace",
    "Indukčne kaliť": "Inductively harden",
    "Prietokomer": "Flow meter",
    "Tvrdomer": "Hardness tester",
    "Mikrotvrdomer": "Microhardness tester",
    "Teplomer": "Thermometer",
    "Stopky": "Stopwatch",
    "Refraktometer": "Refractometer",
    "Voltmeter": "Voltmeter",
    "Priehyb": "Deflection",
    "prizma": "Prism",
}

def load_custom_translations():
    try:
        if os.path.exists(DICT_PATH):
            with open(DICT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading dict: {e}")
    return DEFAULT_CUSTOM_TRANSLATIONS

CUSTOM_TRANSLATIONS = load_custom_translations()

# ==========================================
# PROTECT TECHNICAL VALUES
# ==========================================
TECH_PATTERN = re.compile(
    r'^([A-Z0-9\-–_]{4,})$|'
    r'^(Ø|R|M)?\d+([.,]\d+)?\s*(mm|μm|°|C|%)?$|'
    r'^\d+([.,]\d+)?\s*±\s*\d+([.,]\d+)?$|'
    r'^(Kc|Q|RH|T|Max\.)$'
)

# ==========================================
# M2M100 MODEL
# ==========================================
print("Loading M2M100 translation model...")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

print("Initializing EasyOCR...")
reader = easyocr.Reader(['sk', 'en'], gpu=False)

translation_cache = {}

# ==========================================
# TRANSLATION
# ==========================================
@lru_cache(maxsize=3000)
def translate_text(text: str) -> str:
    if not text or len(text.strip()) < 2:
        return text

    text_stripped = re.sub(r'\s+', ' ', text.strip())
    lower_text = text_stripped.lower()

    # Custom Dictionary
    for sk, en in CUSTOM_TRANSLATIONS.items():
        if sk.lower() in lower_text:
            result = re.sub(re.escape(sk), en, text_stripped, flags=re.IGNORECASE)
            print(f"[DICT] '{text_stripped}' → '{result}'")
            return result.upper() if text.isupper() else result

    if TECH_PATTERN.match(text_stripped):
        print(f"[TECH] Skipping: '{text_stripped}'")
        return text

    # M2M100 Translation
    try:
        tokenizer.src_lang = "sk"
        inputs = tokenizer(text_stripped, return_tensors="pt")
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
            max_length=256
        )
        translated = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"[M2M] '{text_stripped}' → '{translated}'")
        translation_cache[text_stripped] = translated
        return translated
    except Exception as e:
        print(f"[ERROR] Translation failed for '{text_stripped}': {e}")
        return text_stripped


# ==========================================
# FONT FITTING
# ==========================================
def fit_font_size(rect, text):
    if not text:
        return 10.5
    width = rect.width
    height = rect.height
    avg_char_width = 0.58
    size = min(width / (len(text) * avg_char_width), height * 0.88)
    return max(10.0, min(15.0, size))


# ==========================================
# OCR IMAGE ENHANCEMENT
# ==========================================
def enhance_for_ocr(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.equalizeHist(gray)
    enhanced = cv2.fastNlMeansDenoising(enhanced)
    return enhanced


# ==========================================
# MAIN PDF PROCESSING
# ==========================================
def process_industrial_pdf(input_path, output_path, progress_callback=None):
    print(f"[START] Processing: {input_path}")
    global CUSTOM_TRANSLATIONS
    CUSTOM_TRANSLATIONS = load_custom_translations()

    doc = fitz.open(input_path)
    total_pages = len(doc)

    for page_num in range(total_pages):
        if progress_callback:
            progress_callback(page_num + 1, total_pages, f"Processing page {page_num + 1}/{total_pages}")

        page = doc[page_num]
        processed_rects = []

        # ── 1. NATIVE TEXT ──────────────────────────────────────────
        blocks = page.get_text("dict", flags=fitz.TEXT_DEHYPHENATE)["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if len(text) < 2:
                        continue

                    bbox = fitz.Rect(span["bbox"])
                    if any(r.intersects(bbox) for r in processed_rects):
                        continue

                    translated = translate_text(text)
                    if translated == text:
                        continue

                    page.draw_rect(bbox, color=(1,1,1), fill=(1,1,1), overlay=True)
                    font_size = fit_font_size(bbox, translated)
                    page.insert_htmlbox(
                        bbox,
                        f'<p style="font-family:helv; font-size:{font_size}px; margin:0; line-height:1.08; text-align:center;">{translated}</p>'
                    )
                    processed_rects.append(bbox)

        # ── 2. OCR (for vector/image text not caught natively) ──────
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, -1))
        if img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        ocr_results = reader.readtext(
            img_np,
            paragraph=False,
            detail=1,
            width_ths=0.9,
            height_ths=0.9,
            text_threshold=0.25,
            low_text=0.25,
            link_threshold=0.3,
            canvas_size=3200,
            mag_ratio=2.0
        )

        scale = 3.0

        for (bbox_coords, word, prob) in ocr_results:
            if prob < 0.25:
                continue
            word = re.sub(r'\s+', ' ', word.strip())
            if len(word) < 3:
                continue

            ocr_rect = fitz.Rect(
                bbox_coords[0][0]/scale, bbox_coords[0][1]/scale,
                bbox_coords[2][0]/scale, bbox_coords[2][1]/scale
            )

            if any(r.intersects(ocr_rect) for r in processed_rects):
                continue

            translated = translate_text(word)
            if translated == word:
                continue

            print(f"[OCR→PDF] '{word}' → '{translated}' at {ocr_rect}")
            page.draw_rect(ocr_rect, color=(1,1,1), fill=(1,1,1), overlay=True)
            font_size = fit_font_size(ocr_rect, translated)
            page.insert_htmlbox(
                ocr_rect,
                f'<p style="font-family:helv; font-size:{font_size}px; margin:0; text-align:center; font-weight:bold;">{translated}</p>'
            )
            processed_rects.append(ocr_rect)

    # ── 3. SAVE ─────────────────────────────────────────────────────
    doc.save(output_path, garbage=4, deflate=True, clean=True)
    doc.close()
    print(f"✅ Done! Saved as: {output_path}")


if __name__ == "__main__":
    process_industrial_pdf("K031.1IH Shaft.pdf", "translated_output.pdf")
