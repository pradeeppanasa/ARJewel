"""
Generate varied placeholder jewellery PNGs for the gallery.
Run: python assets/generate_placeholders.py
"""
import math
from pathlib import Path
from PIL import Image, ImageDraw

EARRINGS_DIR  = Path(__file__).parent / "jewellery" / "earrings"
NECKLACES_DIR = Path(__file__).parent / "jewellery" / "necklaces"
EARRINGS_DIR.mkdir(parents=True, exist_ok=True)
NECKLACES_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palettes ────────────────────────────────────────────────────────────
GOLD       = (212, 175, 55,  230)
SILVER     = (192, 192, 192, 230)
ROSE_GOLD  = (183, 110, 121, 230)
PEARL      = (234, 224, 200, 230)
RUBY       = (155,  17,  30, 230)
SAPPHIRE   = ( 15,  82, 186, 230)
EMERALD    = (  0, 130,  80, 230)
DIAMOND    = (185, 242, 255, 200)


def new(w, h):
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))


# ══════════════════════════════════════════════════════════════════════════════
#  Earring generators
# ══════════════════════════════════════════════════════════════════════════════

def ear_drop(path, metal, gem):
    img, d = new(80, 150), None
    d = ImageDraw.Draw(img)
    cx = 40
    d.arc([cx-10, 2, cx+10, 24], 0, 180, fill=metal, width=4)
    d.ellipse([cx-18, 26, cx+18, 80], fill=metal)
    d.ellipse([cx-8, 38, cx+8, 58], fill=gem)
    d.ellipse([cx-8, 88, cx+8, 112], fill=metal)
    d.ellipse([cx-4, 118, cx+4, 130], fill=gem)
    img.save(path); print(f"  {path.name}")


def ear_hoop(path, metal):
    img = new(90, 90)
    d   = ImageDraw.Draw(img)
    d.ellipse([5, 5, 85, 85], outline=metal, width=8)
    d.ellipse([30, 30, 60, 60], fill=metal)
    img.save(path); print(f"  {path.name}")


def ear_stud(path, gem, metal):
    img = new(60, 60)
    d   = ImageDraw.Draw(img)
    d.ellipse([5, 5, 55, 55], fill=gem)
    d.ellipse([18, 18, 42, 42], fill=metal)
    d.ellipse([26, 26, 34, 34], fill=DIAMOND)
    img.save(path); print(f"  {path.name}")


def ear_chandelier(path, metal, gem):
    img = new(100, 170)
    d   = ImageDraw.Draw(img)
    cx  = 50
    d.arc([cx-12, 2, cx+12, 28], 0, 180, fill=metal, width=4)
    # top bar
    d.rectangle([cx-28, 30, cx+28, 40], fill=metal)
    # three drops
    for ox in [-24, 0, 24]:
        d.line([(cx+ox, 40), (cx+ox, 90)], fill=metal, width=3)
        d.ellipse([cx+ox-10, 88, cx+ox+10, 110], fill=gem)
        if ox == 0:
            d.line([(cx, 110), (cx, 150)], fill=metal, width=2)
            d.ellipse([cx-8, 148, cx+8, 164], fill=metal)
    img.save(path); print(f"  {path.name}")


def ear_teardrop(path, metal, gem):
    img = new(70, 130)
    d   = ImageDraw.Draw(img)
    cx  = 35
    d.arc([cx-10, 2, cx+10, 24], 0, 180, fill=metal, width=4)
    # teardrop polygon
    pts = [(cx, 30), (cx-20, 70), (cx-20, 100), (cx, 118), (cx+20, 100), (cx+20, 70)]
    d.polygon(pts, fill=gem, outline=metal)
    d.ellipse([cx-8, 50, cx+8, 68], fill=DIAMOND)
    img.save(path); print(f"  {path.name}")


def ear_tassel(path, metal):
    img = new(60, 160)
    d   = ImageDraw.Draw(img)
    cx  = 30
    d.arc([cx-10, 2, cx+10, 24], 0, 180, fill=metal, width=4)
    d.ellipse([cx-10, 24, cx+10, 44], fill=metal)
    for ox in range(-20, 25, 8):
        d.line([(cx+ox, 44), (cx+ox, 140 + (ox % 3)*8)], fill=metal, width=2)
    img.save(path); print(f"  {path.name}")


def ear_cluster(path, metal, gems):
    img = new(80, 100)
    d   = ImageDraw.Draw(img)
    cx  = 40
    d.arc([cx-10, 2, cx+10, 24], 0, 180, fill=metal, width=4)
    positions = [(cx, 45), (cx-18, 58), (cx+18, 58), (cx-10, 75), (cx+10, 75), (cx, 90)]
    for i, (x, y) in enumerate(positions):
        gem = gems[i % len(gems)]
        d.ellipse([x-10, y-10, x+10, y+10], fill=gem, outline=metal)
    img.save(path); print(f"  {path.name}")


def ear_bar(path, metal, gem):
    img = new(80, 90)
    d   = ImageDraw.Draw(img)
    cx  = 40
    d.arc([cx-10, 2, cx+10, 24], 0, 180, fill=metal, width=4)
    d.rectangle([cx-30, 28, cx+30, 44], fill=metal)
    for x in range(cx-22, cx+28, 14):
        d.ellipse([x-5, 56, x+5, 76], fill=gem)
    img.save(path); print(f"  {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  Necklace generators
# ══════════════════════════════════════════════════════════════════════════════

def neck_pendant(path, metal, gem):
    img = new(400, 180)
    d   = ImageDraw.Draw(img)
    cx  = 200
    d.arc([30, -60, 370, 140], 10, 170, fill=metal, width=5)
    d.ellipse([cx-25, 110, cx+25, 160], fill=metal)
    d.ellipse([cx-14, 120, cx+14, 148], fill=gem)
    d.ellipse([cx-6, 128, cx+6, 140], fill=DIAMOND)
    img.save(path); print(f"  {path.name}")


def neck_choker(path, metal, gem):
    img = new(380, 80)
    d   = ImageDraw.Draw(img)
    d.arc([20, -60, 360, 100], 8, 172, fill=metal, width=9)
    for x in range(80, 302, 30):
        d.ellipse([x-7, 22, x+7, 36], fill=gem, outline=metal)
    img.save(path); print(f"  {path.name}")


def neck_chain_pendant(path, metal, gem):
    img = new(400, 200)
    d   = ImageDraw.Draw(img)
    cx  = 200
    for i in range(20, 380, 18):
        d.ellipse([i, 20+int(30*math.sin(i/40)), i+12, 32+int(30*math.sin(i/40))],
                  outline=metal, width=2)
    d.polygon([(cx, 100), (cx-18, 140), (cx, 170), (cx+18, 140)], fill=gem, outline=metal)
    img.save(path); print(f"  {path.name}")


def neck_pearl(path, pearl, metal):
    img = new(400, 100)
    d   = ImageDraw.Draw(img)
    for x in range(20, 380, 24):
        y = 50 + int(20 * math.sin(x / 35))
        d.ellipse([x-10, y-10, x+10, y+10], fill=pearl, outline=metal)
    img.save(path); print(f"  {path.name}")


def neck_statement(path, metal, gems):
    img = new(400, 200)
    d   = ImageDraw.Draw(img)
    cx  = 200
    d.arc([40, -40, 360, 120], 10, 170, fill=metal, width=5)
    sizes = [18, 22, 26, 30, 26, 22, 18]
    xs    = [100, 130, 162, 200, 238, 270, 300]
    for x, s, gem in zip(xs, sizes, gems * 3):
        d.ellipse([x-s//2, 90, x+s//2, 90+s], fill=gem, outline=metal)
        d.ellipse([x-s//4, 98, x+s//4, 98+s//2], fill=DIAMOND)
    img.save(path); print(f"  {path.name}")


def neck_lariat(path, metal, gem):
    img = new(400, 220)
    d   = ImageDraw.Draw(img)
    cx  = 200
    d.arc([40, -40, 360, 120], 10, 170, fill=metal, width=4)
    d.line([(cx-15, 80), (cx-30, 170)], fill=metal, width=3)
    d.line([(cx+15, 80), (cx+30, 170)], fill=metal, width=3)
    d.ellipse([cx-40, 165, cx-16, 195], fill=gem, outline=metal)
    d.ellipse([cx+16, 165, cx+40, 195], fill=gem, outline=metal)
    img.save(path); print(f"  {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  Generate all assets
# ══════════════════════════════════════════════════════════════════════════════

print("Generating earrings…")
ear_drop      (EARRINGS_DIR/"gold_drop.png",         GOLD,     RUBY)
ear_drop      (EARRINGS_DIR/"silver_drop.png",       SILVER,   SAPPHIRE)
ear_hoop      (EARRINGS_DIR/"gold_hoop.png",         GOLD)
ear_hoop      (EARRINGS_DIR/"rose_gold_hoop.png",    ROSE_GOLD)
ear_stud      (EARRINGS_DIR/"diamond_stud.png",      DIAMOND,  SILVER)
ear_stud      (EARRINGS_DIR/"ruby_stud.png",         RUBY,     GOLD)
ear_chandelier(EARRINGS_DIR/"gold_chandelier.png",   GOLD,     EMERALD)
ear_chandelier(EARRINGS_DIR/"silver_chandelier.png", SILVER,   RUBY)
ear_teardrop  (EARRINGS_DIR/"emerald_teardrop.png",  GOLD,     EMERALD)
ear_teardrop  (EARRINGS_DIR/"sapphire_teardrop.png", SILVER,   SAPPHIRE)
ear_tassel    (EARRINGS_DIR/"gold_tassel.png",       GOLD)
ear_tassel    (EARRINGS_DIR/"silver_tassel.png",     SILVER)
ear_cluster   (EARRINGS_DIR/"gem_cluster.png",       GOLD,     [RUBY, SAPPHIRE, EMERALD, DIAMOND])
ear_bar       (EARRINGS_DIR/"diamond_bar.png",       SILVER,   DIAMOND)
ear_bar       (EARRINGS_DIR/"ruby_bar.png",          GOLD,     RUBY)
ear_drop      (EARRINGS_DIR/"pearl_drop.png",        ROSE_GOLD, PEARL)

print("\nGenerating necklaces…")
neck_pendant      (NECKLACES_DIR/"gold_pendant.png",    GOLD,    RUBY)
neck_pendant      (NECKLACES_DIR/"silver_pendant.png",  SILVER,  SAPPHIRE)
neck_choker       (NECKLACES_DIR/"gold_choker.png",     GOLD,    EMERALD)
neck_choker       (NECKLACES_DIR/"diamond_choker.png",  SILVER,  DIAMOND)
neck_chain_pendant(NECKLACES_DIR/"chain_ruby.png",      GOLD,    RUBY)
neck_chain_pendant(NECKLACES_DIR/"chain_sapphire.png",  SILVER,  SAPPHIRE)
neck_pearl        (NECKLACES_DIR/"pearl_strand.png",    PEARL,   GOLD)
neck_pearl        (NECKLACES_DIR/"pearl_silver.png",    PEARL,   SILVER)
neck_statement    (NECKLACES_DIR/"statement_gold.png",  GOLD,    [RUBY, EMERALD, SAPPHIRE, DIAMOND])
neck_statement    (NECKLACES_DIR/"statement_silver.png",SILVER,  [DIAMOND, SAPPHIRE, RUBY, EMERALD])
neck_lariat       (NECKLACES_DIR/"gold_lariat.png",     GOLD,    EMERALD)
neck_lariat       (NECKLACES_DIR/"rose_lariat.png",     ROSE_GOLD, RUBY)

print("\nDone.")
