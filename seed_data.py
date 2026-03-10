#!/usr/bin/env python3
"""Seed the Casadei app with MVP local images + Casadei.com product data.

Usage:
    python seed_data.py

This script is idempotent — it clears and rebuilds data/ on each run.

Product structure:
- A "product" is one shoe design (e.g. "Superblade Slingback")
- A "variation" is one material/color combo (e.g. Topaz, Emerald)
- Multiple views/angles of the same color are "results" within that variation
- Only MVP shoes have real hand-drawn sketches
"""

from __future__ import annotations

import json
import shutil
import uuid
import urllib.request
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
STORE_PATH = DATA_DIR / "store.json"

MVP_DIR = PROJECT_ROOT.parent / "casadei-front" / "mvp"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_id() -> str:
    return uuid.uuid4().hex[:12]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def download_image(url: str, dst: Path, retries: int = 3) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36",
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            })
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                data = resp.read()
                if len(data) < 1000:
                    print(f"  WARNING: tiny response ({len(data)}B) for {url}")
                    return False
                dst.write_bytes(data)
                return True
        except Exception as e:
            print(f"  Attempt {attempt+1}/{retries} failed for {dst.name}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return False


def make_sketch(sketch_id: str, filename: str) -> dict:
    return {"id": sketch_id, "filename": filename, "uploaded_at": utcnow()}


def make_variation(
    var_id: str, material: str, color: str, note: str = "",
    results: list[str] | None = None,
    generated_results: list[dict] | None = None,
    price_tier: str = "", theme: str = "", feature: str = "",
    status: str = "completed",
) -> dict:
    return {
        "id": var_id, "material": material, "color": color, "note": note,
        "pipeline": "qwen_style_transfer", "num_outputs": 1,
        "results": [{"filename": f} for f in (results or [])],
        "generated_results": generated_results or [],
        "spin_frames": [], "price_tier": price_tier,
        "theme": theme, "feature": feature,
        "status": status, "created_at": utcnow(),
    }


def make_product(
    pid: str, name: str, label: str = "", description: str = "",
    sketches: list[dict] | None = None, variations: list[dict] | None = None,
) -> dict:
    return {
        "id": pid, "name": name, "year": "2026", "label": label,
        "description": description, "sketches": sketches or [],
        "generations": [], "variations": variations or [],
        "created_at": utcnow(),
    }


def make_collection(
    cid: str, name: str, product_ids: list[str],
    price_tiers: list[str] | None = None,
    themes: list[str] | None = None,
    features: list[str] | None = None,
    description: str = "",
) -> dict:
    return {
        "id": cid, "name": name, "product_ids": product_ids,
        "target_price_tiers": price_tiers or [],
        "target_themes": themes or [],
        "target_features": features or [],
        "target_description": description, "created_at": utcnow(),
    }


def make_gen_result(filename: str, label: str = "") -> dict:
    return {
        "filename": filename, "pipeline": "qwen_style_transfer",
        "label": label, "created_at": utcnow(),
    }


# ---------------------------------------------------------------------------
# Casadei.com CDN
# ---------------------------------------------------------------------------

CDN = "https://www.casadei.com/dw/image/v2/BGDG_PRD/on/demandware.static/-/Sites-05/default"

# sku_color -> [(hash, view_index), ...]
IMG_MAP: dict[str, list[tuple[str, str]]] = {
    "1F915W100MC1155_9000": [
        ("dw49bd617f", "0"), ("dw396b0262", "1"), ("dw7c603f42", "2"),
        ("dw48aacd90", "3"), ("dwabda7c7e", "4"), ("dwa8436765", "5"),
    ],
    "1F920W100MC1444_9000": [
        ("dw4bdba742", "0"), ("dw51122283", "1"), ("dw0af9e87b", "2"),
        ("dwb0aa8f07", "3"), ("dw801d6c75", "4"), ("dw138f59d2", "5"),
    ],
    "1L488N120MC0596_3200": [
        ("dw6070f45f", "0"), ("dwb7e6aa16", "1"), ("dw2453b04d", "2"),
        ("dw7ab0c659", "3"), ("dwbd7d1b78", "4"), ("dw56e3624a", "5"),
    ],
    "1G590X080MC2924_2309": [
        ("dw89d5bf74", "0"), ("dw532ed55b", "1"), ("dw217546a9", "2"),
        ("dw02d1a3f9", "3"), ("dw46c7f48d", "4"), ("dwa3dde21f", "5"),
    ],
    "1G590X080MC2924_6706": [
        ("dw80242d2a", "0"), ("dw47002aed", "1"), ("dw37ad29e9", "2"),
        ("dwe451a1c7", "3"), ("dw8ee96bfb", "4"), ("dwc6b5b19b", "5"),
    ],
    "1L414B1001TIFFA_9999": [
        ("dw428f04b0", "0"), ("dw9c8aba08", "1"), ("dwdaf47027", "2"),
        ("dw84412e9c", "3"), ("dw7942367b", "4"), ("dwf6d1e455", "5"),
    ],
    "1L414B1001TIFFA_9000": [
        ("dw1d4761fe", "0"), ("dw989ac9bf", "1"), ("dwc171b952", "2"),
        ("dweac73b59", "3"), ("dwc76503c9", "4"), ("dw0b21ae87", "5"),
    ],
    "2X894U070NSALEN_9000": [
        ("dw2e6e3c30", "0"), ("dwa00ac16f", "1"), ("dw3aaf4024", "2"),
        ("dwc521f0c2", "3"), ("dw4b624834", "4"), ("dw649c9f1e", "5"),
    ],
    "2X094B0701C2986_9999": [
        ("dw60ff72b6", "0"), ("dwa30c0bb0", "1"), ("dw6b9a8c4f", "2"),
        ("dwc40ff345", "3"), ("dwe576be0a", "4"),
    ],
    "1L287Y1401SAMUR_3614": [
        ("dw968a5a8c", "0"), ("dwbe529fc9", "1"), ("dwc566e447", "2"),
        ("dw9112b7dd", "3"), ("dw3eb51e49", "4"), ("dwc33f028d", "5"),
    ],
    "1R538A1201NOMAD_9000": [
        ("dw276035b0", "0"), ("dw62e8f1f0", "1"), ("dw498ab91d", "2"),
        ("dwc371cad1", "3"), ("dwc90cabda", "4"), ("dw48c0a13c", "5"),
    ],
    "1R538A1201NOMAD_3702": [
        ("dw99536bc5", "0"), ("dw1b685487", "1"), ("dw95ddfce8", "2"),
        ("dw7eee05d0", "3"), ("dwe2bbcfcc", "4"), ("dwc1b86e04", "5"),
    ],
    "1H061B100MC1155_9000": [
        ("dw551a7bea", "0"), ("dw2af0387e", "1"), ("dwf4d80999", "2"),
        ("dw13922fa8", "3"), ("dw0c6d1e74", "4"),
    ],
    "1LG40D100MC1155_9000": [
        ("dw9b78bcc2", "0"), ("dw0fe20a98", "1"), ("dwd42333c9", "2"),
        ("dwd941d24a", "3"), ("dw5518b77d", "4"), ("dw67d993e0", "5"),
    ],
    "1G671B0101TIFFA_9999": [
        ("dwa04e4046", "0"), ("dw7ca21b7b", "1"), ("dwe4d5a809", "2"),
        ("dwcda7e198", "3"), ("dwbfb2957f", "4"), ("dw3c03c79b", "5"),
    ],
    "1R665E120MC2274_3810": [
        ("dwfd603f14", "0"), ("dwedfbd5db", "1"), ("dw82aef534", "2"),
        ("dw5f23d716", "3"), ("dw0ed6e14e", "4"), ("dw131e9d75", "5"),
    ],
    "1L405A1301DREAM_3217": [
        ("dw733902c5", "0"), ("dwf490499e", "1"), ("dw47ee6687", "2"),
        ("dwf261201f", "3"), ("dwd457d721", "4"), ("dw9649f332", "5"),
    ],
    "1F054B100MT0577_3206": [
        ("dw4955d1cb", "0"), ("dw091d4770", "1"), ("dwe2326638", "2"),
        ("dw1cc30092", "3"), ("dw6020a953", "4"), ("dw5f1bdda1", "5"),
    ],
}


def download_sku(sku_color: str, dst_dir: Path) -> list[str]:
    """Download all views for a SKU/color. Returns list of filenames."""
    images = IMG_MAP.get(sku_color, [])
    filenames = []
    for img_hash, idx in images:
        filename = f"{sku_color}_{idx}.jpg"
        url = f"{CDN}/{img_hash}/images/zoom/{sku_color}_{idx}.jpg"
        print(f"    {filename}...")
        if download_image(url, dst_dir / filename):
            filenames.append(filename)
    return filenames


# ---------------------------------------------------------------------------
# MVP Products (local images with real sketches)
# ---------------------------------------------------------------------------

def seed_mvp_shoe1() -> tuple[dict, str]:
    """Concept Ankle Boot — 2 sketches, 2 variations."""
    pid = new_id()
    print(f"\n--- Concept Ankle Boot [{pid}] ---")

    # Sketches (real hand-drawn)
    sk1_id, sk2_id = new_id(), new_id()
    sk_dir = UPLOADS_DIR / pid
    copy_file(MVP_DIR / "Shoe 1" / "sketches" / "sketch 1.jpg", sk_dir / f"{sk1_id}_sketch-1.jpg")
    copy_file(MVP_DIR / "Shoe 1" / "sketches" / "sketch 2.png", sk_dir / f"{sk2_id}_sketch-2.png")

    # Variation 1: Black Leather (product shots + try-on + context)
    v1_id = new_id()
    v1_dir = RESULTS_DIR / pid / v1_id
    v1_src = MVP_DIR / "Shoe 1" / "generated" / "Varitation 1"
    v1_files, v1_gen = [], []

    for f in sorted(v1_src.glob("black leather *.png")):
        fname = f.name.replace(" ", "-").lower()
        copy_file(f, v1_dir / fname)
        v1_files.append(fname)
        v1_gen.append(make_gen_result(fname, "generated"))

    for f in sorted((v1_src / "Try on").glob("*.png")):
        fname = f"tryon-{f.name.replace(' ', '-').lower()}"
        copy_file(f, v1_dir / fname)
        v1_files.append(fname)
        v1_gen.append(make_gen_result(fname, "try-on"))

    for f in sorted((v1_src / "Create Context").glob("*.png")):
        fname = f"context-{f.name.replace(' ', '-').lower()}"
        copy_file(f, v1_dir / fname)
        v1_files.append(fname)
        v1_gen.append(make_gen_result(fname, "context"))

    print(f"  Black Leather: {len(v1_files)} images (product + try-on + context)")

    # Variation 2: Green Snakeskin
    v2_id = new_id()
    v2_dir = RESULTS_DIR / pid / v2_id
    v2_src = MVP_DIR / "Shoe 1" / "generated" / "Varitation 2"
    v2_files, v2_gen = [], []

    for f in sorted(v2_src.glob("snakeskin *.png")):
        fname = f.name.replace(" ", "-").lower()
        copy_file(f, v2_dir / fname)
        v2_files.append(fname)
        v2_gen.append(make_gen_result(fname, "generated"))

    print(f"  Emerald Snakeskin: {len(v2_files)} images")

    return make_product(
        pid, "Concept Ankle Boot", "Ankle Boots",
        "Square-toe ankle boot with sculptural curved heel. Modern architectural silhouette.",
        sketches=[make_sketch(sk1_id, "sketch-1.jpg"), make_sketch(sk2_id, "sketch-2.png")],
        variations=[
            make_variation(v1_id, "Nappa Leather", "Black",
                "Classic black leather with sculptural heel",
                v1_files, v1_gen, "Core Luxury", "Cyber-Artisan", "Sculptural Hourglass Heel"),
            make_variation(v2_id, "Snakeskin", "Emerald Green",
                "Exotic green snakeskin texture",
                v2_files, v2_gen, "Premium", "Techno-Baroque", "Thermo-reactive Finish"),
        ],
    ), pid


def seed_mvp_shoe2() -> tuple[dict, str]:
    """Flora Platform Sandal — 1 technical sketch, 2 variations."""
    pid = new_id()
    print(f"\n--- Flora Platform Sandal [{pid}] ---")

    sk_id = new_id()
    sk_dir = UPLOADS_DIR / pid
    copy_file(MVP_DIR / "Shoe 2" / "sketches" / "sketch.jpg", sk_dir / f"{sk_id}_technical-drawing.jpg")

    # Variation 1: Blue Patent (angles + variants = all views of same color)
    v1_id = new_id()
    v1_dir = RESULTS_DIR / pid / v1_id
    v1_src = MVP_DIR / "Shoe 2" / "generated" / "Varitation 1"
    v1_files, v1_gen = [], []

    for name in ["hero-front-right", "hero-back-left", "front", "back", "side", "top"]:
        f = v1_src / f"{name}.png"
        if f.exists():
            copy_file(f, v1_dir / f"{name}.png")
            v1_files.append(f"{name}.png")
            v1_gen.append(make_gen_result(f"{name}.png", "generated"))

    for f in sorted(v1_src.glob("var 1*.png")):
        fname = f.name.replace(" ", "-").lower()
        copy_file(f, v1_dir / fname)
        v1_files.append(fname)
        v1_gen.append(make_gen_result(fname, "generated"))

    print(f"  Cobalt Blue Patent: {len(v1_files)} views")

    # Variation 2: Burgundy Leather
    v2_id = new_id()
    v2_dir = RESULTS_DIR / pid / v2_id
    v2_src = MVP_DIR / "Shoe 2" / "generated" / "Varitation 2"
    v2_files, v2_gen = [], []

    for f in sorted(v2_src.glob("var 2*.png")):
        fname = f.name.replace(" ", "-").lower()
        copy_file(f, v2_dir / fname)
        v2_files.append(fname)
        v2_gen.append(make_gen_result(fname, "generated"))

    print(f"  Burgundy Leather: {len(v2_files)} views")

    return make_product(
        pid, "Flora Platform Sandal", "Platforms",
        "Open-toe platform sandal with covered block heel and ankle strap. Patent leather upper, metal buckle. Made in Italy.",
        sketches=[make_sketch(sk_id, "technical-drawing.jpg")],
        variations=[
            make_variation(v1_id, "Patent Leather", "Cobalt Blue",
                "Vibrant blue patent leather, multi-angle views",
                v1_files, v1_gen, "Premium", "Techno-Baroque", "Invisible Strap System"),
            make_variation(v2_id, "Nappa Leather", "Burgundy",
                "Rich burgundy leather with ankle strap",
                v2_files, v2_gen, "Core Luxury", "Mediterranean Summer", "Anatomical Arch Support"),
        ],
    ), pid


# ---------------------------------------------------------------------------
# Website Products — grouped by shoe design, colors as variations
# ---------------------------------------------------------------------------

# Each entry is one PRODUCT with one or more color VARIATIONS
PRODUCTS_GROUPED = [
    {
        "name": "Blade Pump Leather",
        "label": "Pumps",
        "description": "Iconic pumps in soft nappa leather with 10cm steel blade heel. The perfect balance between femininity and design.",
        "feature": "Sculptural Hourglass Heel",
        "variations": [
            {"sku": "1F915W100MC1155_9000", "material": "Nappa Leather", "color": "Black",
             "price_tier": "Core Luxury", "theme": "Mediterranean Summer"},
        ],
    },
    {
        "name": "Superblade Patent Leather",
        "label": "Pumps",
        "description": "Elongated pointed pump with hardened steel stiletto. More dramatic taper culminating in an extended point.",
        "feature": "Sculptural Hourglass Heel",
        "variations": [
            {"sku": "1F920W100MC1444_9000", "material": "Patent Calf Leather", "color": "Black",
             "price_tier": "Core Luxury", "theme": "Techno-Baroque"},
        ],
    },
    {
        "name": "Superblade Slingback",
        "label": "Pumps",
        "description": "Iconic Superblade slingback with 8cm Blade heel. Vinyl detail adds lightness, combining feminine elegance with bold character.",
        "feature": "Invisible Strap System",
        "variations": [
            {"sku": "1G590X080MC2924_2309", "material": "Patent Leather", "color": "Topaz",
             "price_tier": "Premium", "theme": "Techno-Baroque"},
            {"sku": "1G590X080MC2924_6706", "material": "Patent Leather", "color": "Emerald",
             "price_tier": "Premium", "theme": "Techno-Baroque"},
        ],
    },
    {
        "name": "Chantilly Blade Pump",
        "label": "Pumps",
        "description": "Delicate Chantilly lace-detail pump with Blade heel. Bridal and occasion wear.",
        "feature": "Hand-stitched Piping",
        "variations": [
            {"sku": "1F054B100MT0577_3206", "material": "Lace & Leather", "color": "Milk",
             "price_tier": "Premium", "theme": "Mediterranean Summer"},
        ],
    },
    {
        "name": "Blade V Celebrity",
        "label": "Sandals",
        "description": "Minimalist V Celebrity sandal with vinyl triangle at the ankle and iconic 12cm Blade heel. Perfect for elegant evenings.",
        "feature": "Invisible Strap System",
        "variations": [
            {"sku": "1L488N120MC0596_3200", "material": "Nappa Leather", "color": "Ecru",
             "price_tier": "Premium", "theme": "Mediterranean Summer"},
        ],
    },
    {
        "name": "Julia Double Belts",
        "label": "Sandals",
        "description": "Sensual Double Belts sandal with thin crossing straps and double heel straps with metal buckles.",
        "feature": "Anatomical Arch Support",
        "variations": [
            {"sku": "1L414B1001TIFFA_9999", "material": "Patent Leather", "color": "White",
             "price_tier": "Premium", "theme": "Mediterranean Summer"},
            {"sku": "1L414B1001TIFFA_9000", "material": "Patent Leather", "color": "Black",
             "price_tier": "Premium", "theme": "Techno-Baroque"},
        ],
    },
    {
        "name": "Cappa Blade Leather",
        "label": "Sandals",
        "description": "Minimalist Cappa sandal with Blade heel. Clean lines and timeless elegance.",
        "feature": "Sculptural Hourglass Heel",
        "variations": [
            {"sku": "1LG40D100MC1155_9000", "material": "Nappa Leather", "color": "Black",
             "price_tier": "Core Luxury", "theme": "Mediterranean Summer"},
        ],
    },
    {
        "name": "Nexus Sneaker",
        "label": "Sneakers",
        "description": "Techno-futuristic sneaker with ultra-light 70mm wedge sole, iconic C-Chain element, and maxi logo.",
        "feature": "Water-resistant Technical Silk",
        "variations": [
            {"sku": "2X894U070NSALEN_9000", "material": "Calf Leather", "color": "Black",
             "price_tier": "Core Luxury", "theme": "Cyber-Artisan"},
            {"sku": "2X094B0701C2986_9999", "material": "Calf Leather", "color": "White",
             "price_tier": "Core Luxury", "theme": "Cyber-Artisan"},
        ],
    },
    {
        "name": "Flora Platform Samurai",
        "label": "Platforms",
        "description": "Sparkling Flora model with 12cm blade heel and platform. Metallic patent leather designed to shine with every step.",
        "feature": "Thermo-reactive Finish",
        "variations": [
            {"sku": "1L287Y1401SAMUR_3614", "material": "Patent Leather", "color": "Ultrared",
             "price_tier": "Premium", "theme": "Techno-Baroque"},
        ],
    },
    {
        "name": "Patty Dream Lac Platform",
        "label": "Platforms",
        "description": "Patty platform in lacquered leather. Dream collection with chunky sole.",
        "feature": "Thermo-reactive Finish",
        "variations": [
            {"sku": "1L405A1301DREAM_3217", "material": "Lacquered Leather", "color": "Offwhite",
             "price_tier": "Core Luxury", "theme": "Mediterranean Summer"},
        ],
    },
    {
        "name": "Nancy Ankle Boot",
        "label": "Ankle Boots",
        "description": "Platform ankle boot with chunky heel, rubber tank sole, and lace-up fastening with metal hooks. Metropolitan style.",
        "feature": "Anatomical Arch Support",
        "variations": [
            {"sku": "1R538A1201NOMAD_9000", "material": "Suede & Leather", "color": "Black",
             "price_tier": "Core Luxury", "theme": "Cyber-Artisan"},
            {"sku": "1R538A1201NOMAD_3702", "material": "Suede", "color": "Rubino",
             "price_tier": "Core Luxury", "theme": "Cyber-Artisan"},
        ],
    },
    {
        "name": "Superblade Absol Blade",
        "label": "Ankle Boots",
        "description": "Sleek ankle boot combining the Superblade silhouette with the Absol design. Signature steel blade heel.",
        "feature": "Sculptural Hourglass Heel",
        "variations": [
            {"sku": "1H061B100MC1155_9000", "material": "Nappa Leather", "color": "Black",
             "price_tier": "Premium", "theme": "Cyber-Artisan"},
        ],
    },
    {
        "name": "Blade Ankle Boot",
        "label": "Ankle Boots",
        "description": "Ankle boot with 12cm Blade heel. Bold color meets iconic silhouette.",
        "feature": "Sculptural Hourglass Heel",
        "variations": [
            {"sku": "1R665E120MC2274_3810", "material": "Leather", "color": "Ladybug",
             "price_tier": "Premium", "theme": "Cyber-Artisan"},
        ],
    },
    {
        "name": "Ballet Slingback",
        "label": "Flats",
        "description": "Elegant ballet slingback in leather. Effortless style for everyday sophistication.",
        "feature": "Anatomical Arch Support",
        "variations": [
            {"sku": "1G671B0101TIFFA_9999", "material": "Leather", "color": "White",
             "price_tier": "Core Luxury", "theme": "Mediterranean Summer"},
        ],
    },
]


def seed_website_products() -> list[tuple[dict, str]]:
    """Download and create grouped products from casadei.com."""
    products = []

    for pg in PRODUCTS_GROUPED:
        pid = new_id()
        print(f"\n--- {pg['name']} [{pid}] ---")

        variations = []
        for var_def in pg["variations"]:
            var_id = new_id()
            var_dir = RESULTS_DIR / pid / var_id

            print(f"  Variation: {var_def['color']} ({var_def['material']})")
            filenames = download_sku(var_def["sku"], var_dir)

            if not filenames:
                print(f"    SKIPPED: no images downloaded")
                continue

            gen = [make_gen_result(f, "generated") for f in filenames]
            variations.append(make_variation(
                var_id, var_def["material"], var_def["color"],
                f"{len(filenames)} views",
                filenames, gen,
                var_def["price_tier"], var_def["theme"], pg["feature"],
            ))

        if not variations:
            print(f"  SKIPPED: no variations for {pg['name']}")
            continue

        product = make_product(
            pid, pg["name"], pg["label"], pg["description"],
            sketches=[],  # No sketches for website products — only real sketches
            variations=variations,
        )
        products.append((product, pid))

    return products


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

def create_collections(all_products: list[tuple[dict, str]]) -> list[dict]:
    by_label: dict[str, list[str]] = {}
    by_feature: dict[str, list[str]] = {}

    for prod, pid in all_products:
        label = prod.get("label", "")
        if label:
            by_label.setdefault(label, []).append(pid)
        for var in prod.get("variations", []):
            feat = var.get("feature", "")
            if feat:
                by_feature.setdefault(feat, []).append(pid)

    for d in (by_label, by_feature):
        for k in d:
            d[k] = list(dict.fromkeys(d[k]))

    collections = []

    boot_pids = by_label.get("Ankle Boots", [])
    if boot_pids:
        collections.append(make_collection(
            new_id(), "FW26 Boots & Ankle Boots", boot_pids,
            ["Core Luxury", "Premium"], ["Cyber-Artisan"],
            ["Sculptural Hourglass Heel", "Anatomical Arch Support"],
            "Fall/Winter 2026 boot collection featuring Blade ankle boots, Nancy platforms, and the Concept boot.",
        ))

    blade_pids = by_feature.get("Sculptural Hourglass Heel", [])
    if blade_pids:
        collections.append(make_collection(
            new_id(), "SS26 Blade Collection", blade_pids,
            ["Core Luxury", "Premium"], ["Mediterranean Summer", "Techno-Baroque", "Cyber-Artisan"],
            ["Sculptural Hourglass Heel"],
            "The iconic Blade heel across pumps, sandals, slingbacks, and boots. Signature stainless steel stiletto.",
        ))

    platform_pids = list(dict.fromkeys(
        by_label.get("Platforms", []) + by_label.get("Sandals", [])
    ))
    if platform_pids:
        collections.append(make_collection(
            new_id(), "SS26 Platforms & Sandals", platform_pids,
            ["Core Luxury", "Premium"], ["Mediterranean Summer", "Techno-Baroque"],
            ["Thermo-reactive Finish", "Invisible Strap System", "Anatomical Arch Support", "Sculptural Hourglass Heel"],
            "Statement platforms and elegant sandals. Flora, Julia, and Cappa styles.",
        ))

    nexus_pids = by_label.get("Sneakers", [])
    if nexus_pids:
        collections.append(make_collection(
            new_id(), "Nexus Sneakers", nexus_pids,
            ["Core Luxury"], ["Cyber-Artisan"], ["Water-resistant Technical Silk"],
            "The Nexus line: techno-futuristic sneakers with ultra-light wedge soles.",
        ))

    return collections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Casadei Data Seeder v2")
    print("  (grouped products, proper variations)")
    print("=" * 60)

    # Clear
    for sub in [UPLOADS_DIR, RESULTS_DIR]:
        if sub.exists():
            shutil.rmtree(sub)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_products: list[tuple[dict, str]] = []

    # Phase 1: MVP shoes (with real sketches)
    print("\n" + "=" * 60)
    print("  Phase 1: MVP Shoes (with sketches)")
    print("=" * 60)

    shoe1, pid1 = seed_mvp_shoe1()
    all_products.append((shoe1, pid1))

    shoe2, pid2 = seed_mvp_shoe2()
    all_products.append((shoe2, pid2))

    # Phase 2: Website products (grouped, no fake sketches)
    print("\n" + "=" * 60)
    print("  Phase 2: Casadei.com Products (grouped)")
    print("=" * 60)

    website_products = seed_website_products()
    all_products.extend(website_products)

    # Phase 3: Collections
    print("\n" + "=" * 60)
    print("  Phase 3: Collections")
    print("=" * 60)

    collections = create_collections(all_products)
    for c in collections:
        print(f"  {c['name']} ({len(c['product_ids'])} products)")

    # Write store
    store = {
        "products": {pid: prod for prod, pid in all_products},
        "collections": {c["id"]: c for c in collections},
    }
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2, default=str))

    # Summary
    n_prods = len(all_products)
    n_sketches = sum(len(p.get("sketches", [])) for p, _ in all_products)
    n_vars = sum(len(p.get("variations", [])) for p, _ in all_products)
    n_imgs = sum(len(v.get("results", [])) for p, _ in all_products for v in p.get("variations", []))

    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Products:    {n_prods} (2 MVP + {n_prods - 2} website)")
    print(f"  Sketches:    {n_sketches} (MVP only — real hand-drawn)")
    print(f"  Variations:  {n_vars} (each = one material/color)")
    print(f"  Images:      {n_imgs} (views within variations)")
    print(f"  Collections: {len(collections)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
