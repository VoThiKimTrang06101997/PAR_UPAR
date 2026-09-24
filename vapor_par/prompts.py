from .config import ATTRIBUTE_NAMES

def _concept(attr: str) -> str:
    explicit = {
        "Age-Young": "young",
        "Age-Adult": "adult",
        "Age-Old": "elderly",
        "Gender-Female": "female",
        "Hair-Length-Short": "short hair",
        "Hair-Length-Long": "long hair",
        "Hair-Length-Bald": "bald",
        "UpperBody-Length-Short": "short-sleeved upper-body clothing",
        "LowerBody-Length-Short": "short lower-body clothing",
        "LowerBody-Type-Trousers&Shorts": "trousers or shorts",
        "LowerBody-Type-Skirt&Dress": "a skirt or dress",
        "Accessory-Backpack": "a backpack",
        "Accessory-Bag": "a bag",
        "Accessory-Glasses-Normal": "normal eyeglasses",
        "Accessory-Glasses-Sun": "sunglasses",
        "Accessory-Hat": "a hat",
    }
    if attr in explicit:
        return explicit[attr]
    if "UpperBody-Color-" in attr:
        return f"{attr.split('UpperBody-Color-')[-1].lower()} upper-body clothing"
    if "LowerBody-Color-" in attr:
        return f"{attr.split('LowerBody-Color-')[-1].lower()} lower-body clothing"
    return attr.replace("-", " ").lower()

DOMAIN_CONTEXTS = [
    "a surveillance image of",
    "a low-resolution CCTV image of",
    "a nighttime surveillance image of",
    "a blurred surveillance image of",
    "an occluded surveillance image of",
    "a high-angle camera image of",
]

def build_prompt_bank(attributes=ATTRIBUTE_NAMES):
    bank = {}
    for attr in attributes:
        c = _concept(attr)
        if attr.startswith("Age-"):
            pos_core = f"a pedestrian who appears {c}"
            neg_core = f"a pedestrian who does not appear {c}"
        elif attr == "Gender-Female":
            pos_core = "a female pedestrian"
            neg_core = "a pedestrian who is not female"
        elif attr.startswith("Hair-Length-"):
            if c == "bald":
                pos_core = "a bald pedestrian"
                neg_core = "a pedestrian who is not bald"
            else:
                pos_core = f"a pedestrian with {c}"
                neg_core = f"a pedestrian without {c}"
        elif attr.startswith("UpperBody-Color-") or attr == "UpperBody-Length-Short":
            pos_core = f"a pedestrian wearing {c}"
            neg_core = f"a pedestrian not wearing {c}"
        elif attr.startswith("LowerBody-Color-") or attr.startswith("LowerBody-Type-") or attr == "LowerBody-Length-Short":
            pos_core = f"a pedestrian wearing {c}"
            neg_core = f"a pedestrian not wearing {c}"
        else:
            pos_core = f"a pedestrian wearing or carrying {c}"
            neg_core = f"a pedestrian without {c}"

        pos = [f"{ctx} {pos_core}" for ctx in DOMAIN_CONTEXTS]
        neg = [f"{ctx} {neg_core}" for ctx in DOMAIN_CONTEXTS]
        bank[attr] = {"positive": pos, "negative": neg}
    return bank
