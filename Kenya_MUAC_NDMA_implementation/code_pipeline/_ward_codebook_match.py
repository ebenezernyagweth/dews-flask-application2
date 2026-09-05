import re, pandas as pd

# Minimal normalizer for MATCHING KEYS ONLY (data text stays as-is until we assign)
def _key(s: str) -> str:
    s = str(s)
    s = s.replace('\u00A0', ' ')                 # NBSP -> space
    s = re.sub(r'[\u2010-\u2015\u2212]', '-', s) # en/em/minus -> '-'
    s = re.sub(r"[’‘´`ʼ′]", "'", s)              # curly/special apostrophes -> "'"
    s = re.sub(r'\s+', ' ', s).strip()           # collapse spaces
    s = re.sub(r'\s*([/-])\s*', r'\1', s)        # trim around '/' and '-'
    return s.casefold()                           # case-insensitive key

def replace_wards_and_counties(muac: pd.DataFrame) -> pd.DataFrame:
    muac = muac.copy()

    # --- your codebook mapping (raw) ---
    ward_map_raw = {
        "Abakayle": "Abakaile",
        "Adu/Chakama": "Adu",
        "Banisa": "Banissa",
        "Baharini": "Bahari",
        "Barwaqo": "Barwago",
        "Bula Pesa": "Bulla Pesa",
        "Charri": "Chari",
        "Challa": "Chala",
        "Chengoni-Samburu": "Chengoni/Samburu",
        "Elnur": "El-Nur/Tula-tula",
        "Endau/Malalani": "Malalani/Endau",
        "Ewuaso Kidong": "Ewuaso O o Nkidong'i/M",
        "Ewuaso Kedong": "Ewuaso O o Nkidong'i/M",
        "Ewuaso O O Nkidong'I/M": "Ewuaso O o Nkidong'i/M",
        "Hadado": "Hadado/Adhibohol",
        "Hadado/Athibohol": "Hadado/Adhibohol",
        "Heilu Manyatta": "Heillu",
        "Jarajila": "Jara Jila",
        "KANYANGI ": "Kanyangi",
        "KASIKEU": "Kasikeu/Kiou",
        "Kasikeu": "Kasikeu/Kiou",
        "Kakuma/Letea": "Kakuma",
        "Kapchok": "Kapchoki",
        "Kaputiei North": "Kaputei North",
        "Kasemeni": "Kasemani",
        "Khalalio": "Kalaliyo",
        "Kithungo/Kitundu": "Kithungo/kitundu",
        "Korr": "Korr/Ngurunit",
        "Korr/Ngurnit": "Korr/Ngurunit",
        "Lagbogol": "Lagboghol South",
        "Lokori/Kachodin": "Lokori/Kochodin",
        "MERILLE": "Merigi",
        "Merille": "Merigi",
        "MUKAA": "Mukaa",
        "Maalimin": "Maalamin",
        "Mbirikani": "Imbirikani/Eselenkei",
        "NGUU/MASAMBA": "Masumba/Nguu",
        "NGUU/MASUMBA": "Masumba/Nguu",
        "Oldonyiro": "Oldo-nyiro",
        "Oldo-Nyiro": "Oldo-nyiro",
        "Ribko": "Ribikwo",
        "Sagante": "Sagante/Jaldessa",
        "Sagante/Jaldesa": "Sagante/Jaldessa",
        "Saimo Soi": "Saimo/ Kipsaraman",
        "Saimo/Kipsaraman": "Saimo/ Kipsaraman",
        "Simbir Fatuma": "Shimbir Fatuma",
        "Sosian": "Sosiani",
        "Suguta Mar Mar": "Suguta Marmar",
        "Thegu river": "Thegu River",
        "Warankara": "Waranqara",
        "Weiwei": "Wei Wei",
        "Wumingu": "Wumingu/kishushe",
        "Wumingu/Kishushe": "Wumingu/kishushe",
        "Waa-Ng'ombeni": "Waa",
        "Waa-Ng'Ombeni": "Waa",
        "Wusi": "Wusi/Kishamba",
        "Dasheq": "Tarbaj",
        "Kamarandi": "Evurore",
        "Komolion": "Tangulbei/Korossi",
        "Tangulbei": "Tangulbei/Korossi",
        "Nachukui": "Lake Zone",
        "Naroosura": "Maji Moto/Naroosura",
        "Orus": "Churo/Amaya",
        "Kalemng'orok": "Katilu",
        "Laisamis": "Logologo",
        "Loodokilani": "Iloodokilani",
        "Lorugum": "Turkwel",
        "Oldonyonyokie": "Magadi",
        "Tunyai": "Chiakariga",
        "Kanyuambora": "Evurore",
        "Kyome/Thaana": "Migwani",
        "Githiga": "Ngobit",
        "Amwitha": "Kiegoi/Antubochiu",
        "Mosiro": "MOSIRO(Narok East)",
        "Mosiro(Narok East": "MOSIRO(Narok East)",
        "Akoret": "Kapedo/Napeitom",
        "Kinisa": "Heillu",
        "Bori": "Butiye",
        "DELA": "Dela",
        "Kalemngorok": "Katilu",
        "Kangeta/Kalimbene ": "Kangeta",
        "Batalu": "Batalu/Buna",
        # Your example that wasn't working:
        "Kasikeu": "Kasikeu/Kiou",
    }

    # Build normalized-key mapping (so "Kasikeu" always matches)
    ward_map = {_key(k): v for k, v in ward_map_raw.items()}

    # Create a temp normalized key column for Wards in your data
    ward_norm = muac['Ward'].astype(str).apply(_key)

    # Apply mapping where the normalized key is present; keep original otherwise
    mask = ward_norm.isin(ward_map)
    muac.loc[mask, 'Ward'] = ward_norm[mask].map(ward_map)

    # County fixes (triggered by ward) using normalized keys too
    county_fix = {
        _key("Kapenguria"): "West pokot",
        _key("Akoret"): "Turkana",
        _key("Evurore"): "Embu",
        _key("Benane"): "Garissa",
        _key("Nanighi"): "Garissa",
        _key("Kapedo/Napeitom"): "Turkana",
    }
    muac.loc[ward_norm.isin(county_fix), 'County'] = ward_norm.map(county_fix)

    return muac
