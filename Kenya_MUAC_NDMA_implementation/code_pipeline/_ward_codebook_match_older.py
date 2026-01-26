import pandas as pd

def replace_wards_and_counties(muac):
    muac['Ward'] = muac['Ward'].replace({
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
        "Hadado": "Hadado/Adhibohol",
        "Heilu Manyatta": "Heillu",
        "Jarajila": "Jara Jila",
        "KANYANGI ": "Kanyangi",
        "KASIKEU": "Kasikeu/Kiou",
        "Kakuma/Letea": "Kakuma",
        "Kapchok": "Kapchoki",
        "Kaputiei North": "Kaputei North",
        "Kasemeni": "Kasemani",
        "Khalalio": "Kalaliyo",
        "Kithungo/Kitundu": "Kithungo/kitundu",
        "Korr": "Korr/Ngurunit",
        "Lagbogol": "Lagboghol South",
        "Lokori/Kachodin": "Lokori/Kochodin",
        "MERILLE": "Merigi",
        "MUKAA": "Mukaa",
        "Maalimin": "Maalamin",
        "Mbirikani": "Imbirikani/Eselenkei",
        "NGUU/MASAMBA": "Masumba/Nguu",
        "Oldonyiro": "Oldo-nyiro",
        "Ribko": "Ribikwo",
        "Sagante": "Sagante/Jaldessa",
        "Saimo Soi": "Saimo/ Kipsaraman",
        "Simbir Fatuma": "Shimbir Fatuma",
        "Sosian": "Sosiani",
        "Suguta Mar Mar": "Suguta Marmar",
        "Thegu river": "Thegu River",
        "Warankara": "Waranqara",
        "Weiwei": "Wei Wei",
        "Wumingu": "Wumingu/kishushe",
        "Waa-Ng'ombeni": "Waa",
        "Wusi": "Wusi/Kishamba",
        "Dasheq": "Tarbaj",
        "Kamarandi": "Evurore",
        "Komolion": "Tangulbei/Korossi",
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
        "Akoret": "Kapedo/Napeitom",
        "Kinisa": "Heillu",
        "NGUU/MASUMBA" :"Masumba/Nguu",
        'NGUU/MASAMBA':"Masumba/Nguu",
        "Bori":"Butiye",
        "DELA": "Dela",
        "Kalemngorok":"Katilu",
        'Kangeta/Kalimbene ': "Kangeta",
        'Batalu': 'Batalu/Buna'

    })

    # Replace values in 'County' column based on 'Ward' values
    muac.loc[muac['Ward'] == "Kapenguria", 'County'] = "West pokot"
    muac.loc[muac['Ward'] == "Akoret", 'County'] = "Turkana"
    muac.loc[muac['Ward'] == "Evurore", 'County'] = "Embu"
    muac.loc[muac['Ward'] == "Benane", 'County'] = "Garissa"
    muac.loc[muac['Ward'] == "Nanighi", 'County'] = "Garissa"
    muac.loc[muac['Ward'] == "Kapedo/Napeitom", 'County'] = "Turkana"

    return muac
