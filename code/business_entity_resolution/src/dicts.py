"""Static normalization dictionaries (general language/geography knowledge, no business data).

Tables are keyed by country *string* where the meaning is country-specific; unknown countries simply get
no country-specific entries (country is an open set). Everything else is applied to all records.
"""

# ---- names --------------------------------------------------------------------------------------------
# Legal-form tokens (after dots are removed: "p.c." -> "pc", "l.l.c." -> "llc", "s.a.s" -> "sas").
LEGAL = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co", "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt",
    "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc", "pa": "pa", "public": "public",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc",
    "ei": "ei", "ets": "ets", "etablissements": "ets", "gmbh": "gmbh",
}
# Honorifics / filler the noise adds to names (m/s is handled by regex before punctuation removal).
NAME_STOP = {"and", "the", "et", "mr", "mrs", "ms", "smt", "shri", "sri", "dr", "de", "du", "des", "la", "le", "les", "l", "d"}

# ---- addresses ----------------------------------------------------------------------------------------
STREET = {
    "street": "st", "st": "st", "str": "st", "saint": "st", "sainte": "ste", "ste": "ste",
    "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave", "av": "ave", "aenue": "ave",
    "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "place": "pl", "pl": "pl",
    "circle": "cir", "cir": "cir", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy",
    "terrace": "terr", "terr": "terr", "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq",
    "point": "pt", "pt": "pt", "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
    "apartment": "apt", "apt": "apt", "suite": "apt", "unit": "apt",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    # French street types
    "rue": "rue", "r": "rue", "impasse": "imp", "imp": "imp", "allee": "allee", "all": "allee",
    "chemin": "chemin", "che": "chemin", "chem": "chemin", "route": "rte", "rte": "rte",
    "cours": "cours", "crs": "cours", "quai": "quai", "faubourg": "fbg", "fbg": "fbg",
    "residence": "res", "res": "res", "lotissement": "lot",
    # Indian address words
    "nagar": "nagar", "ngr": "nagar", "marg": "marg", "floor": "flr", "flr": "flr", "fl": "flr",
    "sector": "sec", "sec": "sec", "building": "bldg", "bldg": "bldg", "opposite": "opp",
}
# Tokens that carry no identity: number prefixes, landmark markers, null markers, French particles.
ADDR_STOP = {
    "no", "nos", "hno", "dno", "h", "house", "plot", "flat", "door", "shop", "old", "new_no", "kh", "khasra",
    "near", "nr", "opp", "behind", "bh", "c", "o", "co",  # "c/o" -> c o
    "null", "na", "n/a", "cdp", "po", "box", "pmb", "bis", "ter", "quater",
    "de", "du", "des", "la", "le", "les", "l", "d", "au", "aux", "en", "of", "the", "and",
}

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
}
INDIA_STATES = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam",
    "BR": "Bihar", "CG": "Chhattisgarh", "CT": "Chhattisgarh", "CH": "Chandigarh", "DL": "Delhi", "GA": "Goa",
    "GJ": "Gujarat", "HR": "Haryana", "HP": "Himachal Pradesh", "JK": "Jammu and Kashmir", "JH": "Jharkhand",
    "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh", "MP": "Madhya Pradesh", "MH": "Maharashtra",
    "MN": "Manipur", "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "OR": "Odisha",
    "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu",
    "TG": "Telangana", "TS": "Telangana", "TR": "Tripura", "UP": "Uttar Pradesh", "UK": "Uttarakhand",
    "UT": "Uttarakhand", "WB": "West Bengal", "DN": "Dadra and Nagar Haveli and Daman and Diu",
}
INDIA_EXTRA = {"Orissa": "Odisha", "Keralam": "Kerala", "NCT of Delhi": "Delhi", "Pondicherry": "Puducherry",
               "Uttaranchal": "Uttarakhand"}
# French regions and the departements of the regions (both are used as the last address component).
FR_DEPTS = {
    "Hauts-de-France": ["Aisne", "Nord", "Oise", "Pas-de-Calais", "Somme"],
    "Nouvelle-Aquitaine": ["Charente", "Charente-Maritime", "Correze", "Creuse", "Dordogne", "Gironde", "Landes",
                           "Lot-et-Garonne", "Pyrenees-Atlantiques", "Deux-Sevres", "Vienne", "Haute-Vienne"],
    "Pays de la Loire": ["Loire-Atlantique", "Maine-et-Loire", "Mayenne", "Sarthe", "Vendee"],
    "Ile-de-France": ["Paris", "Seine-et-Marne", "Yvelines", "Essonne", "Hauts-de-Seine", "Seine-Saint-Denis",
                      "Val-de-Marne", "Val-d'Oise"],
    "Auvergne-Rhone-Alpes": ["Rhone", "Isere", "Loire", "Ain", "Savoie", "Haute-Savoie", "Puy-de-Dome", "Allier"],
    "Provence-Alpes-Cote d'Azur": ["Bouches-du-Rhone", "Var", "Alpes-Maritimes", "Vaucluse"],
    "Occitanie": ["Haute-Garonne", "Herault", "Gard", "Pyrenees-Orientales", "Aude", "Tarn"],
    "Grand Est": ["Bas-Rhin", "Haut-Rhin", "Moselle", "Meurthe-et-Moselle", "Marne", "Aube"],
    "Bretagne": ["Ille-et-Vilaine", "Finistere", "Morbihan", "Cotes-d'Armor"],
    "Normandie": ["Seine-Maritime", "Calvados", "Manche", "Eure", "Orne"],
    "Centre-Val de Loire": ["Loiret", "Indre-et-Loire", "Loir-et-Cher", "Cher", "Indre", "Eure-et-Loir"],
    "Bourgogne-Franche-Comte": ["Cote-d'Or", "Doubs", "Saone-et-Loire", "Yonne", "Jura", "Nievre"],
    "Corse": ["Corse-du-Sud", "Haute-Corse"],
}


def state_aliases():
    """{country: {normalized alias: canonical state}}. Aliases are normalized with the address normalizer's
    component key (lowercase, accents stripped, punctuation -> space)."""
    from .normalize import comp_key
    out = {"US": {}, "India": {}, "France": {}}
    for code, name in US_STATES.items():
        out["US"][comp_key(code)] = out["US"][comp_key(name)] = comp_key(name)
    for code, name in INDIA_STATES.items():
        out["India"][comp_key(code)] = out["India"][comp_key(name)] = comp_key(name)
    for alias, name in INDIA_EXTRA.items():
        out["India"][comp_key(alias)] = comp_key(name)
    for region, depts in FR_DEPTS.items():
        out["France"][comp_key(region)] = comp_key(region)
        for d in depts:
            out["France"][comp_key(d)] = comp_key(region)
    return out
