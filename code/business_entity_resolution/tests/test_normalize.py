"""Golden cases taken from real rows of the training/test data."""
from src.normalize import norm_addr, norm_name


def core(name):
    return norm_name(name)[0]


def test_legal_prefix_junk_and_repeats():
    a = norm_name("[Co] Leland and Runk Offshore")
    b = norm_name("Leland Leland and Runk Offshore Company")
    assert set(a[0].split()) == set(b[0].split()) == {"leland", "runk", "offshore"}
    assert a[2] == b[2] == "co"


def test_domain_and_concat():
    d = norm_name("greatresources.com")
    assert d[3] is True
    assert d[1] == norm_name("GREAT RESOURCES LLC")[1] == "greatresources"
    assert norm_name("Reflovuecompanies.Com")[1] == norm_name("Reflovue Companies")[1]


def test_indian_legal_and_honorifics():
    assert norm_name("KING (INDIA) AGRI PRIVATE LTD LTD")[1] == norm_name("King (India) Agri Private Limited")[1]
    assert core("M/S Jai Software Enterprises Pvt. Ltd.") == "jai software enterprises"
    assert norm_name("One Infra-Pvt Ltd Ltd")[2] == "ltd pvt"


def test_french_names():
    assert norm_name("Fractales Amis Groupe S.A.S")[2] == "sas"
    assert core("SCI Ptit Àmicale") == "ptit amicale"
    assert norm_name("9-Póint")[1] == norm_name("9-Point")[1]


def test_us_address_variants():
    a = norm_addr("8880 Hawthorn Point, Westerville, OH", "US")
    b = norm_addr("8880 Hawthorn Pt, Westerville, Ohio", "US")
    assert a == b and a[2] == "ohio"
    assert norm_addr("0302 SECOND AVE, ONEIDA, TN", "US")[1] == norm_addr("302 Second Avenue, Oneida, TN", "US")[1] == "302 2"
    assert norm_addr("448-D Chism Street, Albany, Texas", "US")[1] == "448"
    assert "1705" in norm_addr("SIOUX  CITY, 1705 1/2 25TH STREET, IA", "US")[1].split()


def test_india_address_variants():
    a = norm_addr("#129 Unit 1St Floor Bldgno D-2 Wadala, Mumbai, महाराष्ट्र", "India")
    b = norm_addr("H.NO 129 UNIT 1ST FLOOR BLDGNO D-2 WADALA, MUMBAI, Maharashtra", "India")
    assert a[1] == b[1] == "129 1 2"
    assert b[2] == "maharashtra"
    assert norm_addr("303 Dwarka Chs Ltd, Goregaon West, MH", "India")[2] == "maharashtra"


def test_french_address_variants():
    a = norm_addr("30 BIS R. DE VERDUN, PORNIC, Pays de la Loire", "France")
    assert a[0] == "30 rue verdun pornic" and a[1] == "30" and a[2] == "pays de la loire"
    b = norm_addr("N° 17 RUE DES FAUVETTES, DUNKERQUE, Hauts-de-France", "France")
    assert b[0] == "17 rue fauvettes dunkerque"
    # S3 writes the departement where S1 writes the region
    assert norm_addr("34 Impasse Du Grand Ousteau, Lège-cap-ferret, Gironde", "France")[2] == \
        norm_addr("3 Place de la Nation, Bordeaux, Nouvelle-Aquitaine", "France")[2]


def test_slash_numbers_kept_fraction_dropped():
    assert norm_addr("No #126/22, Madras, Chennai", "India")[1] == "126 22"
    assert norm_addr("Kh-45/2 Gali No-6, Karawal Nagar", "India")[1] == "45 2 6"
    assert norm_addr("5586 1/2 WHISPER LN, CINCINNATI, OH", "US")[1] == "5586"
    assert "n" not in norm_addr("Gola, N/A, UP", "India")[0].split()


def test_unknown_country_is_open_set():
    t, nums, state = norm_addr("12 Main Street, Springfield, XX", "Narnia")
    assert nums == "12" and state == ""
