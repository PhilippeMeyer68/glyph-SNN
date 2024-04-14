# -*- coding: utf-8 -*-
"""
This script stores the alphabets, the unicodes of the capital letters and the 
names of the associated font files.

Author: Claire Roman, Philippe Meyer
Email: philippemeyer68@yahoo.fr
Date: 03/2024
"""

# Dictionary of the alphabets, associated unicodes and font files.
dict_alphabets = {
    "Arabic": (list(range(0x0621,0x063B))+list(range(0x0641,0x064b)),"NotoSansArabic"),
    "Armenian": (list(range(0x00531,0x00557)),"NotoSansArmenian"),
    "Avestan": (list(range(0x10B00,0x10B36)),"NotoSansAvestan"),
    "Carian": (list(range(0x102A0,0x102D1)),"NotoSansCarian"),
    "Caucasian Albanian": (list(range(0x10530,0x10564)),"NotoSansCaucasianAlbanian"),
    "Chorasmian": (list(range(0x10FB0,0x10FC5)),"NotoSansChorasmian"),
    "Coptic": ([0x2c80,0x2c82,0x2c84,0x2c86,0x2c88,0x2c8a,0x2c8c,0x2c8e,0x2c90,0x2c92,0x2c94,0x2c96,0x2c98,0x2c9a,0x2c9c,0x2c9e,0x2ca0,0x2ca2,0x2ca4,0x2ca6,0x2ca8,0x2caa,0x2cac,0x2cae,0x2cb0],"NotoSansCoptic"),
    "Cypriot": (list(range(0x10800,0x10806))+list(range(0x1080A,0x10836))+[0x10808,0x10837,0x10838,0x1083C,0x1083F],"NotoSansCypriot"),
    "Cypro-Minoan": (list(range(0x012F90,0x012FF1)),"NotoSansCyproMinoan"),
    "Cyrillic": (list(range(0x00410,0x00430)),"NotoSans"),
    "Elder Futhark": ([0x016A0,0x016A2,0x016A6,0x016A8,0x016B1,0x016B2,0x016B7,0x016B9,0x016BA,0x016BE,0x016C1,0x016C3,0x016C7,0x016C8,0x016C9,0x016CA,0x016CF,0x016D2,0x016D6,0x016D7,0x016DA,0x016DC,0x016DE,0x016DF],"NotoSansRunic"),
    "Elymaic": (list(range(0x10FE0, 0x10FF6)),"NotoSansElymaic"),
    "Ge_ez": ([0x01200,0x01208,0x01210,0x01218,0x01220,0x01228,0x01230,0x01240,0x01260,0x01270,0x01280,0x01290,0x012A0,0x012A8,0x012C8,0x012D0,0x012D8,0x012E8,0x012F0,0x01308,0x01320,0x01330,0x01338,0x01340,0x01348,0x01350],"NotoSansEthiopic"),
    "Georgian Asomtavruli": (list(range(0x10A0,0x10C6)),"NotoSansGeorgian"),
    "Georgian Mkhedruli": (list(range(0x010D0,0x10F1)),"NotoSansGeorgian"),
    "Glagolitic": (list(range(0x02C00,0x02C2F)),"NotoSansGlagolitic"),
    "Gothic": (list(range(0x10330,0x1034B)),"NotoSansGothic"),
    "Greek": (list(range(0x00391,0x003A2))+list(range(0x003A3,0x003AA)),"NotoSans"),
    "Hatran Aramaic": (list(range(0x108E0,0x108F3))+list(range(0x108F4,0x108F6)),"NotoSansHatran"),
    "Hebrew": (list(range(0x005D0,0x005EB)),"NotoSansHebrew"),
    "Imperial Aramaic": (list(range(0x10840,0x10856)),"NotoSansImperialAramaic"),
    "Kharoshthi": ([0x10A00]+list(range(0x10A10,0x10A14))+list(range(0x10A15,0x10A18))+list(range(0x10A19,0x10A36)),"NotoSansKharoshthi"),
    "Latin": (list(range(0x00041,0x0005B)),"NotoSans"),
    "Linear B": (list(range(0x10000,0x1000C))+list(range(0x1000D,0x10027))+list(range(0x10028,0x1003B))+list(range(0x1003C,0x1003E))+list(range(0x1003F,0x10040)),"NotoSansLinearB"),
    "Lycian": (list(range(0x10280,0x1029D)),"NotoSansLycian"),
    "Lydian": (list(range(0x10920,0x1093A)),"NotoSansLydian"),
    "Mandaic": (list(range(0x0840,0x0859)),"NotoSansMandaic"),
    "Manichaean": (list(range(0x10AC0,0x10AC8))+list(range(0x10AC9,0x10AE5)),"NotoSansManichaean"),
    "Meroitic Cursive": (list(range(0x109A0,0x109B8)),"NotoSansMeroitic"),
    "Meroitic Hieroglyphs": (list(range(0x10980,0x1099E)),"NotoSansMeroitic"),
    "Nabataean": (list(range(0x10880,0x1089F)),"NotoSansNabataean"),
    "Ogham": (list(range(0x1681,0x1695)),"NotoSansOgham"),
    "Old Hungarian": (list(range(0x10C80,0x10CB3)),"NotoSansOldHungarian"),
    "Old Italic": (list(range(0x10300,0x1031B)),"NotoSansOldItalic"),
    "Old North Arabian": (list(range(0x10A80,0x10A9D)),"NotoSansOldNorthArabian"),
    "Old Permic": (list(range(0x10350,0x10376)),"NotoSansOldPermic"),
    "Old Persian": (list(range(0x103A0,0x103C4)),"NotoSansOldPersian"),
    "Old Sogdian": ([0x10F00,0x10F02,0x10F04,0x10F05,0x10F07,0x10F08,0x10F09,0x10F0A,0x10F0B,0x10F0C,0x10F0D,0x10F0E,0x10F11,0x10F14,0x10F15,0x10F18,0x10F19,0x10F1A],"NotoSansOldSogdian"),
    "Old South Arabian": (list(range(0x10A60,0x10A7D)),"NotoSansOldSouthArabian"),
    "Old Turkic Orkhon": ([0x10c00,0x10c03,0x10c06,0x10c07,0x10c09,0x10c0B,0x10c0D,0x10c0F,0x10c11,0x10c13,0x10c14,0x10c16,0x10c18,0x10c1A,0x10c1C,0x10c1E,0x10c20,0x10c21,0x10c22,0x10c23,0x10c24,0x10c26,0x10c28,0x10c2a,0x10c2d,0x10c2f,0x10c30,0x10c31,0x10c32,0x10c34,0x10c36,0x10c38,0x10c3a,0x10c3c,0x10c3d,0x10c3e,0x10c3f,0x10c41,0x10c43,0x10c45,0x10c47,0x10c48],"NotoSansOldTurkic"),
    "Old Turkic Yenisei": ([0x10c01,0x10c02,0x10c04,0x10c05,0x10c08,0x10c0a,0x10c0c,0x10c0e,0x10c10,0x10c12,0x10c15,0x10c17,0x10c19,0x10c1b,0x10c1d,0x10c1f,0x10c25,0x10c27,0x10c29,0x10c2b,0x10c2c,0x10c2e,0x10c33,0x10c35,0x10c37,0x10c39,0x10c3b,0x10c40,0x10c42,0x10c44,0x10c46],"NotoSansOldTurkic"),
    "Pahlavi Inscriptional": (list(range(0x10B60,0x10B73)),"NotoSansInscriptionalPahlavi"),
    "Pahlavi Psalter": (list(range(0x10b80,0x10b92)),"NotoSansPsalterPahlavi"),
    "Palmyrene": (list(range(0x10860,0x10877)),"NotoSansPalmyrene"),
    "Parthian Inscriptional": (list(range(0x10B40,0x10B56)),"NotoSansInscriptionalParthian"),
    "Phoenician": (list(range(0x10900,0x10916)),"NotoSansPhoenician"),
    "Samaritan": (list(range(0x0800,0x0816)),"NotoSansSamaritan"),
    "Sogdian": (list(range(0x10F30,0x10F45)),"NotoSansSogdian"),
    "Syriac": ([0x0710,0x0712,0x0713,0x0715]+list(range(0x0717,0x071C))+list(range(0x071D,0x0724))+list(range(0x0725,0x0727))+list(range(0x0728,0x0730)),"NotoSansSyriac"),
    "Tifinagh": ([0x2D30,0x2D31,0x2D33,0x2D37,0x2D39,0x2D3B,0x2D3C,0x2D3D,0x2D40,0x2D43,0x2D44,0x2D45,0x2D47,0x2D49,0x2D4A,0x2D4D,0x2D4E,0x2D4F,0x2D53,0x2D54,0x2D55,0x2D56,0x2D59,0x2D5A,0x2D5B,0x2D5C,0x2D5F,0x2D61,0x2D62,0x2D63,0x2D65],"NotoSansTifinagh"),
    "Ugaritic": (list(range(0x10380,0x1039E)),"NotoSansUgaritic"),
}
