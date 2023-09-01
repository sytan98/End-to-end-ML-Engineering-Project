from enum import StrEnum

from pydantic import BaseModel


class Town(StrEnum):
    ANG_MO_KIO = "ANG MO KIO"
    BEDOK = "BEDOK"
    BISHAN = "BISHAN"
    BUKIT_BATOK = "BUKIT BATOK"
    BUKIT_MERAH = "BUKIT MERAH"
    BUKIT_PANJANG = "BUKIT PANJANG"
    BUKIT_TIMAH = "BUKIT TIMAH"
    CENTRAL_AREA = "CENTRAL AREA"
    CHOA_CHU_KANG = "CHOA CHU KANG"
    CLEMENTI = "CLEMENTI"
    GEYLANG = "GEYLANG"
    HOUGANG = "HOUGANG"
    JURONG_EAST = "JURONG EAST"
    JURONG_WEST = "JURONG WEST"
    KALLANG_WHAMPOA = "KALLANG/WHAMPOA"
    MARINE_PARADE = "MARINE PARADE"
    PASIR_RIS = "PASIR RIS"
    PUNGGOL = "PUNGGOL"
    QUEENSTOWN = "QUEENSTOWN"
    SEMBAWANG = "SEMBAWANG"
    SENGKANG = "SENGKANG"
    SERANGOON = "SERANGOON"
    TAMPINES = "TAMPINES"
    TOA_PAYOH = "TOA PAYOH"
    WOODLANDS = "WOODLANDS"
    YISHUN = "YISHUN"


class StoreyRange(StrEnum):
    RANGE_1_TO_3 = "01 TO 03"
    RANGE_4_TO_6 = "04 TO 06"
    RANGE_7_TO_9 = "07 TO 09"
    RANGE_10_TO_12 = "10 TO 12"
    RANGE_13_TO_15 = "13 TO 15"
    RANGE_16_TO_18 = "16 TO 18"
    RANGE_19_TO_21 = "19 TO 21"
    RANGE_22_TO_24 = "22 TO 24"
    RANGE_25_TO_27 = "25 TO 27"
    RANGE_28_TO_30 = "28 TO 30"
    RANGE_31_TO_33 = "31 TO 33"
    RANGE_34_TO_36 = "34 TO 36"
    RANGE_37_TO_39 = "37 TO 39"
    RANGE_40_TO_42 = "40 TO 42"
    RANGE_43_TO_45 = "43 TO 45"
    RANGE_46_TO_48 = "46 TO 48"
    RANGE_49_TO_51 = "49 TO 51"


class FlatType(StrEnum):
    THREE_ROOM = "3 ROOM"
    FOUR_ROOM = "4 ROOM"
    FIVE_ROOM = "5 ROOM"
    TWO_ROOM = "2 ROOM"
    EXECUTIVE = "EXECUTIVE"
    ONE_ROOM = "1 ROOM"
    MULTI_GENERATION = "MULTI-GENERATION"


class FlatModel(StrEnum):
    IMPROVED = "Improved"
    NEW_GEN = "New Generation"
    MODEL_A = "Model A"
    STANDARD = "Standard"
    SIMPLIFIED = "Simplified"
    PREMIUM_APARTMENT = "Premium Apartment"
    MAISONETTE = "Maisonette"
    APARTMENT = "Apartment"
    MODEL_A2 = "Model A2"
    TYPE_S1 = "Type S1"
    TYPE_S2 = "Type S2"
    ADJOINED_FLAT = "Adjoined flat"
    TERRACE = "Terrace"
    DBSS = "DBSS"
    MODEL_A_MAISONETTE = "Model A-Maisonette"
    PREMIUM_MAISONETTE = "Premium Maisonette"
    MULTI_GENERATION = "Multi Generation"
    PREMIUM_APARTMENT_LOFT = "Premium Apartment Loft"
    IMPROVED_MAISONETTE = "Improved-Maisonette"
    TWO_ROOM = "2-room"


class HouseDetails(BaseModel):
    address: str
    town: Town
    flat_type: FlatType
    storey_range: StoreyRange
    floor_area_sqm: float
    flat_model: FlatModel 
    remaining_lease: int
