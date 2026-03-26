"""
Synthetic Medical Claims Data Generator for Kenyan Healthcare Facilities
WITH FRAUD LAYER (20%) AND MULTI-VISIT PATIENTS (UP TO 10 VISITS)
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from collections import defaultdict

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

class Config:
    """Configuration parameters for data generation"""
    
    # Dataset size - USER CAN MODIFY THIS
    TOTAL_RECORDS = 20  # Change this to desired number
    
    # Data integrity layers
    REAL_DATA_PERCENT = 0.80    # 80% real data (follows ranges)
    FRAUD_DATA_PERCENT = 0.20   # 20% fraud data (outside ranges)
    
    # Disease distribution (equal 20% each)
    DISEASE_DISTRIBUTION = {
        'PNEUMONIA': 0.20,
        'ASTHMA': 0.20,
        'TBI': 0.20,
        'APH_PPH': 0.20,
        'PUERPERAL_SEPSIS': 0.20
    }
    
    # Patient status distribution (within REAL data)
    NORMAL_PERCENT = 0.30
    INFECTED_PERCENT = 0.70
    
    # Gender distribution
    MALE_PERCENT = 0.50
    FEMALE_PERCENT = 0.50
    
    # Age groups distribution (equal 25% each)
    AGE_GROUPS = {
        'Pediatric': (0, 5),
        'Transition': (6, 17),
        'Adult': (18, 64),
        'Geriatric': (65, 100)
    }
    AGE_GROUP_PERCENT = 0.25  # Each group gets 25%
    
    # Pneumonia sub-types distribution
    PNEUMONIA_SUBTYPES = {
        'Bacterial': 1/3,
        'Viral': 1/3,
        'Atypical': 1/3
    }
    
    # Patient visit limits
    MAX_VISITS_PER_PATIENT = 10
    MIN_VISITS_PER_PATIENT = 1
    
    # Date range
    START_DATE = datetime(2022, 2, 1)
    END_DATE = datetime(2026, 2, 28)
    
    # Random seed for reproducibility
    RANDOM_SEED = 42


# =============================================================================
# KENYAN FACILITIES DATA
# =============================================================================

class KenyanFacilities:
    """200 Kenyan healthcare facilities with levels and types"""
    
    FACILITIES = [
        # Level 6 - National Referral (5)
        {'name': 'Kenyatta National Hospital', 'level': 6, 'type': 'National Referral', 'county': 'Nairobi'},
        {'name': 'Moi Teaching & Referral Hospital', 'level': 6, 'type': 'National Referral', 'county': 'Uasin Gishu'},
        {'name': 'Mathari National Teaching & Referral', 'level': 6, 'type': 'National Referral', 'county': 'Nairobi'},
        {'name': 'National Spinal Injury Hospital', 'level': 6, 'type': 'National Referral', 'county': 'Nairobi'},
        {'name': 'Kenyatta University Teaching & Referral', 'level': 6, 'type': 'National Referral', 'county': 'Kiambu'},
        
        # Level 5 - County Referral (15)
        {'name': 'Coast General Teaching & Referral', 'level': 5, 'type': 'County Referral', 'county': 'Mombasa'},
        {'name': 'Nakuru Level 5 Hospital', 'level': 5, 'type': 'County Referral', 'county': 'Nakuru'},
        {'name': 'Kisumu County Hospital', 'level': 5, 'type': 'County Referral', 'county': 'Kisumu'},
        {'name': 'Kakamega County Referral', 'level': 5, 'type': 'County Referral', 'county': 'Kakamega'},
        {'name': 'Meru Level 5 Hospital', 'level': 5, 'type': 'County Referral', 'county': 'Meru'},
        {'name': 'Machakos Level 5 Hospital', 'level': 5, 'type': 'County Referral', 'county': 'Machakos'},
        {'name': 'Kisii Level 5 Hospital', 'level': 5, 'type': 'County Referral', 'county': 'Kisii'},
        {'name': 'Nyeri County Referral', 'level': 5, 'type': 'County Referral', 'county': 'Nyeri'},
        {'name': 'Garissa County Referral', 'level': 5, 'type': 'County Referral', 'county': 'Garissa'},
        {'name': 'Embu Level 5 Hospital', 'level': 5, 'type': 'County Referral', 'county': 'Embu'},
        {'name': 'Kitale County Hospital', 'level': 5, 'type': 'County Referral', 'county': 'Trans Nzoia'},
        {'name': 'Voi County Hospital', 'level': 5, 'type': 'County Referral', 'county': 'Taita Taveta'},
        {'name': 'Malindi Sub-county Hospital', 'level': 5, 'type': 'County Referral', 'county': 'Kilifi'},
        {'name': 'Lodwar County Referral', 'level': 5, 'type': 'County Referral', 'county': 'Turkana'},
        {'name': 'Wajir County Referral', 'level': 5, 'type': 'County Referral', 'county': 'Wajir'},
        
        # Level 4 - Primary Referral (40) - Sample
        {'name': 'Thika Level 4 Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Kiambu'},
        {'name': 'Naivasha Sub-county Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Nakuru'},
        {'name': 'Kilifi County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Kilifi'},
        {'name': 'Msambweni County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Kwale'},
        {'name': 'Homa Bay County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Homa Bay'},
        {'name': 'Migori County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Migori'},
        {'name': 'Siaya County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Siaya'},
        {'name': 'Busia County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Busia'},
        {'name': 'Bungoma County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Bungoma'},
        {'name': 'Vihiga County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Vihiga'},
        {'name': 'Muranga County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Muranga'},
        {'name': 'Kirinyaga County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Kirinyaga'},
        {'name': 'Nyandarua County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Nyandarua'},
        {'name': 'Laikipia County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Laikipia'},
        {'name': 'Samburu County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Samburu'},
        {'name': 'Isiolo County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Isiolo'},
        {'name': 'Marsabit County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Marsabit'},
        {'name': 'Mandera County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Mandera'},
        {'name': 'Lamu County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Lamu'},
        {'name': 'Tana River County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Tana River'},
        {'name': 'Makueni County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Makueni'},
        {'name': 'Kitui County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Kitui'},
        {'name': 'Mwingi Level 4 Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Kitui'},
        {'name': 'Kericho County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Kericho'},
        {'name': 'Bomet County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Bomet'},
        {'name': 'Narok County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Narok'},
        {'name': 'Kajiado County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Kajiado'},
        {'name': 'Transmara Level 4 Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Narok'},
        {'name': 'West Pokot County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'West Pokot'},
        {'name': 'Turkana County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Turkana'},
        {'name': 'Kapenguria County Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'West Pokot'},
        {'name': 'Lugari Sub-county Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Kakamega'},
        {'name': 'Butere Sub-county Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Kakamega'},
        {'name': 'Mbale Sub-county Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Vihiga'},
        {'name': 'Hamisi Sub-county Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Vihiga'},
        {'name': 'Teso Sub-county Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Busia'},
        {'name': 'Samia Sub-county Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Busia'},
        {'name': 'Rachuonyo Sub-county Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Homa Bay'},
        {'name': 'Suba Sub-county Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Homa Bay'},
        {'name': 'Kuria Sub-county Hospital', 'level': 4, 'type': 'Primary Referral', 'county': 'Migori'},
        
        # Level 3 - Health Centres (70) - Sample
        *[{'name': f'{name} Health Centre', 'level': 3, 'type': 'Health Centre', 'county': county} 
          for name, county in zip(
              ['Ruai', 'Kayole', 'Embakasi', 'Langata', 'Dagoretti', 'Kibera', 'Mathare', 'Korogocho', 
               'Mukuru', 'Viwandani', 'Baba Dogo', 'Karlobangi', 'Donholm', 'Buruburu', 'Umoja', 
               'Kasarani', 'Mwiki', 'Githurai', 'Zimmerman', 'Roysambu', 'Ruiru', 'Juja', 'Kikuyu', 
               'Limuru', 'Lari', 'Gatundu', 'Githunguri', 'Kiambu', 'Kabete', 'Wangige', 'Kangemi', 
               'Kawangware', 'Waithaka', 'Uthiru', 'Ngong', 'Ongata Rongai', 'Kitengela', 'Athi River', 
               'Mlolongo', 'Syokimau', 'Machakos Town', 'Athi River', 'Kangundo', 'Tala', 'Matuu', 
               'Masinga', 'Yatta', 'Kathiani', 'Mwala', 'Kalawa', 'Wote', 'Kibwezi', 'Mtito Andei', 
               'Sultan Hamud', 'Email', 'Kimana', 'Loitoktok', 'Namanga', 'Bissil', 'Magadi', 
               'Kiserian', 'Rongai', 'Salgaa', 'Mau Summit', 'Molo', 'Elburgon', 'Turbo', 'Eldoret', 
               'Iten', 'Kapsowar', 'Kacheliba'],
              ['Nairobi', 'Nairobi', 'Nairobi', 'Nairobi', 'Nairobi', 'Nairobi', 'Nairobi', 'Nairobi',
               'Nairobi', 'Nairobi', 'Nairobi', 'Nairobi', 'Nairobi', 'Nairobi', 'Nairobi', 'Nairobi',
               'Nairobi', 'Nairobi', 'Nairobi', 'Nairobi', 'Kiambu', 'Kiambu', 'Kiambu', 'Kiambu',
               'Kiambu', 'Kiambu', 'Kiambu', 'Kiambu', 'Kiambu', 'Kiambu', 'Nairobi', 'Nairobi',
               'Nairobi', 'Nairobi', 'Kajiado', 'Kajiado', 'Kajiado', 'Machakos', 'Machakos', 'Machakos',
               'Machakos', 'Machakos', 'Machakos', 'Machakos', 'Machakos', 'Machakos', 'Machakos',
               'Makueni', 'Makueni', 'Makueni', 'Makueni', 'Makueni', 'Makueni', 'Kajiado', 'Kajiado',
               'Kajiado', 'Kajiado', 'Kajiado', 'Nakuru', 'Nakuru', 'Nakuru', 'Nakuru', 'Nakuru',
               'Uasin Gishu', 'Uasin Gishu', 'Elgeyo Marakwet', 'Elgeyo Marakwet', 'West Pokot'])],
        
        # Level 2 - Dispensaries (70) - Sample
        *[{'name': f'{name} Dispensary', 'level': 2, 'type': 'Dispensary', 'county': county}
          for name, county in zip(
              ['Gatanga', 'Kandara', 'Kigumo', 'Kangari', 'Ihururu', 'Kiharu', 'Mukurweini', 'Othaya',
               'Mathira', 'Tetu', 'Kieni', 'Naro Moru', 'Nyahururu', 'Ol Kalou', 'Kinamba', 'Githunguri',
               'Komothai', 'Kirenga', 'Gatundu South', 'Gatundu North', 'Juja Farm', 'Kalimoni', 
               'Mangu', 'Kamburu', 'Kivaa', 'Makutano', 'Kyuso', 'Tseikuru', 'Mutomo', 'Ikutha',
               'Kanziko', 'Mbitini', 'Kisau', 'Mbooni', 'Kilungu', 'Kaani', 'Makindu', 'Nguu',
               'Masongaleni', 'Mtito Andei', 'Kinyambu', 'Kathekai', 'Kikima', 'Kyambeke', 'Wamunyu',
               'Kalia', 'Muusini', 'Kisasi', 'Zombe', 'Kanyangi', 'Mwitika', 'Mumoni', 'Endau',
               'Mavindini', 'Mivukoni', 'Mtama', 'Ndoleli', 'Bura', 'Taru', 'Mackinnon', 'Maungu',
               'Bachuma', 'Sagala', 'Wundanyi', 'Mbale', 'Mwatate', 'Taveta', 'Njukini', 'Chala',
               'Kishushe', 'Rong''e'],
              ['Muranga', 'Muranga', 'Muranga', 'Muranga', 'Muranga', 'Muranga', 'Nyeri', 'Nyeri',
               'Nyeri', 'Nyeri', 'Nyeri', 'Nyeri', 'Nyandarua', 'Nyandarua', 'Nyandarua', 'Kiambu',
               'Kiambu', 'Kiambu', 'Kiambu', 'Kiambu', 'Kiambu', 'Kiambu', 'Tharaka Nithi', 'Tharaka Nithi',
               'Tharaka Nithi', 'Kitui', 'Kitui', 'Kitui', 'Kitui', 'Kitui', 'Makueni', 'Makueni',
               'Makueni', 'Makueni', 'Makueni', 'Makueni', 'Makueni', 'Makueni', 'Makueni', 'Makueni',
               'Makueni', 'Makueni', 'Makueni', 'Makueni', 'Machakos', 'Machakos', 'Machakos', 'Machakos',
               'Machakos', 'Machakos', 'Kitui', 'Kitui', 'Kitui', 'Kitui', 'Kitui', 'Kitui', 'Tana River',
               'Tana River', 'Kwale', 'Kwale', 'Kwale', 'Kwale', 'Taita Taveta', 'Taita Taveta', 'Taita Taveta',
               'Taita Taveta', 'Taita Taveta', 'Taita Taveta', 'Taita Taveta', 'Taita Taveta'])]
    ]
    
    @classmethod
    def get_random_facility(cls):
        """Return random facility from list"""
        return random.choice(cls.FACILITIES)


# =============================================================================
# ICD-10 CODES AND PROCEDURES
# =============================================================================

DISEASE_INFO = {
    'PNEUMONIA': {
        'category': 'respiratory',  # UPDATED: lowercase, no hyphen
        'icd10': 'J18.9',
        'procedures': ['Chest X-ray', 'Bronchoscopy', 'Oxygen therapy', 'IV antibiotics', 
                       'Chest physiotherapy', 'Pleural tap', 'Blood culture', 'Sputum culture']
    },
    'ASTHMA': {
        'category': 'respiratory',  # UPDATED: lowercase, no hyphen
        'icd10': 'J45.9',
        'procedures': ['Spirometry', 'Nebulization', 'Allergy testing', 'Corticosteroids', 
                       'Peak flow monitoring', 'Chest X-ray', 'Oxygen therapy', 'Bronchodilator therapy']
    },
    'TBI': {
        'category': 'trauma',  # UPDATED: lowercase, no hyphen
        'icd10': 'S06.9',
        'procedures': ['CT Scan', 'ICP monitoring', 'Craniotomy', 'Ventilator support', 
                       'Neurological assessment', 'Skull X-ray', 'MRI', 'Intracranial pressure bolt insertion']
    },
    'APH_PPH': {
        'category': 'obstetric',  # UPDATED: lowercase, no hyphen
        'icd10': 'O46.9/O72.0',
        'procedures': ['C-section', 'Manual placenta removal', 'Hysterectomy', 'Blood transfusion', 
                       'Uterine artery ligation', 'Bakri balloon insertion', 'Uterine massage', 
                       'IV fluids', 'Emergency resuscitation']
    },
    'PUERPERAL_SEPSIS': {
        'category': 'obstetric',  # UPDATED: lowercase, no hyphen
        'icd10': 'O85',
        'procedures': ['Wound debridement', 'Antibiotic therapy', 'D&C', 'ICU admission', 
                       'Blood culture', 'Septic workup', 'IV antibiotics', 'Uterine evacuation']
    }
}


# =============================================================================
# LENGTH OF STAY BY DISEASE
# =============================================================================

LENGTH_OF_STAY = {
    'PNEUMONIA': {'min': 3, 'max': 10},
    'ASTHMA': {'min': 1, 'max': 5},
    'TBI': {'min': 5, 'max': 21},
    'APH_PPH': {'min': 2, 'max': 7},
    'PUERPERAL_SEPSIS': {'min': 4, 'max': 14}
}


# =============================================================================
# CBC REFERENCE RANGES - EXACT FROM EXCEL FILES
# =============================================================================

class CBCRanges:
    """Complete CBC ranges by disease, status, age group, and gender"""
    
    # =========================================================================
    # NORMAL REFERENCE RANGES (Baseline)
    # =========================================================================
    
    @staticmethod
    def get_normal_range(parameter, age_group, sex):
        """Get normal reference range for parameter"""
        
        ranges = {
            'HGB': {
                'Pediatric': {'Male': (11.0, 13.5), 'Female': (11.0, 13.5)},
                'Transition': {'Male': (12.0, 15.0), 'Female': (11.5, 14.5)},
                'Adult': {'Male': (13.5, 17.5), 'Female': (12.0, 15.5)},
                'Geriatric': {'Male': (12.5, 16.5), 'Female': (11.5, 15.0)}
            },
            'HCT': {
                'Pediatric': {'Male': (33, 40), 'Female': (33, 40)},
                'Transition': {'Male': (36, 44), 'Female': (35, 43)},
                'Adult': {'Male': (41, 53), 'Female': (36, 46)},
                'Geriatric': {'Male': (38, 50), 'Female': (34, 45)}
            },
            'MCV': {
                'Pediatric': {'Male': (70, 86), 'Female': (70, 86)},
                'Transition': {'Male': (75, 90), 'Female': (75, 90)},
                'Adult': {'Male': (80, 100), 'Female': (80, 100)},
                'Geriatric': {'Male': (80, 102), 'Female': (80, 102)}
            },
            'MCHC': {
                'Pediatric': {'Male': (32, 36), 'Female': (32, 36)},
                'Transition': {'Male': (32, 36), 'Female': (32, 36)},
                'Adult': {'Male': (32, 36), 'Female': (32, 36)},
                'Geriatric': {'Male': (31, 36), 'Female': (31, 36)}
            },
            'NEU': {
                'Pediatric': {'Male': (30, 55), 'Female': (30, 55)},
                'Transition': {'Male': (40, 65), 'Female': (40, 65)},
                'Adult': {'Male': (40, 60), 'Female': (40, 60)},
                'Geriatric': {'Male': (45, 70), 'Female': (45, 70)}
            },
            'LYM': {
                'Pediatric': {'Male': (35, 60), 'Female': (35, 60)},
                'Transition': {'Male': (25, 45), 'Female': (25, 45)},
                'Adult': {'Male': (20, 40), 'Female': (20, 40)},
                'Geriatric': {'Male': (18, 38), 'Female': (18, 38)}
            },
            'EOS': {
                'Pediatric': {'Male': (1, 4), 'Female': (1, 4)},
                'Transition': {'Male': (1, 4), 'Female': (1, 4)},
                'Adult': {'Male': (1, 4), 'Female': (1, 4)},
                'Geriatric': {'Male': (1, 5), 'Female': (1, 5)}
            },
            'BAS': {
                'Pediatric': {'Male': (0, 1), 'Female': (0, 1)},
                'Transition': {'Male': (0, 1), 'Female': (0, 1)},
                'Adult': {'Male': (0.5, 1), 'Female': (0.5, 1)},
                'Geriatric': {'Male': (0, 1), 'Female': (0, 1)}
            },
            'MON': {
                'Pediatric': {'Male': (3, 8), 'Female': (3, 8)},
                'Transition': {'Male': (3, 9), 'Female': (3, 9)},
                'Adult': {'Male': (2, 8), 'Female': (2, 8)},
                'Geriatric': {'Male': (3, 10), 'Female': (3, 10)}
            },
            'PLT': {
                'Pediatric': {'Male': (150, 450), 'Female': (150, 450)},
                'Transition': {'Male': (150, 450), 'Female': (150, 450)},
                'Adult': {'Male': (150, 400), 'Female': (150, 400)},
                'Geriatric': {'Male': (150, 420), 'Female': (150, 420)}
            }
        }
        
        return ranges.get(parameter, {}).get(age_group, {}).get(sex, (0, 0))
    
    # =========================================================================
    # PNEUMONIA - INFECTED RANGES
    # =========================================================================
    
    @staticmethod
    def get_pneumonia_range(parameter, age_group, sex, subtype):
        """Get pneumonia infected range by subtype"""
        
        # Bacterial ranges
        bacterial = {
            'HGB': {
                'Pediatric': {'Male': (10.5, 13.0), 'Female': (10.0, 13.0)},
                'Transition': {'Male': (11.0, 14.0), 'Female': (10.5, 14.0)},
                'Adult': {'Male': (12.0, 16.0), 'Female': (11.5, 15.0)},
                'Geriatric': {'Male': (11.0, 15.0), 'Female': (10.5, 14.0)}
            },
            'HCT': {
                'Pediatric': {'Male': (32, 38), 'Female': (31, 38)},
                'Transition': {'Male': (34, 42), 'Female': (33, 41)},
                'Adult': {'Male': (38, 50), 'Female': (35, 45)},
                'Geriatric': {'Male': (35, 46), 'Female': (33, 43)}
            },
            'MCV': {
                'Pediatric': {'Male': (70, 86), 'Female': (70, 86)},
                'Transition': {'Male': (75, 90), 'Female': (75, 90)},
                'Adult': {'Male': (80, 100), 'Female': (80, 100)},
                'Geriatric': {'Male': (82, 102), 'Female': (82, 102)}
            },
            'MCHC': {
                'Pediatric': {'Male': (31, 35), 'Female': (31, 35)},
                'Transition': {'Male': (31, 35), 'Female': (31, 35)},
                'Adult': {'Male': (31, 36), 'Female': (31, 36)},
                'Geriatric': {'Male': (31, 35), 'Female': (31, 35)}
            },
            'NEU': {
                'Pediatric': {'Male': (65, 85), 'Female': (65, 85)},
                'Transition': {'Male': (70, 88), 'Female': (70, 88)},
                'Adult': {'Male': (75, 90), 'Female': (75, 90)},
                'Geriatric': {'Male': (75, 90), 'Female': (75, 90)}
            },
            'LYM': {
                'Pediatric': {'Male': (10, 25), 'Female': (10, 25)},
                'Transition': {'Male': (8, 22), 'Female': (8, 22)},
                'Adult': {'Male': (5, 20), 'Female': (5, 20)},
                'Geriatric': {'Male': (5, 18), 'Female': (5, 18)}
            },
            'EOS': {
                'All': {'Male': (0, 2), 'Female': (0, 2)}
            },
            'BAS': {
                'All': {'Male': (0, 1), 'Female': (0, 1)}
            },
            'MON': {
                'Pediatric': {'Male': (5, 10), 'Female': (5, 10)},
                'Transition': {'Male': (4, 10), 'Female': (4, 10)},
                'Adult': {'Male': (3, 9), 'Female': (3, 9)},
                'Geriatric': {'Male': (4, 10), 'Female': (4, 10)}
            },
            'PLT': {
                'Pediatric': {'Male': (200, 500), 'Female': (200, 500)},
                'Transition': {'Male': (200, 520), 'Female': (200, 520)},
                'Adult': {'Male': (220, 550), 'Female': (220, 520)},
                'Geriatric': {'Male': (220, 520), 'Female': (220, 500)}
            }
        }
        
        # Viral ranges (differences only)
        viral = {
            'HGB': bacterial['HGB'],
            'HCT': {
                'Pediatric': {'Male': (32, 38), 'Female': (31, 38)},
                'Transition': {'Male': (34, 42), 'Female': (33, 41)},
                'Adult': {'Male': (38, 48), 'Female': (35, 45)},
                'Geriatric': {'Male': (35, 45), 'Female': (33, 43)}
            },
            'NEU': {
                'Pediatric': {'Male': (20, 40), 'Female': (20, 40)},
                'Transition': {'Male': (25, 45), 'Female': (25, 45)},
                'Adult': {'Male': (30, 55), 'Female': (30, 55)},
                'Geriatric': {'Male': (35, 55), 'Female': (35, 55)}
            },
            'LYM': {
                'Pediatric': {'Male': (50, 70), 'Female': (50, 70)},
                'Transition': {'Male': (45, 65), 'Female': (45, 65)},
                'Adult': {'Male': (40, 65), 'Female': (40, 65)},
                'Geriatric': {'Male': (35, 60), 'Female': (35, 60)}
            },
            'MON': {
                'All': {'Male': (4, 10), 'Female': (4, 10)}
            },
            'PLT': {
                'All': {'Male': (150, 420), 'Female': (150, 420)}
            }
        }
        
        # Atypical ranges (differences only)
        atypical = {
            'HGB': {
                'Pediatric': {'Male': (10.5, 13.0), 'Female': (10.0, 13.0)},
                'Transition': {'Male': (11.0, 14.5), 'Female': (10.5, 14.0)},
                'Adult': {'Male': (12.0, 16.0), 'Female': (11.5, 15.0)},
                'Geriatric': {'Male': (11.0, 15.0), 'Female': (10.5, 14.0)}
            },
            'NEU': {
                'All': {'Male': (40, 60), 'Female': (40, 60)}
            },
            'LYM': {
                'All': {'Male': (30, 50), 'Female': (30, 50)}
            },
            'MON': {
                'Pediatric': {'Male': (8, 14), 'Female': (8, 14)},
                'Transition': {'Male': (8, 15), 'Female': (8, 15)},
                'Adult': {'Male': (8, 15), 'Female': (8, 15)},
                'Geriatric': {'Male': (9, 16), 'Female': (9, 16)}
            },
            'PLT': {
                'All': {'Male': (180, 460), 'Female': (180, 460)}
            }
        }
        
        # Fill missing viral/atypical from bacterial
        for param in ['MCV', 'MCHC', 'EOS', 'BAS']:
            if param not in viral:
                viral[param] = bacterial.get(param, {})
            if param not in atypical:
                atypical[param] = bacterial.get(param, {})
        
        ranges = {
            'Bacterial': bacterial,
            'Viral': viral,
            'Atypical': atypical
        }
        
        subtype_ranges = ranges.get(subtype, {})
        param_range = subtype_ranges.get(parameter, {})
        
        if 'All' in param_range:
            return param_range['All'].get(sex, (0, 0))
        
        age_specific = param_range.get(age_group, {})
        if age_specific:
            return age_specific.get(sex, (0, 0))
        
        return (0, 0)
    
    # =========================================================================
    # ASTHMA - INFECTED RANGES
    # =========================================================================
    
    @staticmethod
    def get_asthma_range(parameter, age_group, sex):
        """Get asthma infected range"""
        
        ranges = {
            'HGB': {
                'Pediatric': {'Male': (10.8, 13.5), 'Female': (10.8, 13.5)},
                'Transition': {'Male': (11.8, 15.0), 'Female': (11.2, 14.5)},
                'Adult': {'Male': (13.0, 17.0), 'Female': (11.5, 15.5)},
                'Geriatric': {'Male': (12.0, 15.5), 'Female': (11.0, 14.5)}
            },
            'HCT': {
                'Pediatric': {'Male': (32, 40), 'Female': (32, 40)},
                'Transition': {'Male': (35, 46), 'Female': (34, 45)},
                'Adult': {'Male': (40, 52), 'Female': (35, 45)},
                'Geriatric': {'Male': (37, 48), 'Female': (33, 44)}
            },
            'MCV': {
                'Pediatric': {'Male': (70, 86), 'Female': (70, 86)},
                'Transition': {'Male': (78, 92), 'Female': (78, 92)},
                'Adult': {'Male': (80, 100), 'Female': (80, 100)},
                'Geriatric': {'Male': (82, 102), 'Female': (82, 102)}
            },
            'MCHC': {
                'Pediatric': {'Male': (31, 36), 'Female': (31, 36)},
                'Transition': {'Male': (31, 36), 'Female': (31, 36)},
                'Adult': {'Male': (31, 36), 'Female': (31, 36)},
                'Geriatric': {'Male': (30, 36), 'Female': (31, 36)}
            },
            'NEU': {
                'Pediatric': {'Male': (40, 60), 'Female': (40, 60)},
                'Transition': {'Male': (45, 65), 'Female': (45, 65)},
                'Adult': {'Male': (45, 75), 'Female': (45, 75)},
                'Geriatric': {'Male': (45, 78), 'Female': (45, 78)}
            },
            'LYM': {
                'Pediatric': {'Male': (35, 55), 'Female': (35, 55)},
                'Transition': {'Male': (22, 38), 'Female': (22, 38)},
                'Adult': {'Male': (18, 35), 'Female': (18, 35)},
                'Geriatric': {'Male': (18, 33), 'Female': (18, 33)}
            },
            'EOS': {
                'All': {'Male': (3, 12), 'Female': (3, 12)}
            },
            'BAS': {
                'All': {'Male': (0, 2), 'Female': (0, 2)}
            },
            'MON': {
                'All': {'Male': (4, 12), 'Female': (4, 12)}
            },
            'PLT': {
                'Pediatric': {'Male': (220, 480), 'Female': (220, 480)},
                'Transition': {'Male': (200, 450), 'Female': (200, 450)},
                'Adult': {'Male': (180, 430), 'Female': (180, 450)},
                'Geriatric': {'Male': (180, 420), 'Female': (180, 430)}
            }
        }
        
        param_range = ranges.get(parameter, {})
        
        if 'All' in param_range:
            return param_range['All'].get(sex, (0, 0))
        
        age_specific = param_range.get(age_group, {})
        return age_specific.get(sex, (0, 0))
    
    # =========================================================================
    # TBI - INFECTED RANGES
    # =========================================================================
    
    @staticmethod
    def get_tbi_range(parameter, age_group, sex):
        """Get TBI infected range"""
        
        normal = CBCRanges.get_normal_range(parameter, age_group, sex)
        
        if parameter in ['HGB', 'HCT']:
            return (normal[0] * 0.7, normal[1])
        elif parameter == 'NEU':
            return (normal[0] * 1.2, min(normal[1] * 1.3, 90))
        elif parameter == 'LYM':
            return (normal[0] * 0.5, normal[1] * 0.8)
        elif parameter == 'PLT':
            return (100, 500)
        elif parameter in ['EOS', 'BAS']:
            return (0, normal[1])
        elif parameter == 'MON':
            return (normal[0], normal[1] * 1.2)
        else:
            return normal
    
    # =========================================================================
    # APH/PPH - INFECTED RANGES
    # =========================================================================
    
    @staticmethod
    def get_aph_pph_range(parameter, age_group, sex):
        """Get APH/PPH infected range"""
        
        ranges = {
            'HGB': {
                'Pediatric': {'Male': (8.0, 11.0), 'Female': (8.0, 11.0)},
                'Transition': {'Male': (8.5, 11.5), 'Female': (8.0, 11.0)},
                'Adult': {'Male': (7.0, 11.0), 'Female': (6.0, 10.0)},
                'Geriatric': {'Male': (7.0, 10.5), 'Female': (7.0, 10.0)}
            },
            'HCT': {
                'Pediatric': {'Male': (24, 33), 'Female': (24, 33)},
                'Transition': {'Male': (26, 35), 'Female': (24, 33)},
                'Adult': {'Male': (21, 33), 'Female': (18, 30)},
                'Geriatric': {'Male': (20, 32), 'Female': (20, 32)}
            },
            'MCV': {
                'All': {'Male': (70, 102), 'Female': (70, 102)}
            },
            'MCHC': {
                'All': {'Male': (30, 34), 'Female': (30, 34)}
            },
            'NEU': {
                'Pediatric': {'Male': (55, 70), 'Female': (55, 70)},
                'Transition': {'Male': (55, 70), 'Female': (55, 70)},
                'Adult': {'Male': (55, 75), 'Female': (55, 75)},
                'Geriatric': {'Male': (55, 75), 'Female': (55, 75)}
            },
            'LYM': {
                'Pediatric': {'Male': (20, 35), 'Female': (20, 35)},
                'Transition': {'Male': (20, 35), 'Female': (20, 35)},
                'Adult': {'Male': (15, 30), 'Female': (15, 30)},
                'Geriatric': {'Male': (15, 30), 'Female': (15, 30)}
            },
            'EOS': {
                'All': {'Male': (0, 2), 'Female': (0, 2)}
            },
            'BAS': {
                'All': {'Male': (0, 1), 'Female': (0, 1)}
            },
            'MON': {
                'Pediatric': {'Male': (5, 10), 'Female': (5, 10)},
                'Transition': {'Male': (5, 10), 'Female': (5, 10)},
                'Adult': {'Male': (5, 10), 'Female': (5, 10)},
                'Geriatric': {'Male': (6, 12), 'Female': (6, 12)}
            },
            'PLT': {
                'Pediatric': {'Male': (180, 500), 'Female': (180, 500)},
                'Transition': {'Male': (180, 500), 'Female': (180, 500)},
                'Adult': {'Male': (180, 520), 'Female': (150, 450)},
                'Geriatric': {'Male': (180, 520), 'Female': (150, 450)}
            }
        }
        
        param_range = ranges.get(parameter, {})
        
        if 'All' in param_range:
            return param_range['All'].get(sex, (0, 0))
        
        age_specific = param_range.get(age_group, {})
        return age_specific.get(sex, (0, 0))
    
    # =========================================================================
    # PUERPERAL SEPSIS - INFECTED RANGES
    # =========================================================================
    
    @staticmethod
    def get_puerperal_sepsis_range(parameter, age_group, sex):
        """Get puerperal sepsis infected range"""
        
        if sex == 'Male':
            return CBCRanges.get_normal_range(parameter, age_group, sex)
        
        ranges = {
            'HGB': {'Adult': {'Female': (8.0, 11.5)}},
            'HCT': {'Adult': {'Female': (25, 34)}},
            'MCV': {'Adult': {'Female': (78, 98)}},
            'NEU': {'Adult': {'Female': (30, 35)}},
            'LYM': {'Adult': {'Female': (15, 25)}},
            'BAS': {
                'Transition': {'Female': (0, 1)},
                'Adult': {'Female': (0, 1)}
            },
            'MON': {'Adult': {'Female': (6, 14)}},
            'PLT': {'Adult': {'Female': (70, 250)}}
        }
        
        param_range = ranges.get(parameter, {})
        if not param_range:
            return CBCRanges.get_normal_range(parameter, age_group, sex)
        
        age_specific = param_range.get(age_group, {})
        if age_specific:
            return age_specific.get(sex, (0, 0))
        
        return CBCRanges.get_normal_range(parameter, age_group, sex)
    
    # =========================================================================
    # FRAUD RANGE GENERATOR
    # =========================================================================
    
    @staticmethod
    def get_fraud_range(parameter, age_group, sex, disease, status=None):
        """Generate values outside normal ranges for fraud detection"""
        
        # Get the normal range for this parameter
        normal_range = CBCRanges.get_normal_range(parameter, age_group, sex)
        normal_min, normal_max = normal_range
        
        # Generate values significantly outside normal range
        fraud_multipliers = {
            'HGB': (0.4, 1.6),      # Severely low or high
            'HCT': (0.4, 1.6),
            'MCV': (0.6, 1.4),
            'MCHC': (0.7, 1.3),
            'NEU': (0.3, 2.0),
            'LYM': (0.3, 2.5),
            'EOS': (0, 5.0),         # Can be extremely high
            'BAS': (0, 4.0),
            'MON': (0.3, 3.0),
            'PLT': (0.2, 3.0)        # Very low or very high platelets
        }
        
        if parameter in fraud_multipliers:
            low_mult, high_mult = fraud_multipliers[parameter]
            
            # 50% chance of being too low, 50% chance of being too high
            if random.random() < 0.5:
                # Too low
                return (normal_min * low_mult * 0.8, normal_min * low_mult * 1.2)
            else:
                # Too high
                return (normal_max * high_mult * 0.8, normal_max * high_mult * 1.2)
        
        return normal_range
    
    # =========================================================================
    # DISPATCHER METHOD
    # =========================================================================
    
    @classmethod
    def get_range(cls, disease, status, parameter, age_group, sex, subtype=None, is_fraud=False):
        """Main dispatcher to get appropriate range"""
        
        if is_fraud:
            return cls.get_fraud_range(parameter, age_group, sex, disease, status)
        
        if status == 'Normal':
            return cls.get_normal_range(parameter, age_group, sex)
        
        if disease == 'PNEUMONIA':
            return cls.get_pneumonia_range(parameter, age_group, sex, subtype)
        elif disease == 'ASTHMA':
            return cls.get_asthma_range(parameter, age_group, sex)
        elif disease == 'TBI':
            return cls.get_tbi_range(parameter, age_group, sex)
        elif disease == 'APH_PPH':
            return cls.get_aph_pph_range(parameter, age_group, sex)
        elif disease == 'PUERPERAL_SEPSIS':
            return cls.get_puerperal_sepsis_range(parameter, age_group, sex)
        else:
            return cls.get_normal_range(parameter, age_group, sex)


# =============================================================================
# DATA GENERATION FUNCTIONS
# =============================================================================

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)


def generate_age(age_group):
    """Generate random age within age group"""
    age_min, age_max = Config.AGE_GROUPS[age_group]
    return random.randint(age_min, age_max)


def generate_date():
    """Generate random date between start and end"""
    days_range = (Config.END_DATE - Config.START_DATE).days
    random_days = random.randint(0, days_range)
    return Config.START_DATE + timedelta(days=random_days)


def generate_cbc_value(param_range):
    """Generate random CBC value within range"""
    min_val, max_val = param_range
    if max_val - min_val < 0.1:
        return round(min_val, 1)
    return round(random.uniform(min_val, max_val), 1)


def generate_patient_record(claim_counter, patient_id, patient_visit_num, 
                           disease, status, sex, age_group, 
                           subtype=None, is_fraud=False):
    """Generate a single patient record"""
    
    record = {}
    
    # IDs
    year = datetime.now().year
    record['claim_id'] = f"CLM-{year}-{claim_counter:06d}"
    record['patient_id'] = patient_id
    
    # Demographics
    record['age'] = generate_age(age_group)
    record['sex'] = sex
    
    # Facility
    facility = KenyanFacilities.get_random_facility()
    record['facility_id'] = f"FAC-{random.randint(10000, 99999)}"
    record['facility_name'] = facility['name']
    record['facility_type'] = facility['type']
    record['facility_level'] = facility['level']
    
    # Disease info - UPDATED: category is lowercase without hyphen
    record['disease_category'] = DISEASE_INFO[disease]['category']
    record['diagnosis'] = disease.replace('_', ' ')
    record['diagnosis_code'] = DISEASE_INFO[disease]['icd10']
    
    # Procedure
    record['procedure'] = random.choice(DISEASE_INFO[disease]['procedures'])
    
    # Dates
    admission_date = generate_date()
    los = random.randint(
        LENGTH_OF_STAY[disease]['min'],
        LENGTH_OF_STAY[disease]['max']
    )
    discharge_date = admission_date + timedelta(days=los)
    processed_date = discharge_date + timedelta(hours=random.randint(24, 48))
    
    record['admission_date'] = admission_date.strftime('%Y-%m-%d')
    record['discharge_date'] = discharge_date.strftime('%Y-%m-%d')
    record['date'] = admission_date.strftime('%Y-%m-%d')
    record['timestamp_processed'] = processed_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Fraud indicator (for analysis)
    record['is_fraud'] = 1 if is_fraud else 0
    
    # CBC Parameters
    cbc_params = ['HGB', 'HCT', 'MCV', 'MCHC', 'NEU', 'LYM', 'EOS', 'BAS', 'MON', 'PLT']
    
    for param in cbc_params:
        param_range = CBCRanges.get_range(
            disease=disease,
            status=status,
            parameter=param,
            age_group=age_group,
            sex=sex,
            subtype=subtype,
            is_fraud=is_fraud
        )
        record[param] = generate_cbc_value(param_range)
    
    return record


def generate_patient_pool(total_records):
    """Generate a pool of patients with their visit counts"""
    
    # Calculate number of unique patients needed
    # Average visits around 3-4, so patients = total_records / 3.5
    avg_visits = 3.5
    num_patients = int(total_records / avg_visits)
    
    patients = []
    for i in range(1, num_patients + 1):
        # Random number of visits (1-10)
        num_visits = random.randint(Config.MIN_VISITS_PER_PATIENT, Config.MAX_VISITS_PER_PATIENT)
        
        # Patient demographics (stay consistent across visits)
        sex = 'Male' if random.random() < Config.MALE_PERCENT else 'Female'
        age_group = random.choice(list(Config.AGE_GROUPS.keys()))
        
        patients.append({
            'patient_id': f"PAT-{datetime.now().year}-{i:06d}",
            'sex': sex,
            'age_group': age_group,
            'num_visits': num_visits,
            'visits_generated': 0
        })
    
    return patients


def generate_dataset(total_records=Config.TOTAL_RECORDS):
    """Main function to generate complete dataset with multi-visit patients"""
    
    print(f"Generating {total_records} synthetic medical records...")
    print("=" * 60)
    
    set_random_seed(Config.RANDOM_SEED)
    
    # Calculate fraud vs real split
    fraud_records = int(total_records * Config.FRAUD_DATA_PERCENT)
    real_records = total_records - fraud_records
    
    print(f"Data Integrity Split:")
    print(f"  Real Data: {real_records} records ({Config.REAL_DATA_PERCENT*100:.0f}%)")
    print(f"  Fraud Data: {fraud_records} records ({Config.FRAUD_DATA_PERCENT*100:.0f}%)")
    print("=" * 60)
    
    # Generate patient pool
    patients = generate_patient_pool(real_records)  # Only real patients have multiple visits
    print(f"Created {len(patients)} unique patients (avg visits: {real_records/len(patients):.1f})")
    
    records = []
    claim_counter = 1
    
    # Track counts for reporting
    counts = {
        'total': 0,
        'real': 0,
        'fraud': 0,
        'by_disease': {d: 0 for d in Config.DISEASE_DISTRIBUTION.keys()},
        'by_status': {'Normal': 0, 'Infected': 0},
        'by_sex': {'Male': 0, 'Female': 0},
        'by_age_group': {ag: 0 for ag in Config.AGE_GROUPS.keys()},
        'patient_visits': defaultdict(int)
    }
    
    # Calculate disease targets for real data
    real_disease_targets = {
        disease: int(real_records * pct) 
        for disease, pct in Config.DISEASE_DISTRIBUTION.items()
    }
    
    # Generate REAL patient records (80%)
    print("\nGenerating REAL patient records (multiple visits per patient)...")
    
    # Shuffle patients to randomize order
    random.shuffle(patients)
    
    real_generated = 0
    patient_index = 0
    
    while real_generated < real_records and patient_index < len(patients):
        patient = patients[patient_index]
        
        # Determine how many visits for this patient
        visits_needed = patient['num_visits']
        
        for visit_num in range(visits_needed):
            if real_generated >= real_records:
                break
            
            # Randomly select disease (weighted by remaining targets)
            available_diseases = [d for d, t in real_disease_targets.items() if t > 0]
            if not available_diseases:
                break
            
            # Calculate weights based on remaining targets
            weights = [real_disease_targets[d] for d in available_diseases]
            disease = random.choices(available_diseases, weights=weights)[0]
            
            # Determine status (Normal/Infected) - 30/70 split
            status = 'Normal' if random.random() < Config.NORMAL_PERCENT else 'Infected'
            
            # Determine subtype for pneumonia
            subtype = None
            if disease == 'PNEUMONIA' and status == 'Infected':
                rand = random.random()
                if rand < Config.PNEUMONIA_SUBTYPES['Bacterial']:
                    subtype = 'Bacterial'
                elif rand < Config.PNEUMONIA_SUBTYPES['Bacterial'] + Config.PNEUMONIA_SUBTYPES['Viral']:
                    subtype = 'Viral'
                else:
                    subtype = 'Atypical'
            
            # Generate record
            record = generate_patient_record(
                claim_counter=claim_counter,
                patient_id=patient['patient_id'],
                patient_visit_num=visit_num + 1,
                disease=disease,
                status=status,
                sex=patient['sex'],
                age_group=patient['age_group'],
                subtype=subtype,
                is_fraud=False
            )
            records.append(record)
            
            # Update counters
            claim_counter += 1
            real_generated += 1
            counts['total'] += 1
            counts['real'] += 1
            counts['by_disease'][disease] += 1
            counts['by_status'][status] += 1
            counts['by_sex'][patient['sex']] += 1
            counts['by_age_group'][patient['age_group']] += 1
            counts['patient_visits'][patient['patient_id']] += 1
            
            # Update disease target
            real_disease_targets[disease] -= 1
        
        patient_index += 1
        
        # Progress indicator
        if real_generated % 1000 == 0:
            print(f"  Generated {real_generated}/{real_records} real records...")
    
    # Generate FRAUD records (20%)
    print(f"\nGenerating FRAUD records ({fraud_records} records)...")
    
    fraud_disease_targets = {
        disease: int(fraud_records * pct) 
        for disease, pct in Config.DISEASE_DISTRIBUTION.items()
    }
    
    for fraud_num in range(fraud_records):
        # Randomly select disease
        available_diseases = [d for d, t in fraud_disease_targets.items() if t > 0]
        if not available_diseases:
            break
        
        disease = random.choice(available_diseases)
        
        # For fraud, we can ignore normal/infected - just generate anomalous values
        # But we'll still assign a status for consistency
        status = random.choice(['Normal', 'Infected'])
        
        # Generate random patient demographics for fraud cases
        sex = 'Male' if random.random() < Config.MALE_PERCENT else 'Female'
        age_group = random.choice(list(Config.AGE_GROUPS.keys()))
        
        # Create fraud patient ID (single visit only)
        fraud_patient_id = f"PAT-{datetime.now().year}-{fraud_num+1:06d}"
        
        # Determine subtype for pneumonia
        subtype = None
        if disease == 'PNEUMONIA' and status == 'Infected':
            subtype = random.choice(['Bacterial', 'Viral', 'Atypical'])
        
        # Generate record
        record = generate_patient_record(
            claim_counter=claim_counter,
            patient_id=fraud_patient_id,
            patient_visit_num=1,
            disease=disease,
            status=status,
            sex=sex,
            age_group=age_group,
            subtype=subtype,
            is_fraud=True
        )
        records.append(record)
        
        # Update counters
        claim_counter += 1
        counts['total'] += 1
        counts['fraud'] += 1
        counts['by_disease'][disease] += 1
        counts['by_sex'][sex] += 1
        counts['by_age_group'][age_group] += 1
        
        fraud_disease_targets[disease] -= 1
        
        # Progress indicator
        if (fraud_num + 1) % 500 == 0:
            print(f"  Generated {fraud_num + 1}/{fraud_records} fraud records...")
    
    # Shuffle records to mix real and fraud
    random.shuffle(records)
    
    return pd.DataFrame(records), counts


def print_summary(counts, total_records):
    """Print generation summary"""
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total Records Generated: {counts['total']} / {total_records}")
    print(f"  Real Data: {counts['real']} ({counts['real']/total_records*100:.1f}%)")
    print(f"  Fraud Data: {counts['fraud']} ({counts['fraud']/total_records*100:.1f}%)")
    print()
    
    print("By Disease:")
    for disease, count in counts['by_disease'].items():
        pct = (count / total_records) * 100
        print(f"  {disease}: {count} ({pct:.1f}%)")
    print()
    
    print("By Status (within Real Data):")
    for status, count in counts['by_status'].items():
        pct = (count / counts['real']) * 100 if counts['real'] > 0 else 0
        print(f"  {status}: {count} ({pct:.1f}%)")
    print()
    
    print("By Sex:")
    for sex, count in counts['by_sex'].items():
        pct = (count / total_records) * 100
        print(f"  {sex}: {count} ({pct:.1f}%)")
    print()
    
    print("By Age Group:")
    for age_group, count in counts['by_age_group'].items():
        pct = (count / total_records) * 100
        print(f"  {age_group}: {count} ({pct:.1f}%)")
    print()
    
    print("Patient Visit Statistics:")
    visit_counts = list(counts['patient_visits'].values())
    if visit_counts:
        print(f"  Unique Patients: {len(counts['patient_visits'])}")
        print(f"  Avg Visits per Patient: {sum(visit_counts)/len(visit_counts):.2f}")
        print(f"  Max Visits: {max(visit_counts)}")
        print(f"  Min Visits: {min(visit_counts)}")
    print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # USER CAN MODIFY THIS VALUE
    DESIRED_RECORDS = 20 # <-- CHANGE THIS TO YOUR DESIRED NUMBER
    
    print("=" * 60)
    print("KENYAN MEDICAL CLAIMS SYNTHETIC DATA GENERATOR")
    print("WITH FRAUD LAYER AND MULTI-VISIT PATIENTS")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Total Records: {DESIRED_RECORDS}")
    print(f"  Real/Fraud Split: 80%/20%")
    print(f"  Diseases: 5 (20% each)")
    print(f"  Normal/Infected (Real only): 30%/70%")
    print(f"  Male/Female: 50%/50%")
    print(f"  Age Groups: 4 (25% each)")
    print(f"  Max Visits per Patient: {Config.MAX_VISITS_PER_PATIENT}")
    print(f"  Date Range: Feb 2022 - Feb 2026")
    print(f"  Random Seed: {Config.RANDOM_SEED}")
    print("=" * 60)
    
    # Generate dataset
    df, counts = generate_dataset(DESIRED_RECORDS)
    
    # Print summary
    print_summary(counts, DESIRED_RECORDS)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kenyan_medical_claims_{timestamp}.csv"
    
    print(f"\nSaving to {filename}...")
    df.to_csv(filename, index=False)
    
    # Also save Excel version with multiple sheets
    excel_filename = f"kenyan_medical_claims_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Claims', index=False)
        
        # Create summary sheet
        summary_data = {
            'Metric': [
                'Total Records', 
                'Real Records', 
                'Fraud Records', 
                'Unique Patients',
                'Date Generated', 
                'Random Seed',
                'Diseases'
            ],
            'Value': [
                len(df),
                counts['real'],
                counts['fraud'],
                len(counts['patient_visits']),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                Config.RANDOM_SEED,
                'PNEUMONIA, ASTHMA, TBI, APH/PPH, PUERPERAL_SEPSIS'
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Create fraud analysis sheet
        fraud_df = df[df['is_fraud'] == 1]
        if len(fraud_df) > 0:
            fraud_df.to_excel(writer, sheet_name='Fraud_Records', index=False)
    
    print(f"Data saved to {filename} and {excel_filename}")
    print("\nFirst 5 records:")
    print(df.head())
    print("=" * 60)
    print("GENERATION COMPLETE!")