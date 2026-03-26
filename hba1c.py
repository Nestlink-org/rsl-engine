"""
Synthetic Medical Claims Data Generator for Kenyan Healthcare Facilities
DIABETIC DISEASES ONLY: DKA/HHS, Diabetic Infections, Diabetic Nephropathy
Tests: HBA1C, CREATININE, UREA
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
    TOTAL_RECORDS = 100  # Change this to desired number
    
    # Disease distribution (equal 33.33% each)
    DISEASE_DISTRIBUTION = {
        'DIABETES_DKA_HHS': 1/3,
        'DIABETIC_INFECTIONS': 1/3,
        'DIABETIC_NEPHROPATHY': 1/3
    }
    
    # Data integrity layers
    REAL_DATA_PERCENT = 0.80    # 80% real data (follows ranges)
    FRAUD_DATA_PERCENT = 0.20   # 20% fraud data (outside ranges)
    
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
    AGE_GROUP_PERCENT = 0.25
    
    # Patient visit limits
    MAX_VISITS_PER_PATIENT = 10
    MIN_VISITS_PER_PATIENT = 1
    
    # Date range
    START_DATE = datetime(2022, 2, 1)
    END_DATE = datetime(2026, 2, 28)
    
    # Length of stay by disease (days)
    LENGTH_OF_STAY = {
        'DIABETES_DKA_HHS': {'min': 3, 'max': 7},
        'DIABETIC_INFECTIONS': {'min': 4, 'max': 10},
        'DIABETIC_NEPHROPATHY': {'min': 2, 'max': 5}
    }
    
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
# DISEASE INFORMATION
# =============================================================================

DISEASE_INFO = {
    'DIABETES_DKA_HHS': {
        'display_name': 'Diabetes - (DKA/HHS)',
        'category': 'diabetes',
        'icd10': 'E10.1/E11.0',
        'procedures': [
            'IV insulin', 'Fluid resuscitation', 'Electrolyte replacement', 
            'ICU admission', 'Blood glucose monitoring', 'Arterial blood gas',
            'Urine ketones', 'IV fluids', 'Potassium replacement'
        ]
    },
    'DIABETIC_INFECTIONS': {
        'display_name': 'Diabetic Infections',
        'category': 'diabetes',
        'icd10': 'E10.6/E11.6',
        'procedures': [
            'IV antibiotics', 'Wound debridement', 'Insulin therapy', 
            'Blood culture', 'Wound culture', 'Fever management',
            'Incision and drainage', 'Foot care', 'Antibiotic therapy'
        ]
    },
    'DIABETIC_NEPHROPATHY': {
        'display_name': 'Diabetic Nephropathy',
        'category': 'diabetes',
        'icd10': 'E10.2/E11.2',
        'procedures': [
            'Dialysis', 'Renal ultrasound', 'Nephrology consult', 
            'ACE inhibitors', 'Protein restriction', 'Blood pressure management',
            'Urinalysis', '24-hour urine collection', 'Renal biopsy'
        ]
    }
}


# =============================================================================
# DIABETIC RANGES - EXACT FROM EXCEL FILES
# =============================================================================

class DiabeticRanges:
    """Complete ranges for diabetic diseases"""
    
    # =========================================================================
    # NORMAL REFERENCE RANGES (Baseline)
    # =========================================================================
    
    @staticmethod
    def get_normal_range(parameter, age_group, sex):
        """Get normal reference range for parameter"""
        
        normal_ranges = {
            'HBA1C': {
                'Pediatric': {'Male': (4.5, 5.5), 'Female': (4.5, 5.5)},
                'Transition': {'Male': (4.5, 5.7), 'Female': (4.5, 5.7)},
                'Adult': {'Male': (4.5, 5.7), 'Female': (4.5, 5.7)},
                'Geriatric': {'Male': (4.8, 6.0), 'Female': (4.8, 6.0)}
            },
            'CREATININE': {
                'Pediatric': {'Male': (0.2, 0.5), 'Female': (0.2, 0.5)},
                'Transition': {'Male': (0.4, 0.9), 'Female': (0.4, 0.8)},
                'Adult': {'Male': (0.7, 1.3), 'Female': (0.6, 1.1)},
                'Geriatric': {'Male': (0.8, 1.4), 'Female': (0.7, 1.3)}
            },
            'UREA': {
                'Pediatric': {'Male': (5, 18), 'Female': (5, 18)},
                'Transition': {'Male': (7, 20), 'Female': (7, 20)},
                'Adult': {'Male': (8, 24), 'Female': (8, 22)},
                'Geriatric': {'Male': (10, 28), 'Female': (10, 26)}
            }
        }
        
        return normal_ranges.get(parameter, {}).get(age_group, {}).get(sex, (0, 0))
    
    # =========================================================================
    # DIABETES DKA/HHS - INFECTED RANGES
    # =========================================================================
    
    @staticmethod
    def get_dka_hhs_range(parameter, age_group, sex):
        """Get DKA/HHS infected range"""
        
        ranges = {
            'HBA1C': {
                'Pediatric': {'Male': (8.0, 13.0), 'Female': (8.0, 13.0)},
                'Transition': {'Male': (8.0, 14.0), 'Female': (8.0, 14.0)},
                'Adult': {'Male': (9.0, 15.0), 'Female': (9.0, 15.0)},
                'Geriatric': {'Male': (9.0, 16.0), 'Female': (9.0, 16.0)}
            },
            'CREATININE': {
                'Pediatric': {'Male': (0.6, 1.4), 'Female': (0.6, 1.3)},
                'Transition': {'Male': (0.9, 1.8), 'Female': (0.8, 1.6)},
                'Adult': {'Male': (1.5, 3.5), 'Female': (1.4, 3.2)},
                'Geriatric': {'Male': (2.0, 5.0), 'Female': (1.8, 4.8)}
            },
            'UREA': {
                'Pediatric': {'Male': (25, 60), 'Female': (25, 60)},
                'Transition': {'Male': (30, 70), 'Female': (30, 70)},
                'Adult': {'Male': (40, 100), 'Female': (40, 100)},
                'Geriatric': {'Male': (60, 140), 'Female': (60, 140)}
            }
        }
        
        return ranges.get(parameter, {}).get(age_group, {}).get(sex, (0, 0))
    
    # =========================================================================
    # DIABETIC INFECTIONS - INFECTED RANGES
    # =========================================================================
    
    @staticmethod
    def get_infections_range(parameter, age_group, sex):
        """Get Diabetic Infections infected range"""
        
        ranges = {
            'HBA1C': {
                'Pediatric': {'Male': (7.0, 10.0), 'Female': (7.0, 10.0)},
                'Transition': {'Male': (7.0, 11.0), 'Female': (7.0, 11.0)},
                'Adult': {'Male': (8.0, 12.0), 'Female': (8.0, 12.0)},
                'Geriatric': {'Male': (8.0, 13.0), 'Female': (8.0, 13.0)}
            },
            'CREATININE': {
                'Pediatric': {'Male': (0.6, 1.2), 'Female': (0.6, 1.1)},
                'Transition': {'Male': (0.8, 1.6), 'Female': (0.7, 1.4)},
                'Adult': {'Male': (1.2, 2.5), 'Female': (1.1, 2.3)},
                'Geriatric': {'Male': (1.5, 3.5), 'Female': (1.4, 3.2)}
            },
            'UREA': {
                'Pediatric': {'Male': (18, 40), 'Female': (18, 40)},
                'Transition': {'Male': (20, 50), 'Female': (20, 50)},
                'Adult': {'Male': (30, 70), 'Female': (30, 70)},
                'Geriatric': {'Male': (40, 90), 'Female': (40, 90)}
            }
        }
        
        return ranges.get(parameter, {}).get(age_group, {}).get(sex, (0, 0))
    
    # =========================================================================
    # DIABETIC NEPHROPATHY - INFECTED RANGES
    # =========================================================================
    
    @staticmethod
    def get_nephropathy_range(parameter, age_group, sex):
        """Get Diabetic Nephropathy infected range"""
        
        ranges = {
            'HBA1C': {
                'Pediatric': {'Male': (7.0, 10.0), 'Female': (7.0, 10.0)},
                'Transition': {'Male': (7.0, 11.0), 'Female': (7.0, 11.0)},
                'Adult': {'Male': (8.0, 13.0), 'Female': (8.0, 13.0)},
                'Geriatric': {'Male': (8.0, 14.0), 'Female': (8.0, 14.0)}
            },
            'CREATININE': {
                'Pediatric': {'Male': (0.6, 1.3), 'Female': (0.6, 1.2)},
                'Transition': {'Male': (0.9, 1.8), 'Female': (0.8, 1.6)},
                'Adult': {'Male': (2.0, 6.0), 'Female': (1.8, 5.5)},
                'Geriatric': {'Male': (2.5, 7.5), 'Female': (2.2, 7.0)}
            },
            'UREA': {
                'Pediatric': {'Male': (20, 45), 'Female': (20, 45)},
                'Transition': {'Male': (25, 60), 'Female': (25, 60)},
                'Adult': {'Male': (50, 150), 'Female': (50, 150)},
                'Geriatric': {'Male': (60, 180), 'Female': (60, 180)}
            }
        }
        
        return ranges.get(parameter, {}).get(age_group, {}).get(sex, (0, 0))
    
    # =========================================================================
    # FRAUD RANGE GENERATOR
    # =========================================================================
    
    @staticmethod
    def get_fraud_range(parameter, age_group, sex):
        """Generate values outside normal ranges for fraud detection"""
        
        normal_range = DiabeticRanges.get_normal_range(parameter, age_group, sex)
        normal_min, normal_max = normal_range
        
        fraud_multipliers = {
            'HBA1C': (0.3, 2.5),      # Very low or very high
            'CREATININE': (0.2, 3.0),  # Extremes
            'UREA': (0.2, 3.0)         # Extremes
        }
        
        if parameter in fraud_multipliers:
            low_mult, high_mult = fraud_multipliers[parameter]
            
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
    def get_range(cls, disease, status, parameter, age_group, sex, is_fraud=False):
        """Main dispatcher to get appropriate range"""
        
        if is_fraud:
            return cls.get_fraud_range(parameter, age_group, sex)
        
        if status == 'Normal':
            return cls.get_normal_range(parameter, age_group, sex)
        
        if disease == 'DIABETES_DKA_HHS':
            return cls.get_dka_hhs_range(parameter, age_group, sex)
        elif disease == 'DIABETIC_INFECTIONS':
            return cls.get_infections_range(parameter, age_group, sex)
        elif disease == 'DIABETIC_NEPHROPATHY':
            return cls.get_nephropathy_range(parameter, age_group, sex)
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


def generate_value(param_range):
    """Generate random value within range"""
    min_val, max_val = param_range
    if max_val - min_val < 0.1:
        return round(min_val, 1)
    
    # Different rounding for different parameters
    if max_val > 50:
        return round(random.uniform(min_val, max_val), 0)  # Whole numbers for large values
    elif max_val > 10:
        return round(random.uniform(min_val, max_val), 1)  # One decimal
    else:
        return round(random.uniform(min_val, max_val), 1)  # One decimal


def generate_patient_record(claim_counter, patient_id, disease, status, sex, age_group, is_fraud=False):
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
    
    # Disease info
    record['disease_category'] = DISEASE_INFO[disease]['category']
    record['diagnosis'] = DISEASE_INFO[disease]['display_name']
    record['diagnosis_code'] = DISEASE_INFO[disease]['icd10']
    
    # Procedure
    record['procedure'] = random.choice(DISEASE_INFO[disease]['procedures'])
    
    # Dates
    admission_date = generate_date()
    los = random.randint(
        Config.LENGTH_OF_STAY[disease]['min'],
        Config.LENGTH_OF_STAY[disease]['max']
    )
    discharge_date = admission_date + timedelta(days=los)
    processed_date = discharge_date + timedelta(hours=random.randint(24, 48))
    
    record['admission_date'] = admission_date.strftime('%Y-%m-%d')
    record['discharge_date'] = discharge_date.strftime('%Y-%m-%d')
    record['date'] = admission_date.strftime('%Y-%m-%d')
    record['timestamp_processed'] = processed_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Fraud indicator
    record['is_fraud'] = 1 if is_fraud else 0
    
    # Diabetic parameters
    parameters = ['HBA1C', 'CREATININE', 'UREA']
    
    for param in parameters:
        param_range = DiabeticRanges.get_range(
            disease=disease,
            status=status,
            parameter=param,
            age_group=age_group,
            sex=sex,
            is_fraud=is_fraud
        )
        record[param] = generate_value(param_range)
    
    return record


def generate_patient_pool(total_real_records):
    """Generate a pool of patients with their visit counts"""
    
    avg_visits = 3.5
    num_patients = int(total_real_records / avg_visits)
    
    patients = []
    for i in range(1, num_patients + 1):
        num_visits = random.randint(Config.MIN_VISITS_PER_PATIENT, Config.MAX_VISITS_PER_PATIENT)
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
    """Main function to generate complete dataset"""
    
    print(f"\n{'='*60}")
    print("DIABETIC DISEASES SYNTHETIC DATA GENERATOR")
    print(f"{'='*60}")
    print(f"Generating {total_records} records...")
    print(f"{'='*60}")
    
    set_random_seed(Config.RANDOM_SEED)
    
    # Calculate fraud vs real split
    fraud_records = int(total_records * Config.FRAUD_DATA_PERCENT)
    real_records = total_records - fraud_records
    
    print(f"\nData Integrity Split:")
    print(f"  Real Data: {real_records} records ({Config.REAL_DATA_PERCENT*100:.0f}%)")
    print(f"  Fraud Data: {fraud_records} records ({Config.FRAUD_DATA_PERCENT*100:.0f}%)")
    
    # Calculate disease targets
    real_disease_targets = {
        disease: int(real_records * pct) 
        for disease, pct in Config.DISEASE_DISTRIBUTION.items()
    }
    
    fraud_disease_targets = {
        disease: int(fraud_records * pct) 
        for disease, pct in Config.DISEASE_DISTRIBUTION.items()
    }
    
    print(f"\nDisease Distribution (each 33.33%):")
    for disease in Config.DISEASE_DISTRIBUTION.keys():
        print(f"  {disease}: {real_disease_targets[disease] + fraud_disease_targets[disease]} records")
    
    # Generate patient pool for real data
    patients = generate_patient_pool(real_records)
    print(f"\nCreated {len(patients)} unique patients for real data")
    print(f"  Avg visits per patient: {real_records/len(patients):.1f}")
    print(f"  Max visits: {Config.MAX_VISITS_PER_PATIENT}")
    
    records = []
    claim_counter = 1
    
    # Track counts
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
    
    # =========================================================
    # Generate REAL patient records (80%)
    # =========================================================
    print(f"\n{'='*60}")
    print("GENERATING REAL PATIENT RECORDS (Multiple Visits)")
    print(f"{'='*60}")
    
    random.shuffle(patients)
    real_generated = 0
    patient_index = 0
    
    while real_generated < real_records and patient_index < len(patients):
        patient = patients[patient_index]
        
        for visit_num in range(patient['num_visits']):
            if real_generated >= real_records:
                break
            
            # Select disease based on remaining targets
            available_diseases = [d for d, t in real_disease_targets.items() if t > 0]
            if not available_diseases:
                break
            
            weights = [real_disease_targets[d] for d in available_diseases]
            disease = random.choices(available_diseases, weights=weights)[0]
            
            # Determine status (30% Normal, 70% Infected)
            status = 'Normal' if random.random() < Config.NORMAL_PERCENT else 'Infected'
            
            # Generate record
            record = generate_patient_record(
                claim_counter=claim_counter,
                patient_id=patient['patient_id'],
                disease=disease,
                status=status,
                sex=patient['sex'],
                age_group=patient['age_group'],
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
            
            real_disease_targets[disease] -= 1
        
        patient_index += 1
        
        if real_generated % 1000 == 0:
            print(f"  Generated {real_generated}/{real_records} real records...")
    
    # =========================================================
    # Generate FRAUD records (20%)
    # =========================================================
    print(f"\n{'='*60}")
    print("GENERATING FRAUD RECORDS (Single Visit)")
    print(f"{'='*60}")
    
    for fraud_num in range(fraud_records):
        # Select disease
        available_diseases = [d for d, t in fraud_disease_targets.items() if t > 0]
        if not available_diseases:
            break
        
        disease = random.choice(available_diseases)
        
        # Random demographics for fraud cases
        sex = 'Male' if random.random() < Config.MALE_PERCENT else 'Female'
        age_group = random.choice(list(Config.AGE_GROUPS.keys()))
        
        # Fraud patient ID (single visit only)
        fraud_patient_id = f"PAT-{datetime.now().year}-FRAUD-{fraud_num+1:06d}"
        
        # Status can be random for fraud
        status = random.choice(['Normal', 'Infected'])
        
        # Generate record
        record = generate_patient_record(
            claim_counter=claim_counter,
            patient_id=fraud_patient_id,
            disease=disease,
            status=status,
            sex=sex,
            age_group=age_group,
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
        
        if (fraud_num + 1) % 500 == 0:
            print(f"  Generated {fraud_num + 1}/{fraud_records} fraud records...")
    
    # Shuffle records
    random.shuffle(records)
    
    return pd.DataFrame(records), counts


def print_summary(counts, total_records):
    """Print generation summary"""
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
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
    print(f"{'='*60}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # USER CAN MODIFY THIS VALUE
    DESIRED_RECORDS = 100  # <-- CHANGE THIS TO YOUR DESIRED NUMBER
    
    print(f"\n{'='*60}")
    print("DIABETIC DISEASES DATA GENERATOR")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Total Records: {DESIRED_RECORDS}")
    print(f"  Diseases: 3 diabetic conditions (33.33% each)")
    print(f"    - DKA/HHS")
    print(f"    - Diabetic Infections")
    print(f"    - Diabetic Nephropathy")
    print(f"  Real/Fraud Split: 80%/20%")
    print(f"  Normal/Infected (Real only): 30%/70%")
    print(f"  Male/Female: 50%/50%")
    print(f"  Age Groups: 4 (25% each)")
    print(f"  Max Visits per Patient: {Config.MAX_VISITS_PER_PATIENT}")
    print(f"  Date Range: Feb 2022 - Feb 2026")
    print(f"  Random Seed: {Config.RANDOM_SEED}")
    print(f"{'='*60}")
    
    # Generate dataset
    df, counts = generate_dataset(DESIRED_RECORDS)
    
    # Print summary
    print_summary(counts, DESIRED_RECORDS)
    
    # Save to CSV only (as requested)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kenyan_diabetic_claims_{timestamp}.csv"
    
    print(f"\nSaving to {filename}...")
    df.to_csv(filename, index=False)
    
    print(f"\n✅ Data saved to {filename}")
    print(f"\nFirst 5 records:")
    print(df.head())
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE!")
    print(f"{'='*60}")