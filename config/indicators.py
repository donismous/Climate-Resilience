"""
Mapping between user-friendly indicator names and ND-GAIN API codes.
"""


indicator_map = {
    "Capacity": "NDGAIN_CAPACITY",
    "Economic": "NDGAIN_ECONOMIC",
    "Ecosystems": "NDGAIN_ECOSYSTEMS",
    "Exposure": "NDGAIN_EXPOSURE",
    "Food": "NDGAIN_FOOD",
    "Governance": "NDGAIN_GOVERNANCE",
    "Habitat": "NDGAIN_HABITAT",
    "Health": "NDGAIN_HEALTH",
    "Infrastructure": "NDGAIN_INFRASTRUCTURE",
    "Readiness": "NDGAIN_READINESS",
    "Sensitivity": "NDGAIN_SENSITIVITY",
    "Social": "NDGAIN_SOCIAL",
    "Vulnerability": "NDGAIN_VULNERABILITY",
    "Water": "NDGAIN_WATER",
}

# Reverse mapping for renaming API codes to friendly names
rename_map = {v: k for k, v in indicator_map.items()}
