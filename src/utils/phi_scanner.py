"""
PHI Scanner - HIPAA Layer 1
Scans datasets for Protected Health Information before processing
"""

import pandas as pd
import re
from typing import List, Dict, Tuple

class PHIScanner:
    """
    Scans for 18 HIPAA identifiers before allowing data processing
    """
    
    # 18 HIPAA identifiers
    HIPAA_IDENTIFIERS = [
        'name', 'ssn', 'social_security', 'social security number',
        'mrn', 'medical_record', 'medical record number',
        'phone', 'telephone', 'fax', 
        'email', 'e-mail',
        'address', 'street', 'city', 'zip', 'zipcode', 'postal',
        'dob', 'date_of_birth', 'birth_date', 'birthdate',
        'admission_date', 'discharge_date', 'death_date',
        'account', 'account_number',
        'certificate', 'license', 'license_number',
        'vehicle', 'license_plate', 'vin',
        'device', 'serial', 'serial_number',
        'url', 'website', 'ip', 'ip_address',
        'biometric', 'fingerprint', 'retina', 'voice',
        'photo', 'image', 'photograph'
    ]
    
    def __init__(self):
        self.violations = []
    
    def scan_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Scan dataset for PHI
        Returns: (is_safe: bool, violations: List[str])
        """
        self.violations = []
        
        # Check 1: Column names
        self._scan_column_names(df)
        
        # Check 2: Pattern matching in data
        self._scan_data_patterns(df)
        
        is_safe = len(self.violations) == 0
        
        return is_safe, self.violations
    
    def _scan_column_names(self, df: pd.DataFrame):
        """Check if column names contain HIPAA identifiers"""
        for col in df.columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            
            for identifier in self.HIPAA_IDENTIFIERS:
                if identifier in col_lower:
                    self.violations.append(
                        f"Column '{col}' may contain PHI ({identifier})"
                    )
                    break
    
    def _scan_data_patterns(self, df: pd.DataFrame):
        """Check for common PHI patterns in data"""
        
        # Only check string/object columns
        string_cols = df.select_dtypes(include=['object']).columns
        
        for col in string_cols:
            # Sample first 100 rows for performance
            sample = df[col].dropna().head(100).astype(str)
            
            if len(sample) == 0:
                continue
            
            # SSN pattern: ###-##-#### or #########
            ssn_pattern = r'\b\d{3}-?\d{2}-?\d{4}\b'
            if sample.str.match(ssn_pattern).any():
                self.violations.append(
                    f"Column '{col}' contains SSN pattern"
                )
            
            # Phone pattern: (###) ###-#### or ###-###-####
            phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            if sample.str.contains(phone_pattern, regex=True).any():
                self.violations.append(
                    f"Column '{col}' may contain phone numbers"
                )
            
            # Email pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            if sample.str.contains(email_pattern, regex=True).any():
                self.violations.append(
                    f"Column '{col}' may contain email addresses"
                )
            
            # ZIP code pattern (if not in a column clearly labeled as zip)
            if 'zip' not in col.lower() and 'postal' not in col.lower():
                zip_pattern = r'\b\d{5}(-\d{4})?\b'
                if sample.str.match(zip_pattern).any():
                    self.violations.append(
                        f"Column '{col}' may contain ZIP codes"
                    )
    
    def get_report(self) -> Dict:
        """Generate detailed scan report"""
        return {
            'is_safe': len(self.violations) == 0,
            'violations_found': len(self.violations),
            'violations': self.violations,
            'status': 'SAFE' if len(self.violations) == 0 else 'PHI DETECTED'
        }


class PHIDetectedError(Exception):
    """Custom exception when PHI is detected"""
    pass


def scan_and_validate(df: pd.DataFrame) -> bool:
    """
    Convenience function to scan and raise error if PHI found
    """
    scanner = PHIScanner()
    is_safe, violations = scanner.scan_dataset(df)
    
    if not is_safe:
        error_msg = "⚠️ PHI DETECTED - Cannot process this dataset\n\n"
        error_msg += "HIPAA Violations Found:\n"
        for v in violations:
            error_msg += f"  • {v}\n"
        error_msg += "\n"
        error_msg += "Please de-identify your dataset before upload.\n"
        error_msg += "Remove or anonymize: names, SSNs, phone numbers, emails, addresses, dates of birth.\n"
        
        raise PHIDetectedError(error_msg)
    
    return True


# Test if run directly
if __name__ == "__main__":
    print("Testing PHI Scanner...")
    
    # Test 1: Safe dataset (our synthetic data)
    print("\nTest 1: Safe dataset (no PHI)")
    safe_df = pd.DataFrame({
        'patient_id': ['SYNTH_0001', 'SYNTH_0002'],
        'age': [45, 62],
        'cholesterol': [220, 180]
    })
    
    scanner = PHIScanner()
    is_safe, violations = scanner.scan_dataset(safe_df)
    print(f"Safe: {is_safe}")
    print(f"Violations: {violations}")
    
    # Test 2: Unsafe dataset (contains PHI)
    print("\nTest 2: Unsafe dataset (contains PHI)")
    unsafe_df = pd.DataFrame({
        'patient_name': ['John Doe', 'Jane Smith'],
        'ssn': ['123-45-6789', '987-65-4321'],
        'age': [45, 62]
    })
    
    scanner2 = PHIScanner()
    is_safe2, violations2 = scanner2.scan_dataset(unsafe_df)
    print(f"Safe: {is_safe2}")
    print(f"Violations: {violations2}")
    
    print("\n✅ PHI Scanner tests complete!")