"""
Anonymization processor.

Replaces sensitive information with tokens instead of removing them.
This preserves text structure and allows the model to learn context.
"""

import re
from typing import Dict, Any

from ..base import BaseProcessor


class AnonymizationProcessor(BaseProcessor):
    """
    Processor for anonymizing sensitive information.
    
    IMPORTANT: This processor REPLACES sensitive data with tokens,
    it does NOT remove them. This preserves:
    - Text structure and flow
    - Sentence boundaries
    - Positional information
    - Context for language modeling
    
    Supported anonymization:
    - Email addresses → <EMAIL>
    - Phone numbers → <PHONE>
    - URLs → <URL>
    
    Configuration:
    - anonymize_emails: Enable email anonymization (default: False)
    - anonymize_phones: Enable phone anonymization (default: False)
    - anonymize_urls: Enable URL anonymization (default: False)
    
    Example:
        Input:  "Contact me at john@example.com or +420 123 456 789"
        Output: "Contact me at <EMAIL> or <PHONE>"
    """

    # Regex patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Phone patterns (international and Czech formats)
    PHONE_PATTERNS = [
        r'\+\d{1,3}[\s-]?\d{3,4}[\s-]?\d{3,4}[\s-]?\d{3,4}',  # +420 123 456 789
        r'\d{3}[\s-]?\d{3}[\s-]?\d{3}',  # 123 456 789
        r'\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}',  # (123) 456-7890
    ]
    
    # URL pattern
    URL_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    def get_name(self) -> str:
        return "AnonymizationProcessor"

    def process(self, text: str) -> str:
        """
        Anonymize sensitive information by replacing with tokens.
        
        Process:
        1. Replace email addresses with <EMAIL>
        2. Replace phone numbers with <PHONE>
        3. Replace URLs with <URL>
        
        Args:
            text: Input text potentially containing sensitive data
            
        Returns:
            Text with sensitive data replaced by tokens
        """
        result = text
        
        # 1. Anonymize emails
        if self.config.get('anonymize_emails', False):
            result = re.sub(self.EMAIL_PATTERN, '<EMAIL>', result)
        
        # 2. Anonymize phone numbers
        if self.config.get('anonymize_phones', False):
            for pattern in self.PHONE_PATTERNS:
                result = re.sub(pattern, '<PHONE>', result)
        
        # 3. Anonymize URLs
        if self.config.get('anonymize_urls', False):
            result = re.sub(self.URL_PATTERN, '<URL>', result)
        
        return result


class AdvancedAnonymizationProcessor(BaseProcessor):
    """
    Advanced anonymization with additional patterns.
    
    Extends AnonymizationProcessor with:
    - IP addresses → <IPADDR>
    - Credit card numbers → <CREDITCARD>
    - Social security numbers (Czech RČ) → <SSN>
    - Dates → <DATE>
    
    Configuration:
    - All from AnonymizationProcessor
    - anonymize_ips: Enable IP address anonymization (default: False)
    - anonymize_creditcards: Enable credit card anonymization (default: False)
    - anonymize_ssn: Enable SSN/RČ anonymization (default: False)
    - anonymize_dates: Enable date anonymization (default: False)
    """

    # Additional patterns
    IP_PATTERN = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    
    # Credit card pattern (spaces or dashes)
    CREDITCARD_PATTERN = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    
    # Czech birth number (rodné číslo) - format: YYMMDD/XXXX
    SSN_PATTERN = r'\b\d{6}/\d{4}\b'
    
    # Date patterns (various formats)
    DATE_PATTERNS = [
        r'\b\d{1,2}\.\s?\d{1,2}\.\s?\d{4}\b',  # DD.MM.YYYY
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
    ]

    def get_name(self) -> str:
        return "AdvancedAnonymizationProcessor"

    def process(self, text: str) -> str:
        """
        Perform advanced anonymization.
        
        Args:
            text: Input text
            
        Returns:
            Text with all configured sensitive data replaced by tokens
        """
        result = text
        
        # Standard anonymization (from parent logic)
        if self.config.get('anonymize_emails', False):
            result = re.sub(AnonymizationProcessor.EMAIL_PATTERN, '<EMAIL>', result)
        
        if self.config.get('anonymize_phones', False):
            for pattern in AnonymizationProcessor.PHONE_PATTERNS:
                result = re.sub(pattern, '<PHONE>', result)
        
        if self.config.get('anonymize_urls', False):
            result = re.sub(AnonymizationProcessor.URL_PATTERN, '<URL>', result)
        
        # Advanced anonymization
        if self.config.get('anonymize_ips', False):
            result = re.sub(self.IP_PATTERN, '<IPADDR>', result)
        
        if self.config.get('anonymize_creditcards', False):
            result = re.sub(self.CREDITCARD_PATTERN, '<CREDITCARD>', result)
        
        if self.config.get('anonymize_ssn', False):
            result = re.sub(self.SSN_PATTERN, '<SSN>', result)
        
        if self.config.get('anonymize_dates', False):
            for pattern in self.DATE_PATTERNS:
                result = re.sub(pattern, '<DATE>', result)
        
        return result