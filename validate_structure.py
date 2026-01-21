#!/usr/bin/env python3
"""
Project Structure Validator
============================

Validates the new modular structure without requiring all dependencies.
"""

import os
import sys
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_structure():
    """Check if the directory structure is correct"""
    print(f"\n{BLUE}=== Checking Project Structure ==={RESET}\n")
    
    required_dirs = [
        'data_processing',
        'data_processing/external_apis',
        'data_processing/news',
        'data_processing/scrapers',
        'machine_learning',
        'machine_learning/models',
        'machine_learning/training',
        'machine_learning/prediction',
        'machine_learning/utils',
        'trading',
        'trading/brokers',
        'trading/strategies',
        'trading/execution',
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"{GREEN}✓{RESET} {dir_path}/")
        else:
            print(f"{RED}✗{RESET} {dir_path}/ (missing)")
            all_ok = False
    
    return all_ok


def check_files():
    """Check if key files exist"""
    print(f"\n{BLUE}=== Checking Key Files ==={RESET}\n")
    
    required_files = [
        'README.md',
        'MIGRATION_GUIDE.md',
        'requirements.txt',
        '.gitignore',
        'integration_example.py',
        'main.py',
        
        # Module files
        'data_processing/__init__.py',
        'data_processing/external_apis/api_integrations.py',
        'data_processing/news/news_aggregator.py',
        'data_processing/scrapers/web_scraper.py',
        
        'machine_learning/__init__.py',
        'machine_learning/models/ml_models.py',
        'machine_learning/training/model_training.py',
        'machine_learning/prediction/prediction_engine.py',
        
        'trading/__init__.py',
        'trading/brokers/broker_integrations.py',
        'trading/strategies/trading_strategies.py',
    ]
    
    all_ok = True
    for file_path in required_files:
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"{GREEN}✓{RESET} {file_path} ({size} bytes)")
        else:
            print(f"{RED}✗{RESET} {file_path} (missing)")
            all_ok = False
    
    return all_ok


def check_python_syntax():
    """Check Python syntax of key files"""
    print(f"\n{BLUE}=== Checking Python Syntax ==={RESET}\n")
    
    python_files = [
        'integration_example.py',
        'data_processing/external_apis/api_integrations.py',
        'data_processing/news/news_aggregator.py',
        'data_processing/scrapers/web_scraper.py',
        'machine_learning/models/ml_models.py',
        'machine_learning/training/model_training.py',
        'machine_learning/prediction/prediction_engine.py',
        'trading/brokers/broker_integrations.py',
        'trading/strategies/trading_strategies.py',
    ]
    
    all_ok = True
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
            print(f"{GREEN}✓{RESET} {file_path}")
        except SyntaxError as e:
            print(f"{RED}✗{RESET} {file_path} - Syntax Error: {e}")
            all_ok = False
        except Exception as e:
            print(f"{YELLOW}⚠{RESET} {file_path} - Warning: {e}")
    
    return all_ok


def check_documentation():
    """Check documentation completeness"""
    print(f"\n{BLUE}=== Checking Documentation ==={RESET}\n")
    
    docs = {
        'README.md': ['Instalación', 'Uso', 'Arquitectura'],
        'MIGRATION_GUIDE.md': ['Migration', 'Nueva Estructura'],
    }
    
    all_ok = True
    for doc_file, keywords in docs.items():
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            found = all(keyword.lower() in content.lower() for keyword in keywords)
            if found:
                print(f"{GREEN}✓{RESET} {doc_file} (contains key sections)")
            else:
                print(f"{YELLOW}⚠{RESET} {doc_file} (may be incomplete)")
                all_ok = False
        except Exception as e:
            print(f"{RED}✗{RESET} {doc_file} - Error: {e}")
            all_ok = False
    
    return all_ok


def print_summary(results):
    """Print summary of checks"""
    print(f"\n{BLUE}=== Validation Summary ==={RESET}\n")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"{check}: {status}")
    
    print()
    if all_passed:
        print(f"{GREEN}✓ All validations passed!{RESET}")
        print(f"\nNext steps:")
        print(f"  1. Install dependencies: pip install -r requirements.txt")
        print(f"  2. Run integration example: python integration_example.py")
        print(f"  3. Read MIGRATION_GUIDE.md for migration instructions")
        return 0
    else:
        print(f"{RED}✗ Some validations failed. Please review the errors above.{RESET}")
        return 1


def main():
    """Main validation function"""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}PROJECT STRUCTURE VALIDATION{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")
    
    results = {
        'Directory Structure': check_structure(),
        'Required Files': check_files(),
        'Python Syntax': check_python_syntax(),
        'Documentation': check_documentation(),
    }
    
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
