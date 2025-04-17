#!/usr/bin/env python

"""
Test script to verify the Conda environment setup.
This script validates that all required packages are correctly installed
and accessible in the Conda environment.
"""

import sys
import importlib.util
from typing import List, Dict

def check_package(package_name: str) -> Dict[str, str]:
    """
    Check if a package is installed and get its version.
    
    Args:
        package_name: The name of the package to check
        
    Returns:
        Dictionary with status and version information
    """
    result = {"name": package_name, "installed": False, "version": None, "error": None}
    
    try:
        # Check if the package is available
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            result["error"] = "Package not found"
            return result
            
        # Try to import the package
        module = importlib.import_module(package_name)
        result["installed"] = True
        
        # Get version if available
        try:
            result["version"] = getattr(module, "__version__", "Unknown")
        except AttributeError:
            result["version"] = "Version attribute not found"
            
    except Exception as e:
        result["error"] = str(e)
        
    return result

def main():
    """Main function to check all required packages."""
    # List of core packages to check
    core_packages = [
        "chromadb",
        "openai",
        "numpy",
        "tqdm",
        "dotenv",
        "psutil",
        "flask",
        "pytest",
        "black",
        "tree_sitter",
    ]
    
    # Print Python information
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("\nPackage Status:")
    print("-" * 60)
    
    # Check each package
    all_installed = True
    for package in core_packages:
        result = check_package(package)
        status = "✓" if result["installed"] else "✗"
        version = result["version"] if result["installed"] else "N/A"
        error = f" (Error: {result['error']})" if result["error"] else ""
        
        print(f"{status} {result['name']:15} {version:10} {error}")
        
        if not result["installed"]:
            all_installed = False
    
    print("-" * 60)
    
    # Print summary
    if all_installed:
        print("\n✓ All required packages are installed correctly!")
        sys.exit(0)
    else:
        print("\n✗ Some packages are missing or could not be imported.")
        print("  Please check your environment setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()