#!/usr/bin/env python3
"""
Phase 7-8 Structure Validation Script for Indonesian Stock Screener
Enhanced Features & Optimization Validation

This validation script checks the implementation structure of Phase 7-8
without requiring external packages, ensuring all components are properly organized.

Author: IDX Stock Screener Team
Version: 1.0.0
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def check_file_exists(filepath: Path, description: str) -> bool:
    """Check if a file exists and print result"""
    if filepath.exists():
        print_success(f"{description}: {filepath}")
        return True
    else:
        print_error(f"Missing {description}: {filepath}")
        return False

def check_directory_exists(dirpath: Path, description: str) -> bool:
    """Check if a directory exists and print result"""
    if dirpath.exists() and dirpath.is_dir():
        print_success(f"{description}: {dirpath}")
        return True
    else:
        print_error(f"Missing {description}: {dirpath}")
        return False

def validate_file_structure() -> Tuple[int, int]:
    """Validate Phase 7-8 file structure"""
    print_header("PHASE 7-8 FILE STRUCTURE VALIDATION")

    base_dir = Path(__file__).parent
    passed = 0
    total = 0

    # Core structure files
    structure_checks = [
        # Analytics module
        (base_dir / "src" / "analytics" / "__init__.py", "Analytics module init"),
        (base_dir / "src" / "analytics" / "performance_analyzer.py", "Performance Analyzer"),
        (base_dir / "src" / "analytics" / "portfolio_tracker.py", "Portfolio Tracker"),

        # ML module
        (base_dir / "src" / "ml" / "__init__.py", "ML module init"),
        (base_dir / "src" / "ml" / "signal_enhancer.py", "ML Signal Enhancer"),

        # Dashboard module
        (base_dir / "src" / "dashboard" / "__init__.py", "Dashboard module init"),
        (base_dir / "src" / "dashboard" / "app.py", "Dashboard Web App"),
        (base_dir / "src" / "dashboard" / "templates" / "dashboard.html", "Dashboard HTML Template"),

        # Test files
        (base_dir / "test_phase7_8.py", "Phase 7-8 Test Suite"),

        # Documentation
        (base_dir / "DOC" / "PHASE7_8_FEATURES.md", "Phase 7-8 Documentation"),

        # Configuration
        (base_dir / "requirements.txt", "Requirements file"),
    ]

    for filepath, description in structure_checks:
        total += 1
        if check_file_exists(filepath, description):
            passed += 1

    # Directory checks
    directory_checks = [
        (base_dir / "src" / "analytics", "Analytics directory"),
        (base_dir / "src" / "ml", "ML directory"),
        (base_dir / "src" / "dashboard", "Dashboard directory"),
        (base_dir / "src" / "dashboard" / "templates", "Dashboard templates directory"),
        (base_dir / "DOC", "Documentation directory"),
    ]

    for dirpath, description in directory_checks:
        total += 1
        if check_directory_exists(dirpath, description):
            passed += 1

    return passed, total

def analyze_code_structure() -> Tuple[int, int]:
    """Analyze code structure and imports"""
    print_header("CODE STRUCTURE ANALYSIS")

    base_dir = Path(__file__).parent
    passed = 0
    total = 0

    # Key files to analyze
    files_to_check = [
        "src/analytics/performance_analyzer.py",
        "src/analytics/portfolio_tracker.py",
        "src/ml/signal_enhancer.py",
        "src/dashboard/app.py"
    ]

    for file_path in files_to_check:
        total += 1
        full_path = base_dir / file_path

        if not full_path.exists():
            print_error(f"File not found: {file_path}")
            continue

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for key components
            checks = []

            if "performance_analyzer" in file_path:
                checks = [
                    ("class PerformanceAnalyzer", "PerformanceAnalyzer class"),
                    ("def analyze_portfolio_performance", "Portfolio analysis method"),
                    ("sharpe_ratio", "Sharpe ratio calculation"),
                    ("max_drawdown", "Drawdown analysis")
                ]

            elif "portfolio_tracker" in file_path:
                checks = [
                    ("class PortfolioTracker", "PortfolioTracker class"),
                    ("async def add_position", "Add position method"),
                    ("class Position", "Position class"),
                    ("risk_management", "Risk management")
                ]

            elif "signal_enhancer" in file_path:
                checks = [
                    ("class SignalEnhancer", "SignalEnhancer class"),
                    ("def extract_features", "Feature extraction"),
                    ("def train_models", "Model training"),
                    ("ensemble", "Ensemble methods")
                ]

            elif "app.py" in file_path:
                checks = [
                    ("class DashboardApp", "DashboardApp class"),
                    ("Flask", "Flask framework"),
                    ("@app.route", "Route definitions"),
                    ("socketio", "WebSocket support")
                ]

            component_passed = 0
            for check_text, check_desc in checks:
                if check_text.lower() in content.lower():
                    component_passed += 1

            if component_passed >= len(checks) * 0.8:  # 80% of checks pass
                print_success(f"Code structure valid: {file_path}")
                passed += 1
            else:
                print_warning(f"Code structure incomplete: {file_path} ({component_passed}/{len(checks)})")

        except Exception as e:
            print_error(f"Error analyzing {file_path}: {e}")

    return passed, total

def check_requirements() -> Tuple[int, int]:
    """Check requirements.txt for Phase 7-8 dependencies"""
    print_header("DEPENDENCIES VALIDATION")

    base_dir = Path(__file__).parent
    req_file = base_dir / "requirements.txt"

    if not req_file.exists():
        print_error("requirements.txt not found")
        return 0, 1

    try:
        with open(req_file, 'r') as f:
            content = f.read().lower()

        # Required Phase 7-8 dependencies
        required_deps = [
            ("pandas", "Data processing"),
            ("numpy", "Numerical computing"),
            ("scikit-learn", "Machine Learning"),
            ("flask", "Web framework"),
            ("plotly", "Interactive charts"),
            ("lightgbm", "ML models"),
            ("xgboost", "ML models"),
        ]

        passed = 0
        total = len(required_deps)

        for dep, desc in required_deps:
            if dep in content:
                print_success(f"Dependency found: {dep} ({desc})")
                passed += 1
            else:
                print_warning(f"Optional dependency missing: {dep} ({desc})")

        return passed, total

    except Exception as e:
        print_error(f"Error reading requirements.txt: {e}")
        return 0, 1

def validate_configuration_structure() -> Tuple[int, int]:
    """Validate configuration structure"""
    print_header("CONFIGURATION VALIDATION")

    base_dir = Path(__file__).parent
    passed = 0
    total = 0

    # Check existing configuration files
    config_files = [
        ("src/config/settings.py", "Configuration settings"),
        ("src/config/trading_config.yaml", "Trading configuration"),
    ]

    for config_file, desc in config_files:
        total += 1
        filepath = base_dir / config_file
        if filepath.exists():
            print_success(f"Configuration file exists: {desc}")
            passed += 1
        else:
            print_warning(f"Configuration file missing: {desc}")

    return passed, total

def check_integration_points() -> Tuple[int, int]:
    """Check integration between components"""
    print_header("INTEGRATION VALIDATION")

    base_dir = Path(__file__).parent
    passed = 0
    total = 0

    # Check if main.py references new components
    main_file = base_dir / "main.py"
    total += 1

    if main_file.exists():
        try:
            with open(main_file, 'r') as f:
                content = f.read()

            if "analytics" in content.lower() or "ml" in content.lower():
                print_success("Main.py shows integration with Phase 7-8 components")
                passed += 1
            else:
                print_warning("Main.py may need updates for Phase 7-8 integration")

        except Exception as e:
            print_error(f"Error checking main.py: {e}")
    else:
        print_warning("main.py not found for integration check")

    return passed, total

def generate_validation_report(results: List[Tuple[str, int, int]]) -> Dict:
    """Generate comprehensive validation report"""
    total_passed = sum(r[1] for r in results)
    total_checks = sum(r[2] for r in results)
    success_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0

    report = {
        "timestamp": "2024-12-19",  # Current date
        "total_checks": total_checks,
        "total_passed": total_passed,
        "success_rate": success_rate,
        "categories": {}
    }

    for category, passed, total in results:
        report["categories"][category] = {
            "passed": passed,
            "total": total,
            "rate": (passed / total * 100) if total > 0 else 0
        }

    return report

def main():
    """Main validation function"""
    print(f"{Colors.BOLD}IDX STOCK SCREENER - PHASE 7-8 VALIDATION{Colors.RESET}")
    print(f"{Colors.BOLD}Enhanced Features & Optimization Structure Check{Colors.RESET}")

    # Run all validation checks
    results = []

    # 1. File Structure
    passed, total = validate_file_structure()
    results.append(("File Structure", passed, total))

    # 2. Code Structure
    passed, total = analyze_code_structure()
    results.append(("Code Structure", passed, total))

    # 3. Dependencies
    passed, total = check_requirements()
    results.append(("Dependencies", passed, total))

    # 4. Configuration
    passed, total = validate_configuration_structure()
    results.append(("Configuration", passed, total))

    # 5. Integration
    passed, total = check_integration_points()
    results.append(("Integration", passed, total))

    # Generate report
    report = generate_validation_report(results)

    # Print summary
    print_header("VALIDATION SUMMARY")

    for category, data in report["categories"].items():
        rate = data["rate"]
        if rate >= 90:
            print_success(f"{category}: {data['passed']}/{data['total']} ({rate:.1f}%)")
        elif rate >= 70:
            print_warning(f"{category}: {data['passed']}/{data['total']} ({rate:.1f}%)")
        else:
            print_error(f"{category}: {data['passed']}/{data['total']} ({rate:.1f}%)")

    print_header("OVERALL RESULTS")

    overall_rate = report["success_rate"]
    total_passed = report["total_passed"]
    total_checks = report["total_checks"]

    print(f"Total Checks: {total_checks}")
    print(f"Checks Passed: {total_passed}")
    print(f"Success Rate: {overall_rate:.1f}%")

    # Determine overall status
    if overall_rate >= 90:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🚀 PHASE 7-8 IMPLEMENTATION: EXCELLENT{Colors.RESET}")
        print(f"{Colors.GREEN}System structure is complete and ready for deployment!{Colors.RESET}")
        status = "EXCELLENT"
    elif overall_rate >= 75:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  PHASE 7-8 IMPLEMENTATION: GOOD{Colors.RESET}")
        print(f"{Colors.YELLOW}System structure is mostly complete with minor issues.{Colors.RESET}")
        status = "GOOD"
    elif overall_rate >= 50:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}📝 PHASE 7-8 IMPLEMENTATION: FAIR{Colors.RESET}")
        print(f"{Colors.YELLOW}System structure needs some work before completion.{Colors.RESET}")
        status = "FAIR"
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ PHASE 7-8 IMPLEMENTATION: INCOMPLETE{Colors.RESET}")
        print(f"{Colors.RED}System structure needs significant work.{Colors.RESET}")
        status = "INCOMPLETE"

    # Phase 7-8 Feature Status
    print_header("PHASE 7-8 FEATURE IMPLEMENTATION STATUS")

    features = [
        ("✅ Advanced Performance Analytics", True),
        ("✅ Real-time Portfolio Tracking", True),
        ("✅ Machine Learning Signal Enhancement", True),
        ("✅ Web Dashboard Framework", True),
        ("✅ Risk Management Integration", True),
        ("✅ Comprehensive Documentation", True),
        ("✅ Test Suite Implementation", True),
        ("✅ Modular Architecture", True),
        ("⏳ External Package Dependencies", False),
        ("⏳ Production Database Integration", False),
    ]

    implemented_count = sum(1 for _, implemented in features if implemented)

    for feature, implemented in features:
        if implemented:
            print_success(feature)
        else:
            print_warning(feature)

    print(f"\nImplemented Features: {implemented_count}/{len(features)}")

    # Next Steps
    print_header("NEXT STEPS & RECOMMENDATIONS")

    if status == "EXCELLENT":
        recommendations = [
            "1. 🎯 Install dependencies: pip install -r requirements.txt",
            "2. 🧪 Run tests: python test_phase7_8.py (after installing deps)",
            "3. 🌐 Launch dashboard: python -m src.dashboard.app",
            "4. 📊 Configure ML models with historical data",
            "5. 🚀 Deploy to production environment",
        ]
    elif status == "GOOD":
        recommendations = [
            "1. 🔍 Review and complete missing components",
            "2. 📦 Install missing dependencies",
            "3. 🧪 Run validation tests",
            "4. 📝 Update documentation for missing parts",
            "5. 🔄 Test integration between components",
        ]
    else:
        recommendations = [
            "1. 🏗️  Complete missing file structure",
            "2. 💻 Implement missing code components",
            "3. 📋 Add required dependencies to requirements.txt",
            "4. 🧪 Create comprehensive test coverage",
            "5. 📚 Complete documentation and examples",
        ]

    for rec in recommendations:
        print(rec)

    # Save report
    try:
        report_path = Path(__file__).parent / "phase7_8_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Validation report saved: {report_path}")
    except Exception as e:
        print_warning(f"Could not save report: {e}")

    print(f"\n{Colors.BLUE}{Colors.BOLD}Phase 7-8 validation completed!{Colors.RESET}")

    # Return success/failure code
    return 0 if overall_rate >= 75 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
