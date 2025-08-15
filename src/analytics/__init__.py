"""
Performance Analytics Module for Indonesian Stock Screener

This module provides comprehensive performance analytics and reporting capabilities
for the Indonesian Stock Screener system, including:

- Portfolio performance analysis
- Strategy performance metrics
- Risk-adjusted returns
- Drawdown analysis
- Trade analysis and statistics
- Benchmark comparisons
- Performance reporting and visualization

Author: IDX Stock Screener Team
Version: 1.0.0
"""

from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from .portfolio_tracker import PortfolioTracker, PortfolioState
from .report_generator import ReportGenerator, ReportConfig
from .benchmark_analyzer import BenchmarkAnalyzer, BenchmarkComparison
from .risk_analytics import RiskAnalyzer, RiskMetrics

__all__ = [
    'PerformanceAnalyzer',
    'PerformanceMetrics',
    'PortfolioTracker',
    'PortfolioState',
    'ReportGenerator',
    'ReportConfig',
    'BenchmarkAnalyzer',
    'BenchmarkComparison',
    'RiskAnalyzer',
    'RiskMetrics'
]

__version__ = '1.0.0'
