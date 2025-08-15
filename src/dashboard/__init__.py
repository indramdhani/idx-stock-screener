"""
Web Dashboard Module for Indonesian Stock Screener

This module provides a comprehensive web-based dashboard interface for:
- Real-time portfolio monitoring
- Performance analytics visualization
- Signal management and tracking
- Risk monitoring and alerts
- Interactive charts and reports
- Strategy performance comparison
- Backtesting results visualization

Author: IDX Stock Screener Team
Version: 1.0.0
"""

from .app import create_app, DashboardApp

__all__ = [
    'create_app',
    'DashboardApp'
]

__version__ = '1.0.0'

__all__ = [
    'create_app',
    'DashboardApp',
    'main_routes',
    'api_routes',
    'portfolio_routes',
    'DashboardState',
    'ChartData',
    'AlertConfig',
    'ChartGenerator',
    'ChartType',
    'AlertManager',
    'AlertType',
    'format_currency',
    'format_percentage',
    'calculate_performance'
]

__version__ = '1.0.0'
