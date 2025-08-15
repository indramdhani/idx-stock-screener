"""
Routes Package

This package contains all the route definitions for the dashboard application.
Routes are organized into different blueprints:
- main_routes: Main page routes
- api_routes: API endpoints
- portfolio_routes: Portfolio management routes
"""

from .main_routes import main_routes
from .api_routes import api_routes
from .portfolio_routes import portfolio_routes

__all__ = ['main_routes', 'api_routes', 'portfolio_routes']
