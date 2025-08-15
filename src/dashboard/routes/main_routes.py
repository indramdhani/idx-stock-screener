"""
Main Routes Module

This module contains the main page routes for the stock screener dashboard.
"""

from datetime import datetime
from flask import Blueprint, render_template, current_app
from loguru import logger

# Create Blueprint
main_routes = Blueprint('main_routes', __name__, url_prefix='')

@main_routes.route('/')
def index():
    """Main dashboard page"""
    try:
        return render_template('dashboard.html',
                             config=current_app.dashboard_app.dashboard_state,
                             last_update=datetime.now())
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return str(e), 500

@main_routes.route('/portfolio')
def portfolio():
    """Portfolio management page"""
    try:
        return render_template('portfolio.html',
                             portfolio_state=current_app.dashboard_app._get_portfolio_state())
    except Exception as e:
        logger.error(f"Error rendering portfolio page: {e}")
        return str(e), 500

@main_routes.route('/analytics')
def analytics():
    """Performance analytics page"""
    try:
        return render_template('analytics.html',
                             performance_data=current_app.dashboard_app._get_performance_data())
    except Exception as e:
        logger.error(f"Error rendering analytics page: {e}")
        return str(e), 500

@main_routes.route('/signals')
def signals():
    """Trading signals page"""
    try:
        return render_template('signals.html',
                             signals=current_app.dashboard_app.dashboard_state['active_signals'])
    except Exception as e:
        logger.error(f"Error rendering signals page: {e}")
        return str(e), 500

@main_routes.route('/settings')
def settings():
    """Settings and configuration page"""
    try:
        return render_template('settings.html',
                             config=current_app.dashboard_app.config.dict())
    except Exception as e:
        logger.error(f"Error rendering settings page: {e}")
        return str(e), 500
