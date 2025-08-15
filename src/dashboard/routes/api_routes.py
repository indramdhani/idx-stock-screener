"""
API Routes Module

This module contains the API routes for the stock screener dashboard.
These endpoints return JSON responses for AJAX calls from the frontend.
"""

from datetime import datetime
from flask import Blueprint, jsonify, current_app
from loguru import logger

# Create Blueprint
api_routes = Blueprint('api_routes', __name__)

@api_routes.route('/api/portfolio/state')
def portfolio_state():
    """Get current portfolio state"""
    try:
        if current_app.dashboard_app.portfolio_tracker:
            state = current_app.dashboard_app.portfolio_tracker.current_state
            return jsonify(state.to_dict())
        else:
            return jsonify({'error': 'Portfolio tracker not available'}), 404
    except Exception as e:
        logger.error(f"Error getting portfolio state: {e}")
        return jsonify({'error': str(e)}), 500

@api_routes.route('/api/portfolio/positions')
def portfolio_positions():
    """Get current portfolio positions"""
    try:
        if current_app.dashboard_app.portfolio_tracker:
            positions = [p.to_dict() for p in current_app.dashboard_app.portfolio_tracker.get_open_positions()]
            return jsonify(positions)
        else:
            return jsonify([])
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return jsonify({'error': str(e)}), 500

@api_routes.route('/api/performance/metrics')
def performance_metrics():
    """Get performance metrics"""
    try:
        if (current_app.dashboard_app.portfolio_tracker and
            len(current_app.dashboard_app.portfolio_tracker.daily_returns) > 0):
            import pandas as pd
            returns_series = pd.Series(current_app.dashboard_app.portfolio_tracker.daily_returns)
            metrics = current_app.dashboard_app.performance_analyzer.analyze_portfolio_performance(returns_series)
            return jsonify(metrics.to_dict())
        else:
            return jsonify({'error': 'No performance data available'}), 404
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return jsonify({'error': str(e)}), 500

@api_routes.route('/api/charts/portfolio_value')
def chart_portfolio_value():
    """Get portfolio value chart data"""
    try:
        chart_data = current_app.dashboard_app._generate_portfolio_value_chart()
        return jsonify(chart_data)
    except Exception as e:
        logger.error(f"Error generating portfolio chart: {e}")
        return jsonify({'error': str(e)}), 500

@api_routes.route('/api/charts/performance')
def chart_performance():
    """Get performance chart data"""
    try:
        chart_data = current_app.dashboard_app._generate_performance_chart()
        return jsonify(chart_data)
    except Exception as e:
        logger.error(f"Error generating performance chart: {e}")
        return jsonify({'error': str(e)}), 500

@api_routes.route('/api/charts/positions')
def chart_positions():
    """Get positions allocation chart data"""
    try:
        chart_data = current_app.dashboard_app._generate_positions_chart()
        return jsonify(chart_data)
    except Exception as e:
        logger.error(f"Error generating positions chart: {e}")
        return jsonify({'error': str(e)}), 500

@api_routes.route('/api/alerts')
def alerts():
    """Get active alerts"""
    return jsonify(current_app.dashboard_state['alerts'])

@api_routes.route('/api/status')
def status():
    """Get system status"""
    status = {
        'status': 'running',
        'last_update': current_app.dashboard_state['last_update'].isoformat(),
        'portfolio_connected': current_app.dashboard_app.portfolio_tracker is not None,
        'total_positions': current_app.dashboard_state['total_positions'],
        'portfolio_value': current_app.dashboard_state['portfolio_value'],
        'daily_pnl': current_app.dashboard_state['daily_pnl']
    }
    return jsonify(status)
