"""
Portfolio Routes Module

This module contains portfolio-specific routes for the stock screener dashboard.
These endpoints handle portfolio management, position tracking, and trade execution.
"""

from datetime import datetime
from flask import Blueprint, jsonify, request, current_app
from loguru import logger

from src.data.models.signal import TradingSignal
from src.data.models.portfolio import Position, PositionStatus
from src.analytics.portfolio_tracker import PortfolioTracker

# Create Blueprint
portfolio_routes = Blueprint('portfolio_routes', __name__, url_prefix='/portfolio')

@portfolio_routes.route('/portfolio/positions', methods=['GET'])
def get_positions():
    """Get all portfolio positions"""
    try:
        if current_app.dashboard_app.portfolio_tracker:
            positions = current_app.dashboard_app.portfolio_tracker.get_all_positions()
            return jsonify([pos.to_dict() for pos in positions])
        return jsonify([])
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return jsonify({"error": str(e)}), 500

@portfolio_routes.route('/portfolio/positions/open', methods=['GET'])
def get_open_positions():
    """Get open portfolio positions"""
    try:
        if current_app.dashboard_app.portfolio_tracker:
            positions = current_app.dashboard_app.portfolio_tracker.get_open_positions()
            return jsonify([pos.to_dict() for pos in positions])
        return jsonify([])
    except Exception as e:
        logger.error(f"Error fetching open positions: {e}")
        return jsonify({"error": str(e)}), 500

@portfolio_routes.route('/portfolio/positions/<symbol>', methods=['GET'])
def get_position(symbol):
    """Get position details for a specific symbol"""
    try:
        if current_app.dashboard_app.portfolio_tracker:
            position = current_app.dashboard_app.portfolio_tracker.get_position(symbol)
            if position:
                return jsonify(position.to_dict())
        return jsonify({"error": "Position not found"}), 404
    except Exception as e:
        logger.error(f"Error fetching position for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@portfolio_routes.route('/portfolio/trades', methods=['GET'])
def get_trades():
    """Get trade history"""
    try:
        if current_app.dashboard_app.portfolio_tracker:
            trades = current_app.dashboard_app.portfolio_tracker.get_trade_history()
            return jsonify([trade.to_dict() for trade in trades])
        return jsonify([])
    except Exception as e:
        logger.error(f"Error fetching trade history: {e}")
        return jsonify({"error": str(e)}), 500

@portfolio_routes.route('/portfolio/performance', methods=['GET'])
def get_performance():
    """Get portfolio performance metrics"""
    try:
        if current_app.dashboard_app.portfolio_tracker:
            performance = current_app.dashboard_app.portfolio_tracker.get_performance_metrics()
            return jsonify(performance)
        return jsonify({"error": "Portfolio tracker not available"}), 404
    except Exception as e:
        logger.error(f"Error fetching performance metrics: {e}")
        return jsonify({"error": str(e)}), 500

@portfolio_routes.route('/portfolio/signals/active', methods=['GET'])
def get_active_signals():
    """Get active trading signals"""
    try:
        if current_app.dashboard_app.portfolio_tracker:
            signals = current_app.dashboard_app.portfolio_tracker.get_active_signals()
            return jsonify([signal.to_dict() for signal in signals])
        return jsonify([])
    except Exception as e:
        logger.error(f"Error fetching active signals: {e}")
        return jsonify({"error": str(e)}), 500

@portfolio_routes.route('/portfolio/risk', methods=['GET'])
def get_risk_metrics():
    """Get portfolio risk metrics"""
    try:
        if current_app.dashboard_app.portfolio_tracker:
            risk_metrics = current_app.dashboard_app.portfolio_tracker.get_risk_metrics()
            return jsonify(risk_metrics)
        return jsonify({"error": "Portfolio tracker not available"}), 404
    except Exception as e:
        logger.error(f"Error fetching risk metrics: {e}")
        return jsonify({"error": str(e)}), 500
