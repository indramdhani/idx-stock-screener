"""
Flask Web Application for Indonesian Stock Screener Dashboard

This module provides a comprehensive web-based dashboard interface for:
- Real-time portfolio monitoring
- Performance analytics visualization
- Signal management and tracking
- Risk monitoring and alerts
- Interactive charts and reports

Author: IDX Stock Screener Team
Version: 1.0.0
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import sys
from pathlib import Path

# Configure logger before imports
from loguru import logger

# Web framework imports
try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    logger.warning("Flask/SocketIO not available. Web dashboard disabled.")

# Local imports
from ..analytics.portfolio_tracker import PortfolioTracker, PortfolioState
from ..analytics.performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from ..config.settings import TradingConfig

# Import routes after other dependencies
from .routes import main_routes, api_routes, portfolio_routes
from ..utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


class DashboardConfig:
    """Dashboard configuration settings"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'idx-screener-dashboard-key')
    HOST = os.environ.get('DASHBOARD_HOST', '127.0.0.1')
    PORT = int(os.environ.get('DASHBOARD_PORT', 5000))
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    # Dashboard settings
    UPDATE_INTERVAL = 30  # seconds
    MAX_CHART_POINTS = 100
    ALERT_THRESHOLD = 0.05  # 5% change threshold


class DashboardApp:
    """Main Dashboard Application Class"""

    def __init__(
        self,
        portfolio_tracker: Optional[PortfolioTracker] = None,
        performance_analyzer: Optional[PerformanceAnalyzer] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize Dashboard Application

        Args:
            portfolio_tracker: Portfolio tracker instance
            performance_analyzer: Performance analyzer instance
            config_path: Path to configuration file
        """
        if not WEB_AVAILABLE:
            raise ImportError("Web framework not available. Install with: pip install flask flask-cors flask-socketio plotly")

        self.portfolio_tracker = portfolio_tracker
        self.performance_analyzer = performance_analyzer or PerformanceAnalyzer()

        # Load configuration
        try:
            self.config = TradingConfig.from_yaml(config_path) if config_path else TradingConfig()
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            self.config = TradingConfig()

        # Initialize Flask app
        self.app = Flask(__name__,
                        template_folder=str(Path(__file__).parent / 'templates'),
                        static_folder=str(Path(__file__).parent / 'static'))

        self.app.config.from_object(DashboardConfig)

        # Initialize extensions
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Dashboard state
        self.dashboard_state = {
            'last_update': datetime.now(),
            'active_signals': [],
            'portfolio_value': 0.0,
            'daily_pnl': 0.0,
            'total_positions': 0,
            'alerts': []
        }

        # Setup routes
        self._setup_routes()
        self._setup_socket_handlers()

        # Background task flag
        self._background_task_started = False

        logger.info("Dashboard application initialized")

    def _setup_routes(self):
        """Setup Flask routes"""
        if not WEB_AVAILABLE:
            logger.warning("Web framework not available, skipping route setup")
            return

        # Store dashboard app instance in Flask app context first
        self.app.dashboard_app = self
        self.app.dashboard_state = self.dashboard_state

        # Then register blueprints
        try:
            self.app.register_blueprint(main_routes)
            self.app.register_blueprint(api_routes)
            self.app.register_blueprint(portfolio_routes)
            logger.info("Successfully registered all route blueprints")
        except Exception as e:
            logger.error(f"Error registering blueprints: {e}")
            raise

    def _setup_socket_handlers(self):
        """Setup WebSocket handlers for real-time updates"""

        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info("Client connected to dashboard")
            emit('status', {'message': 'Connected to IDX Stock Screener Dashboard'})

            # Send initial data
            emit('portfolio_update', self._get_portfolio_state())
            emit('alerts_update', self.dashboard_state['alerts'])

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("Client disconnected from dashboard")

        @self.socketio.on('request_update')
        def handle_update_request():
            """Handle manual update request"""
            logger.info("Manual update requested")
            self._update_dashboard_data()
            emit('portfolio_update', self._get_portfolio_state())

    def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state for frontend"""
        try:
            if self.portfolio_tracker:
                state = self.portfolio_tracker.current_state
                return {
                    'total_value': state.total_portfolio_value,
                    'cash_available': state.cash_available,
                    'invested_capital': state.invested_capital,
                    'total_pnl': state.total_pnl,
                    'total_pnl_pct': state.total_pnl_pct,
                    'day_pnl': state.day_pnl,
                    'day_pnl_pct': state.day_pnl_pct,
                    'open_positions': state.open_positions,
                    'total_positions': state.total_positions,
                    'risk_utilization': state.risk_utilization_pct,
                    'last_update': datetime.now().isoformat()
                }
            else:
                return {
                    'total_value': 0,
                    'cash_available': 0,
                    'invested_capital': 0,
                    'total_pnl': 0,
                    'total_pnl_pct': 0,
                    'day_pnl': 0,
                    'day_pnl_pct': 0,
                    'open_positions': 0,
                    'total_positions': 0,
                    'risk_utilization': 0,
                    'last_update': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting portfolio state: {e}")
            return {}

    def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance analysis data"""
        try:
            if self.portfolio_tracker and len(self.portfolio_tracker.daily_returns) > 0:
                import pandas as pd
                returns_series = pd.Series(self.portfolio_tracker.daily_returns)
                metrics = self.performance_analyzer.analyze_portfolio_performance(returns_series)

                return {
                    'total_return': metrics.total_return,
                    'annualized_return': metrics.annualized_return,
                    'volatility': metrics.annualized_volatility,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'sortino_ratio': metrics.sortino_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'total_trades': metrics.total_trades,
                    'profit_factor': metrics.profit_factor
                }
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            return {}

    def _generate_portfolio_value_chart(self) -> Dict[str, Any]:
        """Generate portfolio value chart data"""
        try:
            if not self.portfolio_tracker or len(self.portfolio_tracker.daily_values) < 2:
                return {'data': [], 'layout': {}}

            # Get recent data (last 30 days or all data if less)
            values = self.portfolio_tracker.daily_values[-30:]
            dates = [(datetime.now() - timedelta(days=len(values)-i-1)) for i in range(len(values))]

            # Create Plotly chart
            trace = go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            )

            layout = go.Layout(
                title='Portfolio Value Over Time',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Value (IDR)', tickformat=',.0f'),
                hovermode='x unified',
                template='plotly_white'
            )

            fig = go.Figure(data=[trace], layout=layout)
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))

        except Exception as e:
            logger.error(f"Error generating portfolio value chart: {e}")
            return {'data': [], 'layout': {}}

    def _generate_performance_chart(self) -> Dict[str, Any]:
        """Generate performance comparison chart"""
        try:
            if not self.portfolio_tracker or len(self.portfolio_tracker.daily_returns) < 2:
                return {'data': [], 'layout': {}}

            # Calculate cumulative returns
            returns = self.portfolio_tracker.daily_returns[-30:]  # Last 30 days
            cum_returns = [(1 + sum(returns[:i+1])) for i in range(len(returns))]
            dates = [(datetime.now() - timedelta(days=len(returns)-i-1)) for i in range(len(returns))]

            # Portfolio performance
            trace1 = go.Scatter(
                x=dates,
                y=cum_returns,
                mode='lines',
                name='Portfolio',
                line=dict(color='#2ca02c', width=2)
            )

            # Benchmark (flat line at 1.0 for now)
            trace2 = go.Scatter(
                x=dates,
                y=[1.0] * len(dates),
                mode='lines',
                name='Benchmark',
                line=dict(color='#d62728', width=1, dash='dash')
            )

            layout = go.Layout(
                title='Cumulative Performance Comparison',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Cumulative Return', tickformat='.2%'),
                hovermode='x unified',
                template='plotly_white'
            )

            fig = go.Figure(data=[trace1, trace2], layout=layout)
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))

        except Exception as e:
            logger.error(f"Error generating performance chart: {e}")
            return {'data': [], 'layout': {}}

    def _generate_positions_chart(self) -> Dict[str, Any]:
        """Generate positions allocation pie chart"""
        try:
            if not self.portfolio_tracker:
                return {'data': [], 'layout': {}}

            positions = self.portfolio_tracker.get_open_positions()
            if not positions:
                return {'data': [], 'layout': {}}

            # Calculate position values and create chart data
            symbols = []
            values = []
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            for i, position in enumerate(positions):
                symbols.append(position.symbol)
                values.append(position.get_position_value())

            trace = go.Pie(
                labels=symbols,
                values=values,
                marker=dict(colors=colors[:len(symbols)]),
                textinfo='label+percent',
                textposition='inside'
            )

            layout = go.Layout(
                title='Portfolio Allocation by Position',
                template='plotly_white'
            )

            fig = go.Figure(data=[trace], layout=layout)
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))

        except Exception as e:
            logger.error(f"Error generating positions chart: {e}")
            return {'data': [], 'layout': {}}

    def _update_dashboard_data(self):
        """Update dashboard data and broadcast to connected clients"""
        try:
            # Update portfolio data
            if self.portfolio_tracker:
                asyncio.create_task(self.portfolio_tracker.update_portfolio_state())

                # Update dashboard state
                state = self.portfolio_tracker.current_state
                self.dashboard_state.update({
                    'last_update': datetime.now(),
                    'portfolio_value': state.total_portfolio_value,
                    'daily_pnl': state.day_pnl,
                    'total_positions': state.total_positions
                })

            # Check for alerts
            self._check_alerts()

            # Broadcast updates to connected clients
            self.socketio.emit('portfolio_update', self._get_portfolio_state())
            self.socketio.emit('alerts_update', self.dashboard_state['alerts'])

        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")

    def _check_alerts(self):
        """Check for portfolio alerts and notifications"""
        try:
            alerts = []

            if self.portfolio_tracker:
                state = self.portfolio_tracker.current_state

                # Check daily P&L alert
                if abs(state.day_pnl_pct) > DashboardConfig.ALERT_THRESHOLD:
                    alert_type = 'warning' if state.day_pnl_pct < 0 else 'info'
                    alerts.append({
                        'type': alert_type,
                        'message': f'Daily P&L: {state.day_pnl_pct:.2%}',
                        'timestamp': datetime.now().isoformat()
                    })

                # Check risk utilization
                if state.risk_utilization_pct > 0.8:  # 80% of max risk
                    alerts.append({
                        'type': 'warning',
                        'message': f'High risk utilization: {state.risk_utilization_pct:.1%}',
                        'timestamp': datetime.now().isoformat()
                    })

                # Check for positions approaching stop loss
                for position in self.portfolio_tracker.get_open_positions():
                    if position.stop_loss and position.current_price:
                        distance_to_stop = (position.current_price - position.stop_loss) / position.current_price
                        if distance_to_stop < 0.02:  # Within 2% of stop loss
                            alerts.append({
                                'type': 'danger',
                                'message': f'{position.symbol} approaching stop loss',
                                'timestamp': datetime.now().isoformat()
                            })

            # Update alerts (keep only recent alerts)
            current_time = datetime.now()
            self.dashboard_state['alerts'] = [
                alert for alert in alerts
                if (current_time - datetime.fromisoformat(alert['timestamp'])).seconds < 3600  # 1 hour
            ]

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    def run(self, host: str = None, port: int = None, debug: bool = None):
        """Run the dashboard application"""
        try:
            host = host or DashboardConfig.HOST
            port = port or DashboardConfig.PORT
            debug = debug if debug is not None else DashboardConfig.DEBUG

            logger.info(f"Starting IDX Stock Screener Dashboard at http://{host}:{port}")

            # Start background update task
            if not self._background_task_started:
                self.socketio.start_background_task(self._background_update_task)
                self._background_task_started = True

            # Run the application
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=debug,
                allow_unsafe_werkzeug=True
            )

        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
            raise

    def _background_update_task(self):
        """Background task for periodic updates"""
        while True:
            try:
                self.socketio.sleep(DashboardConfig.UPDATE_INTERVAL)
                self._update_dashboard_data()
            except Exception as e:
                logger.error(f"Error in background update task: {e}")
                self.socketio.sleep(60)  # Wait longer on error


def create_app(
    portfolio_tracker: Optional[PortfolioTracker] = None,
    performance_analyzer: Optional[PerformanceAnalyzer] = None,
    config_path: Optional[Path] = None
) -> DashboardApp:
    """
    Factory function to create dashboard application

    Args:
        portfolio_tracker: Portfolio tracker instance
        performance_analyzer: Performance analyzer instance
        config_path: Path to configuration file

    Returns:
        Configured DashboardApp instance
    """
    return DashboardApp(
        portfolio_tracker=portfolio_tracker,
        performance_analyzer=performance_analyzer,
        config_path=config_path
    )


# CLI entry point for running dashboard
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='IDX Stock Screener Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', type=Path, help='Configuration file path')

    args = parser.parse_args()

    # Setup logging
    setup_logger(level=logging.DEBUG if args.debug else logging.INFO)

    # Create and run dashboard
    app = create_app(config_path=args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)
