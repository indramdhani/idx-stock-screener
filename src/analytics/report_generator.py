"""
Report Generator Module for Indonesian Stock Screener

This module provides functionality for generating detailed reports and analytics
for trading performance, portfolio status, and market analysis.

Author: IDX Stock Screener Team
Version: 1.0.0
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from ..data.models.signal import TradingSignal
from .performance_analyzer import PerformanceMetrics
from .portfolio_tracker import PortfolioState


@dataclass
class ReportConfig:
    """Configuration for report generation"""

    output_dir: Path
    include_charts: bool = True
    export_formats: List[str] = None  # e.g., ["pdf", "html", "csv"]
    template_path: Optional[Path] = None
    custom_metrics: Dict[str, str] = None  # Custom metric name -> calculation method

    def __post_init__(self):
        """Validate and initialize report configuration"""
        if self.export_formats is None:
            self.export_formats = ["pdf"]

        if self.custom_metrics is None:
            self.custom_metrics = {}

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)


class ReportGenerator:
    """Generator for trading and portfolio performance reports"""

    def __init__(self, config: ReportConfig):
        """Initialize report generator with configuration"""
        self.config = config
        self.performance_metrics = None
        self.portfolio_state = None
        logger.info(f"Report generator initialized with config: {config}")

    def set_data(self,
                 performance_metrics: PerformanceMetrics = None,
                 portfolio_state: PortfolioState = None,
                 signals: List[TradingSignal] = None):
        """Set data sources for report generation"""
        self.performance_metrics = performance_metrics
        self.portfolio_state = portfolio_state
        self.signals = signals if signals else []

    def generate_daily_report(self, date: datetime = None) -> Path:
        """Generate daily performance report"""
        if not date:
            date = datetime.now()

        logger.info(f"Generating daily report for {date.date()}")

        try:
            # Generate report filename
            filename = f"daily_report_{date.strftime('%Y%m%d')}"
            report_path = self.config.output_dir / filename

            # Create report sections
            sections = {
                "portfolio_summary": self._generate_portfolio_summary(),
                "daily_performance": self._generate_daily_performance(date),
                "trading_signals": self._generate_signal_summary(),
            }

            # Export in requested formats
            paths = []
            for format in self.config.export_formats:
                output_path = self._export_report(sections, format, report_path)
                paths.append(output_path)

            logger.info(f"Daily report generated successfully: {paths}")
            return paths[0]  # Return primary format path

        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            raise

    def generate_monthly_report(self, year: int, month: int) -> Path:
        """Generate monthly performance report"""
        logger.info(f"Generating monthly report for {year}-{month}")

        try:
            # Implementation for monthly report
            filename = f"monthly_report_{year}{month:02d}"
            report_path = self.config.output_dir / filename

            # TODO: Implement monthly report generation
            logger.warning("Monthly report generation not yet implemented")
            return report_path

        except Exception as e:
            logger.error(f"Failed to generate monthly report: {e}")
            raise

    def _generate_portfolio_summary(self) -> Dict:
        """Generate portfolio summary section"""
        if not self.portfolio_state:
            return {}

        return {
            "total_value": self.portfolio_state.total_value,
            "cash_balance": self.portfolio_state.cash_balance,
            "positions": len(self.portfolio_state.positions),
            "unrealized_pnl": self.portfolio_state.unrealized_pnl,
            "realized_pnl": self.portfolio_state.realized_pnl
        }

    def _generate_daily_performance(self, date: datetime) -> Dict:
        """Generate daily performance metrics"""
        if not self.performance_metrics:
            return {}

        return {
            "date": date.date(),
            "total_trades": self.performance_metrics.total_signals,
            "winning_trades": self.performance_metrics.winning_signals,
            "win_rate": self.performance_metrics.win_rate,
            "total_pnl": self.performance_metrics.total_pnl,
            "average_pnl": self.performance_metrics.average_pnl
        }

    def _generate_signal_summary(self) -> Dict:
        """Generate summary of trading signals"""
        if not self.signals:
            return {}

        # Group signals by type
        signal_summary = {}
        for signal in self.signals:
            signal_type = signal.signal_type.value
            if signal_type not in signal_summary:
                signal_summary[signal_type] = {"count": 0, "symbols": []}

            signal_summary[signal_type]["count"] += 1
            signal_summary[signal_type]["symbols"].append(signal.symbol)

        return signal_summary

    def _export_report(self,
                      sections: Dict,
                      format: str,
                      base_path: Path) -> Path:
        """Export report in specified format"""
        output_path = base_path.with_suffix(f".{format}")

        if format == "csv":
            # Export as CSV
            df = pd.DataFrame(sections)
            df.to_csv(output_path, index=False)

        elif format == "html":
            # Basic HTML export
            html_content = "<html><body>"
            for section, data in sections.items():
                html_content += f"<h2>{section}</h2>"
                html_content += f"<pre>{data}</pre>"
            html_content += "</body></html>"

            with open(output_path, "w") as f:
                f.write(html_content)

        elif format == "pdf":
            logger.warning("PDF export not yet implemented")
            # TODO: Implement PDF export

        else:
            raise ValueError(f"Unsupported export format: {format}")

        return output_path
