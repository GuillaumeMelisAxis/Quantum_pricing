"""STN-GPR replication package."""

from .config import PaperConfig, VolatilityExtendedConfig
from .coordinates import CoordinateTransform, MarketCoordinatePricer
from .greeks import (
    finite_difference_greeks,
    finite_difference_hybrid_greeks,
    geometric_basket_put_spot_greeks,
    project_symmetric_matrix_psd,
)
from .grid_geometry import (
    interval_fill_distance,
    local_coverage_metrics,
    normalized_node_displacement,
    physical_log_moneyness_nodes,
)
from .portfolio import BasketOptionTrade, Portfolio
from .pricers import (
    EuropeanArithmeticBasketVolPricer,
    EuropeanGeometricBasketVolPricer,
    arithmetic_basket_put_levy,
    basket_put_monte_carlo,
    geometric_basket_put,
)
from .risk import (
    MonteCarloVaREngine,
    ScenarioValuation,
    VaRResult,
    compare_loss_distributions,
    domain_coverage,
    expected_shortfall_contributions,
    var_es,
    var_es_uncertainty,
)
from .scenarios import GBMScenarioGenerator
from .vol_extension import (
    VOL_GRID_MODES,
    VOL_RISK_GRID_MODES,
    BoundedVolStandardizedRiskTransform,
    InterleavedVolTransform,
    VolExtendedPriceSurrogate,
    VolStandardizedRiskTransform,
    build_vol_extended_grid,
    build_vol_extended_risk_grid,
    build_vol_grid,
    sample_market_points,
    strike_axis_log_moneyness_nodes,
)

__all__ = [
    "VOL_GRID_MODES",
    "VOL_RISK_GRID_MODES",
    "BasketOptionTrade",
    "BoundedVolStandardizedRiskTransform",
    "CoordinateTransform",
    "EuropeanArithmeticBasketVolPricer",
    "EuropeanGeometricBasketVolPricer",
    "GBMScenarioGenerator",
    "InterleavedVolTransform",
    "MarketCoordinatePricer",
    "MonteCarloVaREngine",
    "PaperConfig",
    "Portfolio",
    "ScenarioValuation",
    "VaRResult",
    "VolExtendedPriceSurrogate",
    "VolStandardizedRiskTransform",
    "VolatilityExtendedConfig",
    "arithmetic_basket_put_levy",
    "basket_put_monte_carlo",
    "build_vol_extended_grid",
    "build_vol_extended_risk_grid",
    "build_vol_grid",
    "compare_loss_distributions",
    "domain_coverage",
    "expected_shortfall_contributions",
    "finite_difference_greeks",
    "finite_difference_hybrid_greeks",
    "geometric_basket_put",
    "geometric_basket_put_spot_greeks",
    "interval_fill_distance",
    "local_coverage_metrics",
    "normalized_node_displacement",
    "physical_log_moneyness_nodes",
    "project_symmetric_matrix_psd",
    "sample_market_points",
    "strike_axis_log_moneyness_nodes",
    "var_es",
    "var_es_uncertainty",
]
