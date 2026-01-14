"""FastAPI REST API for reqcheck."""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from reqcheck.core.analyzer import RequirementsAnalyzer
from reqcheck.core.config import Settings, get_settings
from reqcheck.core.models import (
    AnalysisReport,
    Issue,
    Requirement,
    RequirementType,
    ScoreBreakdown,
)
from reqcheck.output.formatters import format_checklist, format_markdown, format_summary

logger = logging.getLogger(__name__)


# Request/Response models
class RequirementRequest(BaseModel):
    """Request model for requirement analysis."""

    title: str = Field(..., min_length=1, description="Requirement title")
    description: str = Field(default="", description="Detailed description")
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="List of acceptance criteria",
    )
    type: RequirementType = Field(
        default=RequirementType.STORY,
        description="Type of requirement",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def to_requirement(self) -> Requirement:
        """Convert to internal Requirement model."""
        return Requirement(
            title=self.title,
            description=self.description,
            acceptance_criteria=self.acceptance_criteria,
            type=self.type,
            metadata=self.metadata,
        )


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""

    requirement_id: str
    requirement_title: str
    issues: list[Issue]
    scores: ScoreBreakdown
    summary: str
    recommendations: list[str]
    is_ready_for_dev: bool
    blocker_count: int
    warning_count: int
    suggestion_count: int

    @classmethod
    def from_report(cls, report: AnalysisReport) -> "AnalysisResponse":
        """Create response from analysis report."""
        return cls(
            requirement_id=report.requirement_id,
            requirement_title=report.requirement_title,
            issues=report.issues,
            scores=report.scores,
            summary=report.summary,
            recommendations=report.recommendations,
            is_ready_for_dev=report.is_ready_for_dev,
            blocker_count=report.blocker_count,
            warning_count=report.warning_count,
            suggestion_count=report.suggestion_count,
        )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    llm_available: bool


class BatchRequest(BaseModel):
    """Request for batch analysis."""

    requirements: list[RequirementRequest] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of requirements to analyze (max 10)",
    )


class BatchResponse(BaseModel):
    """Response for batch analysis."""

    results: list[AnalysisResponse]
    total: int
    ready_count: int
    blocker_count: int


# Global state
_analyzer: RequirementsAnalyzer | None = None
_settings: Settings | None = None

# Rate limiter - initialized with default settings, reconfigured in lifespan
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _analyzer, _settings
    _settings = get_settings()
    _analyzer = RequirementsAnalyzer(_settings)

    # Configure rate limiter based on settings
    if _settings.rate_limit_enabled:
        limiter.enabled = True
        logger.info(
            f"Rate limiting enabled: default={_settings.rate_limit_default}, "
            f"analyze={_settings.rate_limit_analyze}, batch={_settings.rate_limit_batch}"
        )
    else:
        limiter.enabled = False
        logger.info("Rate limiting disabled")

    logger.info("reqcheck API started")
    yield
    logger.info("reqcheck API shutting down")


# Create FastAPI app
app = FastAPI(
    title="reqcheck API",
    description="AI-powered requirements quality checker for user stories, tickets, and specifications",
    version="0.1.0",
    lifespan=lifespan,
)

# Add rate limiter state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_default_limit() -> str:
    """Get default rate limit from settings."""
    settings = _settings or get_settings()
    return settings.rate_limit_default


def _get_analyze_limit() -> str:
    """Get analyze rate limit from settings."""
    settings = _settings or get_settings()
    return settings.rate_limit_analyze


def _get_batch_limit() -> str:
    """Get batch rate limit from settings."""
    settings = _settings or get_settings()
    return settings.rate_limit_batch


@app.get("/health", response_model=HealthResponse)
@limiter.limit(_get_default_limit)
async def health_check(request: Request):
    """Check API health status."""
    settings = _settings or get_settings()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        llm_available=settings.llm_available,
    )


@app.post("/analyze", response_model=AnalysisResponse)
@limiter.limit(_get_analyze_limit)
async def analyze_requirement(request: Request, body: RequirementRequest):
    """
    Analyze a single requirement for quality issues.

    Returns detailed analysis including:
    - Quality issues found (blockers, warnings, suggestions)
    - Scores for ambiguity, completeness, and testability
    - Executive summary and recommendations
    """
    global _analyzer

    if _analyzer is None:
        _analyzer = RequirementsAnalyzer(get_settings())

    try:
        requirement = body.to_requirement()
        report = _analyzer.analyze(requirement)
        return AnalysisResponse.from_report(report)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch", response_model=BatchResponse)
@limiter.limit(_get_batch_limit)
async def analyze_batch(request: Request, body: BatchRequest):
    """
    Analyze multiple requirements in batch.

    Limited to 10 requirements per request.
    """
    global _analyzer

    if _analyzer is None:
        _analyzer = RequirementsAnalyzer(get_settings())

    results = []
    total_blockers = 0
    ready_count = 0

    for req in body.requirements:
        try:
            requirement = req.to_requirement()
            report = _analyzer.analyze(requirement)
            response = AnalysisResponse.from_report(report)
            results.append(response)

            total_blockers += response.blocker_count
            if response.is_ready_for_dev:
                ready_count += 1
        except Exception as e:
            logger.error(f"Analysis failed for '{req.title}': {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed for '{req.title}': {str(e)}",
            )

    return BatchResponse(
        results=results,
        total=len(results),
        ready_count=ready_count,
        blocker_count=total_blockers,
    )


@app.post("/analyze/markdown", response_class=PlainTextResponse)
@limiter.limit(_get_analyze_limit)
async def analyze_markdown(
    request: Request,
    body: RequirementRequest,
    include_evidence: bool = Query(default=True, description="Include evidence snippets"),
):
    """
    Analyze a requirement and return Markdown-formatted report.

    Useful for integration with documentation systems.
    """
    global _analyzer, _settings

    if _analyzer is None:
        _analyzer = RequirementsAnalyzer(get_settings())

    if _settings is None:
        _settings = get_settings()

    try:
        requirement = body.to_requirement()
        report = _analyzer.analyze(requirement)

        _settings.include_evidence = include_evidence
        return format_markdown(report, _settings)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/summary", response_class=PlainTextResponse)
@limiter.limit(_get_analyze_limit)
async def analyze_summary(request: Request, body: RequirementRequest):
    """
    Analyze a requirement and return brief summary.

    Useful for CI/CD integration or quick checks.
    """
    global _analyzer

    if _analyzer is None:
        _analyzer = RequirementsAnalyzer(get_settings())

    try:
        requirement = body.to_requirement()
        report = _analyzer.analyze(requirement)
        return format_summary(report)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/checklist", response_class=PlainTextResponse)
@limiter.limit(_get_analyze_limit)
async def analyze_checklist(request: Request, body: RequirementRequest):
    """
    Analyze a requirement and return checklist format.

    Useful for quick pass/fail checks.
    """
    global _analyzer

    if _analyzer is None:
        _analyzer = RequirementsAnalyzer(get_settings())

    try:
        requirement = body.to_requirement()
        report = _analyzer.analyze(requirement)
        return format_checklist(report)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create FastAPI app with custom settings (for testing)."""
    global _analyzer, _settings
    _settings = settings or get_settings()
    _analyzer = RequirementsAnalyzer(_settings)

    # Configure rate limiter based on settings
    limiter.enabled = _settings.rate_limit_enabled

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "reqcheck.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
