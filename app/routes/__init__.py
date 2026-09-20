"""Route blueprints."""
from app.routes.auth import bp as auth_bp
from app.routes.candidate import bp as candidate_bp
from app.routes.hr import bp as hr_bp
from app.routes.ops import bp as ops_bp

__all__ = ["auth_bp", "candidate_bp", "hr_bp", "ops_bp"]
