"""TDD RED Phase: Tests for CI/CD configuration following ADR-005."""

import yaml
from pathlib import Path


def test_ci_uses_latest_actions():
    """Test CI uses latest non-deprecated actions."""
    # This will fail - current ci.yml uses deprecated actions (RED phase)
    ci_path = Path(".github/workflows/ci.yml")

    with open(ci_path) as f:
        ci_config = yaml.safe_load(f)

    # Check setup-python version
    for job_name, job_config in ci_config["jobs"].items():
        if "steps" in job_config:
            for step in job_config["steps"]:
                if "uses" in step:
                    uses = step["uses"]

                    # Check for deprecated actions
                    assert (
                        "actions/setup-python@v5" in uses
                        or "actions/setup-python" not in uses
                    ), "setup-python should be v5"
                    assert (
                        "codecov/codecov-action@v5" in uses
                        or "codecov/codecov-action" not in uses
                    ), "codecov should be v5"
                    assert (
                        "actions/upload-artifact@v4" in uses
                        or "actions/upload-artifact" not in uses
                    ), "upload-artifact should be v4"


def test_ci_uses_ruff_instead_of_separate_tools():
    """Test CI uses ruff instead of separate flake8/black/isort."""
    ci_path = Path(".github/workflows/ci.yml")

    with open(ci_path) as f:
        ci_config = yaml.safe_load(f)

    # Check that ruff is used for linting and formatting
    test_job = ci_config["jobs"]["test"]
    steps = test_job["steps"]

    # Should have ruff check step
    ruff_check_found = False
    ruff_format_found = False

    for step in steps:
        if "run" in step:
            run_command = step["run"]
            if "ruff check" in run_command:
                ruff_check_found = True
            if "ruff format" in run_command:
                ruff_format_found = True

    assert ruff_check_found, "Should use 'ruff check' instead of flake8"
    assert ruff_format_found, "Should use 'ruff format' instead of black/isort"


def test_ci_has_deploy_job():
    """Test CI has deploy job for main branch."""
    ci_path = Path(".github/workflows/ci.yml")

    with open(ci_path) as f:
        ci_config = yaml.safe_load(f)

    # Should have deploy job
    assert "deploy" in ci_config["jobs"], "Should have deploy job"

    deploy_job = ci_config["jobs"]["deploy"]

    # Deploy should run on main branch only
    assert deploy_job["if"] == "github.ref == 'refs/heads/main'", (
        "Deploy should only run on main"
    )

    # Deploy should depend on test and security
    assert "needs" in deploy_job, "Deploy should have dependencies"
    assert set(deploy_job["needs"]) == {"test", "security"}, (
        "Deploy should need test and security"
    )


def test_ci_has_fallback_for_codecov():
    """Test CI has fallback for codecov failures."""
    ci_path = Path(".github/workflows/ci.yml")

    with open(ci_path) as f:
        ci_config = yaml.safe_load(f)

    test_job = ci_config["jobs"]["test"]
    steps = test_job["steps"]

    # Find codecov step
    codecov_step = None
    for step in steps:
        if "uses" in step and "codecov/codecov-action" in step["uses"]:
            codecov_step = step
            break

    assert codecov_step is not None, "Should have codecov step"
    assert codecov_step.get("continue-on-error", False) is True, (
        "Codecov should not fail pipeline"
    )


def test_ci_uploads_test_artifacts():
    """Test CI uploads test artifacts properly."""
    ci_path = Path(".github/workflows/ci.yml")

    with open(ci_path) as f:
        ci_config = yaml.safe_load(f)

    test_job = ci_config["jobs"]["test"]
    steps = test_job["steps"]

    # Should upload test artifacts
    upload_found = False
    for step in steps:
        if "uses" in step and "actions/upload-artifact" in step["uses"]:
            upload_found = True
            # Check if uploads coverage files
            if "with" in step and "path" in step["with"]:
                paths = step["with"]["path"]
                if isinstance(paths, str):
                    paths = [paths]
                assert any("coverage.xml" in path for path in paths), (
                    "Should upload coverage.xml"
                )
                assert any("htmlcov/" in path for path in paths), (
                    "Should upload htmlcov/"
                )

    assert upload_found, "Should upload test artifacts"


def test_ci_has_proper_environment_setup():
    """Test CI has proper environment configuration."""
    ci_path = Path(".github/workflows/ci.yml")

    with open(ci_path) as f:
        ci_config = yaml.safe_load(f)

    # Check deploy job has environment
    if "deploy" in ci_config["jobs"]:
        deploy_job = ci_config["jobs"]["deploy"]
        assert deploy_job.get("environment") == "production", (
            "Deploy should use production environment"
        )
