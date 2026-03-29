"""Model routing: evaluation and fit analysis tier selection."""

import pytest


@pytest.fixture(autouse=True)
def _clear_eval_env(monkeypatch):
    monkeypatch.delenv("USE_HIGH_QUALITY_EVAL", raising=False)
    monkeypatch.delenv("USE_HIGH_QUALITY_FIT_ANALYSIS", raising=False)
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_CHAT_MODEL_EVAL_HIGH", raising=False)
    monkeypatch.delenv("MODEL_FAST", raising=False)
    monkeypatch.delenv("MODEL_HIGH_QUALITY", raising=False)


def test_default_uses_same_model_for_eval_as_chat(monkeypatch):
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    from importlib import reload

    import app.core.config as cfg

    reload(cfg)
    assert cfg.settings.use_high_quality_eval is False
    assert cfg.settings.openai_eval_chat_model == "gpt-4o-mini"


def test_high_quality_eval_switches_model(monkeypatch):
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_CHAT_MODEL_EVAL_HIGH", "gpt-4o")
    monkeypatch.setenv("USE_HIGH_QUALITY_EVAL", "true")
    from importlib import reload

    import app.core.config as cfg

    reload(cfg)
    assert cfg.settings.use_high_quality_eval is True
    assert cfg.settings.openai_eval_chat_model == "gpt-4o"


def test_use_high_quality_eval_env_false(monkeypatch):
    monkeypatch.setenv("USE_HIGH_QUALITY_EVAL", "false")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    from importlib import reload

    import app.core.config as cfg

    reload(cfg)
    assert cfg.settings.use_high_quality_eval is False
    assert cfg.settings.openai_eval_chat_model == "gpt-4o-mini"


def test_fit_analysis_uses_model_fast_by_default(monkeypatch):
    monkeypatch.setenv("MODEL_FAST", "gpt-4o-mini")
    monkeypatch.setenv("MODEL_HIGH_QUALITY", "gpt-4o")
    from importlib import reload

    import app.core.config as cfg

    reload(cfg)
    assert cfg.settings.use_high_quality_fit_analysis is False
    assert cfg.settings.chat_model_fit_analysis() == "gpt-4o-mini"


def test_fit_analysis_high_quality_flag(monkeypatch):
    monkeypatch.setenv("MODEL_FAST", "gpt-4o-mini")
    monkeypatch.setenv("MODEL_HIGH_QUALITY", "gpt-4o")
    monkeypatch.setenv("USE_HIGH_QUALITY_FIT_ANALYSIS", "true")
    from importlib import reload

    import app.core.config as cfg

    reload(cfg)
    assert cfg.settings.use_high_quality_fit_analysis is True
    assert cfg.settings.chat_model_fit_analysis() == "gpt-4o"
