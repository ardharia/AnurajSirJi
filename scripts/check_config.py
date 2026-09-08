"""Verify configuration loads successfully."""

from __future__ import annotations

from seismic_damage.config import get_settings, reset_settings_cache


def main() -> None:
    reset_settings_cache()
    settings = get_settings()
    print(f"app={settings.app.name} v{settings.app.version}")
    print(f"rag.enabled={settings.rag.enabled} model={settings.rag.embedding_model}")
    print(f"vlm.enabled={settings.vlm.enabled} model={settings.vlm.model}")
    print(f"fragility.method={settings.fragility.method.value}")


if __name__ == "__main__":
    main()
