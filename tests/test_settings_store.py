"""Tests for settings_store.py — per-slug settings persistence, using a
temp directory so tests never touch the real settings/ folder."""

import json
import time

import pytest

import settings_store as ss


@pytest.fixture(autouse=True)
def temp_settings_dir(tmp_path, monkeypatch):
    """Redirect SETTINGS_DIR/SETTINGS_FILE to a temp location for every test."""
    monkeypatch.setattr(ss, "SETTINGS_DIR", tmp_path / "settings")
    monkeypatch.setattr(ss, "SETTINGS_FILE", tmp_path / "legacy_settings.json")
    yield


class TestLoadSavedSettings:
    def test_missing_file_returns_empty_dict(self):
        assert ss.load_saved_settings("nobody") == {}

    def test_loads_own_slug_file(self):
        ss.SETTINGS_DIR.mkdir(parents=True)
        (ss.SETTINGS_DIR / "alice.json").write_text(json.dumps({"vosk_model": "en"}))
        assert ss.load_saved_settings("alice") == {"vosk_model": "en"}

    def test_seeds_new_slug_from_legacy_file(self):
        ss.SETTINGS_FILE.write_text(json.dumps({"vosk_model": "legacy-en"}))
        # "bob" has no per-slug file yet — should seed from the legacy file.
        result = ss.load_saved_settings("bob")
        assert result == {"vosk_model": "legacy-en"}

    def test_seeds_new_slug_from_main_session_over_legacy_file(self):
        # main's *current* settings take priority over the old legacy file
        # once main itself has a settings file — a new named session should
        # inherit from what main is configured with right now, not a stale
        # pre-named-sessions snapshot.
        ss.SETTINGS_FILE.write_text(json.dumps({"vosk_model": "legacy-en"}))
        ss.SETTINGS_DIR.mkdir(parents=True)
        (ss.SETTINGS_DIR / "main.json").write_text(json.dumps({"vosk_model": "main-current"}))
        result = ss.load_saved_settings("brand-new-session")
        assert result == {"vosk_model": "main-current"}

    def test_main_itself_does_not_seed_from_itself(self):
        # main.json doesn't exist yet — loading "main" must not try to seed
        # from "main" (infinite-loop-shaped edge case) and should fall
        # through to the legacy file / empty dict instead.
        ss.SETTINGS_FILE.write_text(json.dumps({"vosk_model": "legacy-en"}))
        result = ss.load_saved_settings("main")
        assert result == {"vosk_model": "legacy-en"}

    def test_new_session_diverges_independently_after_first_save(self):
        ss.SETTINGS_DIR.mkdir(parents=True)
        (ss.SETTINGS_DIR / "main.json").write_text(json.dumps({"vosk_model": "main-v1"}))
        # New session seeds from main...
        seeded = ss.load_saved_settings("khyretos")
        assert seeded == {"vosk_model": "main-v1"}
        # ...but once it's saved, it's independent — a later change to main
        # must not retroactively affect it.
        ss.persist_settings("khyretos", {"vosk_model": "khyretos-v1", "recognition_engine": "vosk"})
        import time as _time
        _time.sleep(1.3)
        (ss.SETTINGS_DIR / "main.json").write_text(json.dumps({"vosk_model": "main-v2"}))
        reloaded = ss.load_saved_settings("khyretos")
        assert reloaded["vosk_model"] == "khyretos-v1"

    def test_own_slug_file_takes_priority_over_legacy(self):
        ss.SETTINGS_FILE.write_text(json.dumps({"vosk_model": "legacy-en"}))
        ss.SETTINGS_DIR.mkdir(parents=True)
        (ss.SETTINGS_DIR / "carol.json").write_text(json.dumps({"vosk_model": "carol-es"}))
        assert ss.load_saved_settings("carol") == {"vosk_model": "carol-es"}

    def test_corrupt_file_returns_empty_dict_not_raises(self):
        ss.SETTINGS_DIR.mkdir(parents=True)
        (ss.SETTINGS_DIR / "dave.json").write_text("{not valid json")
        assert ss.load_saved_settings("dave") == {}


class TestVadThresholdMigration:
    def test_old_0_to_1_scale_migrated_to_db(self):
        ss.SETTINGS_DIR.mkdir(parents=True)
        (ss.SETTINGS_DIR / "eve.json").write_text(json.dumps({"vad_threshold": 0.3}))
        result = ss.load_saved_settings("eve")
        # Migrated value should be a negative dB figure, not the old 0-1 scale.
        assert isinstance(result["vad_threshold"], float)
        assert result["vad_threshold"] < 0

    def test_already_db_scale_value_untouched(self):
        ss.SETTINGS_DIR.mkdir(parents=True)
        (ss.SETTINGS_DIR / "frank.json").write_text(json.dumps({"vad_threshold": -30.0}))
        result = ss.load_saved_settings("frank")
        assert result["vad_threshold"] == -30.0


class TestPersistSettings:
    def test_persist_writes_to_disk_after_debounce(self):
        settings = {"vosk_model": "en", "recognition_engine": "vosk", "not_persistable": "x"}
        ss.persist_settings("grace", settings)
        # Debounced 1s — wait past it.
        time.sleep(1.3)
        saved = json.loads((ss.SETTINGS_DIR / "grace.json").read_text())
        assert saved["vosk_model"] == "en"
        assert saved["recognition_engine"] == "vosk"
        assert "not_persistable" not in saved  # only PERSISTABLE_KEYS are saved

    def test_rapid_successive_calls_are_debounced_to_last_value(self):
        ss.persist_settings("henry", {"vosk_model": "v1"})
        ss.persist_settings("henry", {"vosk_model": "v2"})
        ss.persist_settings("henry", {"vosk_model": "v3"})
        time.sleep(1.3)
        saved = json.loads((ss.SETTINGS_DIR / "henry.json").read_text())
        assert saved["vosk_model"] == "v3"

    def test_different_slugs_persist_independently(self):
        ss.persist_settings("ivy", {"vosk_model": "ivy-model"})
        ss.persist_settings("jack", {"vosk_model": "jack-model"})
        time.sleep(1.3)
        ivy_saved = json.loads((ss.SETTINGS_DIR / "ivy.json").read_text())
        jack_saved = json.loads((ss.SETTINGS_DIR / "jack.json").read_text())
        assert ivy_saved["vosk_model"] == "ivy-model"
        assert jack_saved["vosk_model"] == "jack-model"


class TestSettingsPath:
    def test_sanitizes_slug_in_filename(self):
        path = ss._settings_path("weird/../slug!!")
        assert ".." not in str(path)
        assert "/" not in path.name

    def test_empty_slug_falls_back_to_default(self):
        path = ss._settings_path("")
        assert path.name == "default.json"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
