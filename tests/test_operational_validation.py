"""Safety boundaries for the opt-in real-data validation harness (no network)."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('operational_validation',Path(__file__).resolve().parents[1]/'scripts/run_operational_validation.py')
validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)


def test_validation_rejects_production_location(tmp_path,monkeypatch):
    monkeypatch.setattr(validation,'ROOT',tmp_path)
    (tmp_path/'outputs').mkdir()
    (tmp_path/'outputs/operational_validation_location.json').write_text(json.dumps({'root':str(tmp_path)}))
    with pytest.raises(ValueError,match='isolated'):
        validation.location()


@pytest.mark.parametrize('script',['08_publish_to_google_sheet','../production'])
def test_validation_cannot_run_publisher_or_arbitrary_script(tmp_path,script):
    with pytest.raises(ValueError,match='non-publishing'):
        validation.entrypoint(tmp_path,script)
