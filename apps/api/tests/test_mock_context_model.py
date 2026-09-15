from app.services.mock_context_model import MockContextModel

def test_mock_model_is_deterministic():
    model = MockContextModel()
    first = model.predict("Work was exhausting today")
    second = model.predict("Work was exhausting today")
    assert first == second
    assert 0 <= first.energy <= 1
    assert 0 <= first.stress <= 1
