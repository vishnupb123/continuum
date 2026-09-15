from dataclasses import dataclass
import hashlib

@dataclass(frozen=True)
class MockContextResult:
    energy: float
    stress: float
    confidence: float
    model_version: str = "mock-context-mode-m0"

class MockContextModel:
    """Deterministic M0 adapter. Replace behind this boundary in M4."""

    def predict(self, text: str) -> MockContextResult:
        normalized = text.strip().lower()
        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
        base_energy = 0.35 + (digest[0] / 255) * 0.4
        base_stress = 0.30 + (digest[1] / 255) * 0.45

        stress_terms = ("stress", "tired", "exhaust", "hard", "overwhelm", "pressure", "bad")
        positive_terms = ("good", "hope", "happy", "better", "great", "calm", "excited")
        stress_hits = sum(term in normalized for term in stress_terms)
        positive_hits = sum(term in normalized for term in positive_terms)

        stress = min(0.98, base_stress + 0.07 * stress_hits - 0.03 * positive_hits)
        energy = max(0.05, min(0.95, base_energy - 0.06 * stress_hits + 0.05 * positive_hits))
        return MockContextResult(
            energy=round(energy, 3),
            stress=round(stress, 3),
            confidence=0.82,
        )
