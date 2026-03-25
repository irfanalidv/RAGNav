from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_integration_suite_placeholder():
    pytest.skip("No integration tests yet; add API/network cases here.")
