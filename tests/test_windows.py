import pytest
import mdct.windows


def test_kbd():
    mdct.windows.kaiser_derived(50)

    with pytest.raises(ValueError):
        mdct.windows.kaiser_derived(51)
