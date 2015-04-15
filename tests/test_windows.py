import pytest
import mdct.windows


def test_kbd():
    mdct.windows.kaiser_derived(50)
