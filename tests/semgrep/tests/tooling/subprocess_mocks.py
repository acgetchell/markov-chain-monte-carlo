import subprocess
import unittest.mock
from unittest import mock
from unittest.mock import MagicMock, Mock


def adhoc_mock_stdout() -> None:
    # ruleid: mcmc.python.no-adhoc-completedprocess-mock
    result = Mock()
    result.stdout = "ok"


def adhoc_mock_returncode() -> None:
    # ruleid: mcmc.python.no-adhoc-completedprocess-mock
    result = MagicMock()
    result.returncode = 0


def adhoc_mock_stdout_constructor() -> None:
    # ruleid: mcmc.python.no-adhoc-completedprocess-mock
    Mock(stdout="ok")


def adhoc_unittest_mock_returncode_constructor() -> None:
    # ruleid: mcmc.python.no-adhoc-completedprocess-mock
    unittest.mock.Mock(returncode=0)


def adhoc_mock_magic_stdout_constructor() -> None:
    # ruleid: mcmc.python.no-adhoc-completedprocess-mock
    mock.MagicMock(stdout="ok")


def typed_completed_process() -> subprocess.CompletedProcess[str]:
    # ok: mcmc.python.no-adhoc-completedprocess-mock
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")
