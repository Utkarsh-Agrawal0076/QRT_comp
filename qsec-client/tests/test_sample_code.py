from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest as pt

from qsec_client.sample_code import (
    DEFAULT_SFTP_PORT,
    TARGETS_SCHEMA,
    Region,
    prepare_targets_file,
    upload_targets_file,
    validate_targets_file,
)


@pt.fixture
def minimum_targets():
    # Set up a sample valid DataFrame with the correct schema
    return pd.DataFrame(
        {
            "internal_code": ["code1", "code2"],
            "currency": ["USD", "EUR"],
            "target_notional": [100.0, -120.0],
        }
    )


@pt.fixture
def valid_targets():
    # Set up a sample valid DataFrame with the correct schema
    return pd.DataFrame(
        {
            "internal_code": ["code1", "code2"],
            "ric": ["ric1", np.nan],
            "ticker": [np.nan, "ticker2"],
            "target_notional": [100.0, 200.0],
            "currency": ["USD", "EUR"],
            "target_contracts": [10, np.nan],
            "ref_price": [50.0, np.nan],
        }
    )


@pt.fixture
def invalid_targets():
    # Set up a sample invalid DataFrame (wrong data types)
    return pd.DataFrame(
        {
            "internal_code": [np.nan, "code2"],  # internal_code has a None value
            "ric": ["ric1", "ric2"],
            "ticker": ["ticker1", "ticker2"],
            "target_notional": [-400, 200.0],  # Wrong type
            "currency": ["USD", "EUR"],
            "target_contracts": [10, 20.5],
            "ref_price": ["-", "-"],
        }
    )


def test_prepare_targets_fails(minimum_targets, tmp_path):
    with pt.raises(ValueError):
        prepare_targets_file(pd.DataFrame, "GRP01", "AMER", tmp_path)

    with pt.raises(ValueError):
        prepare_targets_file(minimum_targets.drop(columns="currency"), "GRP01", "AMER", tmp_path)

    with pt.raises(ValueError):
        prepare_targets_file(minimum_targets, "", "AMER", tmp_path)

    with pt.raises(ValueError):
        prepare_targets_file(minimum_targets, None, "AMER", tmp_path)

    with pt.raises(ValueError):
        prepare_targets_file(minimum_targets, "GRP01", "", tmp_path)

    with pt.raises(ValueError):
        prepare_targets_file(minimum_targets, "GRP01", None, tmp_path)

    with pt.raises(AttributeError):
        prepare_targets_file(minimum_targets, "GRP01", "AMER", None)

    with pt.raises(ValueError):
        prepare_targets_file(minimum_targets, "GRP01", "AMER", "m:/invalid/path")


def test_prepare_targets_success(minimum_targets, tmp_path):
    target_filepath = prepare_targets_file(minimum_targets, "GRP01", Region.AMER, tmp_path)
    targets = pd.read_csv(target_filepath)
    assert targets.shape == (2, 12)
    assert targets.columns.to_list() == [col.name for col in TARGETS_SCHEMA]
    assert targets.extra_key.iloc[0] == "GRP01_code1"
    assert targets.strategy.iloc[0] == "GRP01_AMER"
    assert targets.advisor_name.iloc[1] == "GRP01"
    assert targets.ref_price.iloc[1] == 0.0
    assert targets.ref_price.iloc[1] == 0


def test_validate_target_file_success(valid_targets, tmp_path):
    target_file = prepare_targets_file(valid_targets, "GRP01", Region.AMER, tmp_path)
    errors = validate_targets_file(target_file)
    assert len(errors) == 0


def test_validate_target_failure(invalid_targets, tmp_path):
    target_file = prepare_targets_file(invalid_targets, "GRP02", Region.EMEA, tmp_path)
    errors = validate_targets_file(target_file)
    assert len(errors) == 4


@patch("qsec_client.sample_code.Transport")
@patch("qsec_client.sample_code.SFTPClient")
@patch("qsec_client.sample_code.RSAKey.from_private_key_file")
def test_upload_target_file_succeeds(
    mock_private_key_file, mock_sftp_client_class, mock_transport_class, valid_targets, tmp_path
):
    # Mocking the SFTP transport and connection
    mock_transport = MagicMock()
    mock_transport_class.return_value = mock_transport
    mock_sftp = MagicMock()
    mock_sftp_client_class.from_transport.return_value = mock_sftp

    # Mock the private key loading
    mock_private_key_file.return_value = MagicMock()

    # create target file
    target_filepath = prepare_targets_file(valid_targets, "GRP01", Region.AMER, tmp_path)

    # Call the upload function
    upload_targets_file(
        targets_csv_path=target_filepath,
        region=Region.AMER,
        sftp_username="user_1234",
        private_key_path=mock_private_key_file,
        sftp_host="sftp.example.com",
    )

    # Check that the transport and SFTP were called
    mock_transport_class.assert_called_once_with(("sftp.example.com", DEFAULT_SFTP_PORT))
    handle = mock_transport_class().__enter__()
    handle.connect.assert_called_once_with(
        username="user_1234", pkey=mock_private_key_file.return_value
    )

    mock_sftp_client_class.from_transport.assert_called_once()

    # Check if the file was "uploaded"
    expected_remote_path = f"incoming/amer/{target_filepath.name}"
    handle = mock_sftp.__enter__()
    handle.put.assert_called_once_with(target_filepath, expected_remote_path, confirm=False)


@patch("qsec_client.sample_code.RSAKey.from_private_key_file")
def test_upload_invalid_target_file(mock_private_key_file, invalid_targets, tmp_path):
    target_filepath = prepare_targets_file(invalid_targets, "GRP01", Region.AMER, tmp_path)
    mock_private_key_file.return_value = MagicMock()

    # Test that upload raises an exception if targets are invalid
    with pt.raises(ExceptionGroup):
        upload_targets_file(
            targets_csv_path=target_filepath,
            region=Region.AMER,
            sftp_host="sftp.example.com",
            sftp_username="test_user",
            private_key_path=mock_private_key_file,
        )


@patch("qsec_client.sample_code.RSAKey.from_private_key_file")
def test_upload_target_file_fails(mock_private_key_file, valid_targets, tmp_path):
    target_filepath = prepare_targets_file(valid_targets, "GRP01", Region.AMER, tmp_path)
    mock_private_key_file.return_value = MagicMock()

    with pt.raises(ValueError):
        upload_targets_file(
            targets_csv_path="m:/wrong/path",
            region=Region.AMER,
            sftp_host="sftp.example.com",
            sftp_username="test_user",
            private_key_path=mock_private_key_file,
        )

    with pt.raises(ValueError):
        upload_targets_file(
            targets_csv_path=None,
            region=Region.AMER,
            sftp_host="sftp.example.com",
            sftp_username="test_user",
            private_key_path=mock_private_key_file,
        )

    with pt.raises(ValueError):
        upload_targets_file(
            targets_csv_path=target_filepath,
            region="APAC",
            sftp_host="sftp.example.com",
            sftp_username="test_user",
            private_key_path=mock_private_key_file,
        )

    with pt.raises(ValueError):
        upload_targets_file(
            targets_csv_path=target_filepath,
            region=None,
            sftp_host="sftp.example.com",
            sftp_username="test_user",
            private_key_path=mock_private_key_file,
        )

    with pt.raises(TypeError):
        upload_targets_file(
            targets_csv_path=target_filepath,
            region="AMER",
            sftp_username="test_user",
            private_key_path=mock_private_key_file,
        )

    with pt.raises(ValueError):
        upload_targets_file(
            targets_csv_path=target_filepath,
            region="AMER",
            sftp_host="sftp.example.com",
            sftp_username="test_user",
            private_key_path="n:/invalid/path",
        )
