from heart_disease_mlop.data_pipeline import FEATURE_COLUMNS, build_preprocessor, load_dataset


def test_dataset_loaded_and_clean():
    df = load_dataset()

    assert not df.empty
    assert "target" in df.columns
    assert set(df["target"].unique()).issubset({0, 1})
    assert all(column in df.columns for column in FEATURE_COLUMNS)


def test_preprocessor_builds_valid_matrix():
    df = load_dataset()
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df[FEATURE_COLUMNS])

    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] > 0
