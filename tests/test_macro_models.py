"""Controls for target timing and out-of-time model partitions."""

from src.data_ingestion import _generate_macro_fallback, generate_panel
from src.macro_industry_models import prepare_features


def test_next_quarter_target_contains_both_classes():
    macro = _generate_macro_fallback()
    panel = generate_panel(1_000, seed=12, macro_df=macro)
    _, target, train_mask, test_mask, metadata = prepare_features(panel, macro)
    assert set(target[train_mask].unique()) == {0, 1}
    assert set(target[test_mask].unique()) == {0, 1}
    assert metadata.loc[train_mask, "snapshot_date"].max() < metadata.loc[test_mask, "snapshot_date"].min()


def test_feature_row_precedes_default_event():
    macro = _generate_macro_fallback()
    panel = generate_panel(750, seed=13, macro_df=macro)
    _, target, _, _, metadata = prepare_features(panel, macro)
    positive_rows = metadata[target == 1]
    default_dates = panel[panel["is_new_default"] == 1].set_index("obligor_id")["snapshot_date"]
    expected = positive_rows["obligor_id"].map(default_dates)
    assert ((expected - positive_rows["snapshot_date"]).dt.days.between(89, 93)).all()
