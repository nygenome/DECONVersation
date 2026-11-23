import sys
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("scanpy")

from deconversation import embeddings
from deconversation.core import deconverse


@pytest.fixture
def atlas_api(monkeypatch):
    calls = []
    vocab = pd.DataFrame({'ensembl_gene': ['ENSG1', 'ENSG2', 'ENSG3']})

    def prepare(frame, vocab, normalization_vocab, *, input_scale):
        calls.append(('prepare', input_scale))
        aligned = frame.reindex(columns=vocab.ensembl_gene).fillna(-10).to_numpy(dtype=float)
        return aligned, pd.DataFrame({'observed_genes': (aligned >= 0).sum(axis=1)}, index=frame.index)

    def select(values, n_genes):
        available = np.flatnonzero((values >= 0).all(axis=0))
        variance = np.var(values[:, available], axis=0)
        selected = available[np.argsort(-variance, kind='stable')[:n_genes]]
        return np.sort(selected)

    def extract(model, values, panel, **kwargs):
        calls.append(('extract', list(panel), kwargs))
        return values[:, panel]

    fake = types.ModuleType('bulkformer')
    fake.load_atlas = lambda path, device: (object(), vocab, vocab)
    fake.prepare_atlas_input = prepare
    fake.select_variable_genes = select
    fake.extract_atlas = extract
    monkeypatch.setitem(sys.modules, 'bulkformer', fake)
    return calls


def test_atlas_uses_same_reference_selected_panel_for_both_inputs(atlas_api):
    bulk = pd.DataFrame([[8, 9]], index=['sample'], columns=['ENSG1', 'ENSG2'])
    sig = pd.DataFrame([[1, 4, 3], [3, 4, 8]], index=['A', 'B'],
                       columns=['ENSG1', 'ENSG2', 'ENSG3'])
    bulk_emb, sig_emb, panel, bulk_cov, sig_cov = embeddings.get_embedding_bulkformer_atlas(
        bulk, sig, 'atlas', 'tpm', 'counts', n_genes=3)
    assert panel == ['ENSG1']  # ENSG3 varies but is absent from the bulk input.
    assert list(bulk_emb.index) == ['sample']
    assert list(sig_emb.index) == ['A', 'B']
    assert bulk_emb.columns.equals(sig_emb.columns)
    assert list(bulk_cov.index) == ['sample'] and list(sig_cov.index) == ['A', 'B']
    assert [call[1] for call in atlas_api if call[0] == 'prepare'] == ['tpm', 'counts']
    extraction_calls = [call for call in atlas_api if call[0] == 'extract']
    assert [call[1] for call in extraction_calls] == [[0], [0]]
    assert all(call[2]['precision'] == 'fp32' for call in extraction_calls)


def test_atlas_rejects_panel_missing_from_bulk(atlas_api):
    bulk = pd.DataFrame([[1]], columns=['ENSG1'])
    sig = pd.DataFrame([[1, 2]], columns=['ENSG1', 'ENSG2'])
    with pytest.raises(ValueError, match='observed in all'):
        embeddings.get_embedding_bulkformer_atlas(
            bulk, sig, 'atlas', 'counts', 'counts', gene_panel=['ENSG2'])


def test_deconverse_bulkformer_writes_embeddings_and_solves(tmp_path, monkeypatch):
    bulk = pd.DataFrame({'sample': [3., 1.]}, index=['ENSG1', 'ENSG2'])
    sig = pd.DataFrame({'A': [1., 0.], 'B': [0., 1.]}, index=bulk.index)
    bulk.to_csv(tmp_path / 'bulk.csv')
    sig.to_csv(tmp_path / 'sig.csv')

    def fake_extract(bulk_df, signature_df, model, bulk_scale, sig_scale, **kwargs):
        assert (bulk_scale, sig_scale) == ('counts', 'tpm')
        columns = ['BulkFormer_0', 'BulkFormer_1']
        return (pd.DataFrame([[3., 1.]], index=bulk_df.index, columns=columns),
                pd.DataFrame([[1., 0.], [0., 1.]], index=signature_df.index, columns=columns),
                ['ENSG1', 'ENSG2'], pd.DataFrame(index=bulk_df.index),
                pd.DataFrame(index=signature_df.index))

    monkeypatch.setattr(embeddings, 'get_embedding_bulkformer_atlas', fake_extract)
    result = deconverse(str(tmp_path / 'bulk.csv'), model='atlas', mode='bulkformer',
                        sig_df=str(tmp_path / 'sig.csv'), temp_output_dir=str(tmp_path / 'out'),
                        bulkformer_bulk_scale='counts', bulkformer_signature_scale='tpm')
    np.testing.assert_allclose(result.loc['sample', ['A', 'B']], [0.75, 0.25])
    assert pd.read_csv(tmp_path / 'out' / 'bulkformer_gene_panel.csv').ensembl_gene.tolist() == ['ENSG1', 'ENSG2']
    assert (tmp_path / 'out' / 'bulk_embedding.csv').is_file()
    assert (tmp_path / 'out' / 'signature_embedding.csv').is_file()
