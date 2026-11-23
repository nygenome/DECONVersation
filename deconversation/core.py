from . import preprocessing,embeddings,deconvolution
import pandas as pd
import scanpy as sc
import warnings
import logging
import os
from deconversation.resource_loader import load_example_csv

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DATASETS_VERBOSITY", "error")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
#os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
import io
from contextlib import redirect_stdout


# ============================================
# Main function for extracting embeddings and deconvolution
# ============================================
def deconverse(
    bulk_df,
    model = None,
    mode = None,
    adata = None,
    sig_df = None,
    temp_output_dir = "temp",
    cell_type_col = "type",
    sample_col = "sample",
    solver = "nnls",
    demo = False,
    transpose = False,
    batch_size = 5,
    bulkformer_bulk_scale = None,
    bulkformer_signature_scale = None,
    bulkformer_n_genes = 4096,
    bulkformer_gene_panel = None,
    scconcept_cache_dir = None,
    scconcept_species = 'hsapiens'):
    """
    Extracting embeddings for bulk and ref signature data, then run NNLS-based and other methods for deconvolution
    
    Parameters
    ----------
    bulk_df : str
        path to bulk expression matrix (Rows:Genes, Columns:Samples)
    
    adata : str
        path to reference single cell adata object, either adata or precomputed sig_df must be specified
    
    cell_type_col : str
        adata object obs column designating cell type

    sample_col : str
        adata object obs column designating sample or batch

    model : str
        path to zero-shot or fine-tuned model

    mode : str
        scFM model used, support: 
        - "geneformer"
        - "c2s"
        - "cellhermes"
        - "scgpt"
        - "bulkformer" (Atlas)
        - "scconcept"
        - "scvi"
        - "pca"
        - "raw"

    sig_df : str
        path to signature matrix 

    temp_output_dir : str
        path for saving temp files

    solver : str
        solvers currently supported: nnls, ridge, elasticnet, nusvr
    
    transpose : bool
        if tables are in sample x gene format, will need to be transposed

    batch_size : int
        default batch processing of 5, lower if memory is a constraint

    bulkformer_bulk_scale, bulkformer_signature_scale : str, optional
        Required for BulkFormer: counts, tpm, log1p_tpm, or log2_tpm.
        The reference scale describes the values in the signature matrix (or
        adata.X before averaging). Inputs use Ensembl gene IDs.
    bulkformer_n_genes : int
        Maximum panel size when selecting reference-variable genes (1–4096).
    bulkformer_gene_panel : str, optional
        CSV containing an 'ensembl_gene' column from a previous Atlas run.

    scconcept_cache_dir : str, optional
        Directory for scConcept checkpoint resources. Defaults to a directory
        under temp_output_dir. For mode='scconcept', model is a checkpoint name
        such as 'corpus40M-model30M', not a local model path.
    scconcept_species : str
        Species name understood by the selected scConcept checkpoint.

    Returns
    -------
    pd.DataFrame
        Estimated cell-type proportions (samples × cell types)
    """
    for noisy in ("transformers", "datasets", "scanpy", "anndata"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    if model is None and mode not in {"raw", "pca"}:
        raise ValueError("model is required unless mode is 'raw' or 'pca'")
    if mode is None:
        mode = embeddings.infer_model(model)
    print("Using model: " + mode)
    os.makedirs(temp_output_dir, exist_ok=True)

    # prep ref data
    # make ref signature matrix
    #print("Prepping signature...")
    if demo:
        print("Running tiny demo example...")
        sig_mat = load_example_csv("demo_sig_mat.csv")
    else:
        if sig_df is not None:
            sig_mat = pd.read_csv(sig_df, index_col=0)
            if transpose:
                sig_mat = sig_mat.T
        else:
            if adata is None:
                raise ValueError("adata and sig_df cannot both be empty")
            adata = sc.read_h5ad(adata)
            sig_mat = preprocessing.create_signature_matrix(adata = adata,
                                                            sample_col = sample_col,
                                                            cell_type_col = cell_type_col,
                                                            groupby = cell_type_col,
                                                            output_path = None)
            sig_mat.to_csv(temp_output_dir + "/signature.csv")
    # convert to ensembl id if geneformer or scvi
    if mode in {"geneformer", "scvi"} and ("ENS" not in sig_mat.index[0]):
        print("Signature rows are not ENSG ids, converting...")
        sig_mat.index = preprocessing.gene_id_name_map(gene_list=sig_mat.index, mode="to_ensembl" )
    sig_mat = sig_mat.loc[sig_mat.index.dropna()].T
    if sig_mat.shape[1] == 0:
        print("No genes left in signature data, please check input.")
    #sig_mat = sig_mat.T

    # load bulk query data
    #print("Prepping bulk...")
    if demo:
        bulk_df = load_example_csv("demo_bulk.csv")
    else:
        bulk_df = pd.read_csv(bulk_df, index_col=0)
        if transpose:
            bulk_df = bulk_df.T
    # convert to ensembl id if geneformer or scvi
    if mode in {"geneformer", "scvi"} and ("ENS" not in bulk_df.index[0]):
        print("Bulk data rows are not ENSG ids, converting...")
        bulk_df.index = preprocessing.gene_id_name_map(gene_list=bulk_df.index, mode="to_ensembl" )
    bulk_df = bulk_df.loc[bulk_df.index.dropna()].T
    if bulk_df.shape[1] == 0:
        print("No genes left in bulk data, please check input.")

    # extract embeddings
    if mode == "bulkformer":
        panel = None
        if bulkformer_gene_panel is not None:
            panel = pd.read_csv(bulkformer_gene_panel)['ensembl_gene'].tolist()
        bulk_embed, sig_mat_embed, panel, bulk_coverage, sig_coverage = (
            embeddings.get_embedding_bulkformer_atlas(
                bulk_df, sig_mat, model, bulkformer_bulk_scale,
                bulkformer_signature_scale, batch_size=batch_size,
                n_genes=bulkformer_n_genes, gene_panel=panel))
        pd.DataFrame({'ensembl_gene': panel}).to_csv(
            os.path.join(temp_output_dir, 'bulkformer_gene_panel.csv'), index=False)
        bulk_coverage.to_csv(os.path.join(temp_output_dir, 'bulkformer_bulk_coverage.csv'))
        sig_coverage.to_csv(os.path.join(temp_output_dir, 'bulkformer_signature_coverage.csv'))
        bulk_embed.to_csv(os.path.join(temp_output_dir, 'bulk_embedding.csv'))
        sig_mat_embed.to_csv(os.path.join(temp_output_dir, 'signature_embedding.csv'))
    elif mode == "scconcept":
        cache_dir = scconcept_cache_dir or os.path.join(temp_output_dir, 'scconcept_cache')
        bulk_embed, sig_mat_embed = embeddings.get_scconcept_embeddings(
            bulk_df, sig_mat, model, cache_dir, batch_size=batch_size,
            species=scconcept_species)
        bulk_embed.to_csv(os.path.join(temp_output_dir, 'bulk_embedding.csv'))
        sig_mat_embed.to_csv(os.path.join(temp_output_dir, 'signature_embedding.csv'))
    elif mode == "pca":
        res = embeddings.get_embedding_pca(bulk_df, sig_mat)
        bulk_embed = res["pca_bulk"]
        sig_mat_embed = res["sig_pca"]
    else:
        print("Extracting signature embedding...")
        with redirect_stdout(io.StringIO()):
            if mode == "raw":
                sig_mat_embed = sig_mat
            else:
                sig_mat_embed = embeddings.extract_embs(
                    bulk_df = sig_mat,
                    mode = mode,
                    model_path= model,
                    temp_output_dir = temp_output_dir + "/sig",
                    delete_temp_files = False,
                    batch_size=batch_size
                )
                sig_mat_embed.to_csv(temp_output_dir + "/signature_embedding.csv")
        print("Extracting bulk embedding...")
        with redirect_stdout(io.StringIO()):
            if mode == "raw":
                bulk_embed = bulk_df
            else:
                bulk_embed = embeddings.extract_embs(
                    bulk_df = bulk_df,
                    mode = mode,
                    model_path= model,
                    temp_output_dir = temp_output_dir + "/bulk",
                    delete_temp_files = False,
                    batch_size=batch_size
                )
            bulk_embed.to_csv(temp_output_dir + "/bulk_embedding.csv")
    
    # solve
    print("Solving deconvolution...")
    cell_prop_pred = deconvolution.run_deconv(bulk_df = bulk_embed.T,
                                signature_df = sig_mat_embed.T, 
                                solver= solver)
    # remove index and column names
    cell_prop_pred.index.name = None
    cell_prop_pred.columns.name = None

    return cell_prop_pred
