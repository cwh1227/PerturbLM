"""Static configuration for BMG subgraph extraction."""

from pathlib import Path

BMG_ROOT = Path("e:/BMG/BioMedGraphica-Conn")
CACHE_DIR = Path("e:/BMG/.cache")
CACHE_DIR.mkdir(exist_ok=True)

RELATION_CSV = BMG_ROOT / "Relation" / "BioMedGraphica_Conn_Relation.csv"

DISPLAY_NAME_FILES = {
    "Gene": BMG_ROOT / "Entity/Gene/BioMedGraphica_Conn_Gene_Display_Name.csv",
    "Drug": BMG_ROOT / "Entity/Drug/BioMedGraphica_Conn_Drug_Display_Name.csv",
    "Protein": BMG_ROOT / "Entity/Protein/BioMedGraphica_Conn_Protein_Display_Name.csv",
    "Transcript": BMG_ROOT / "Entity/Transcript/BioMedGraphica_Conn_Transcript_Display_Name.csv",
    "Disease": BMG_ROOT / "Entity/Disease/BioMedGraphica_Conn_Disease_Display_Name.csv",
    "Pathway": BMG_ROOT / "Entity/Pathway/BioMedGraphica_Conn_Pathway_Display_Name.csv",
    "Phenotype": BMG_ROOT / "Entity/Phenotype/BioMedGraphica_Conn_Phenotype_Display_Name.csv",
    "Metabolite": BMG_ROOT / "Entity/Metabolite/BioMedGraphica_Conn_Metabolite_Display_Name.csv",
    "Exposure": BMG_ROOT / "Entity/Exposure/BioMedGraphica_Conn_Exposure_Display_Name.csv",
    "Microbiota": BMG_ROOT / "Entity/Microbiota/BioMedGraphica_Conn_Microbiota_Display_Name.csv",
    "Promoter": BMG_ROOT / "Entity/Promoter/BioMedGraphica_Conn_Promoter_Display_Name.csv",
}

DESCRIPTION_FILES = {
    "Gene":       BMG_ROOT / "Entity/Gene/BioMedGraphica_Conn_Gene_Description_Combined.csv",
    "Drug":       BMG_ROOT / "Entity/Drug/BioMedGraphica_Conn_Drug_Description_Combined.csv",
    "Protein":    BMG_ROOT / "Entity/Protein/BioMedGraphica_Conn_Protein_Description_Combined.csv",
    "Transcript": BMG_ROOT / "Entity/Transcript/BioMedGraphica_Conn_Transcript_Description_Combined.csv",
    "Disease":    BMG_ROOT / "Entity/Disease/BioMedGraphica_Conn_Disease_Description_Combined.csv",
    "Pathway":    BMG_ROOT / "Entity/Pathway/BioMedGraphica_Conn_Pathway_Description_Combined.csv",
    "Phenotype":  BMG_ROOT / "Entity/Phenotype/BioMedGraphica_Conn_Phenotype_Description_Combined.csv",
    "Metabolite": BMG_ROOT / "Entity/Metabolite/BioMedGraphica_Conn_Metabolite_Description_Combined.csv",
    "Exposure":   BMG_ROOT / "Entity/Exposure/BioMedGraphica_Conn_Exposure_Description_Combined.csv",
}

NAME_COL = {
    "Gene": "BMG_Gene_Name",
    "Drug": "BMG_Drug_Name",
    "Protein": "BMG_Protein_Name",
    "Transcript": "BMG_Transcript_Name",
    "Disease": "BMG_Disease_Name",
    "Pathway": "BMG_Pathway_Name",
    "Phenotype": "BMG_Phenotype_Name",
    "Metabolite": "BMG_Metabolite_Name",
    "Exposure": "BMG_Exposure_Name",
    "Microbiota": "BMG_Microbiota_Name",
    "Promoter": "BMG_Promoter_Name",
}

DEFAULT_EDGE_TYPES = [
    "Promoter-Gene",
    "Gene-Transcript",
    "Transcript-Protein",
    "Protein-Protein",
    "Protein-Pathway",
    "Pathway-Protein",
    "Drug-Protein",
    "Drug-Disease",
    "Drug-Pathway",
    "Protein-Disease",
    "Protein-Phenotype",
    "Disease-Phenotype",
    "Phenotype-Disease",
    "Gene-Transcript",
    "Exposure-Gene",
    "Metabolite-Protein",
    "Metabolite-Disease",
]

# Edge types that are stored as A→B only but should be traversable in both
# directions. Reverse edges are synthesised as B→A with type "B-A".
# Protein-Protein is intentionally excluded — it is already near-fully
# bidirectional in the source data, so adding its reverse would ~double
# the 16 M edges for no gain.
BIDIRECTIONAL_EDGE_TYPES: set[str] = {
    # Drug edges — primary use case: reach Drug nodes from Protein/Disease/Pathway
    "Drug-Protein",
    "Drug-Disease",
    "Drug-Pathway",
    "Drug-Metabolite",
    # Metabolite edges — reach Metabolite nodes from Protein/Disease
    "Metabolite-Protein",
    "Metabolite-Disease",
    # Excluded intentionally to prevent BFS fan-out explosions:
    #   Exposure-Gene / Exposure-Disease: from Exposure → many Genes at next hop
    #   Microbiota-Disease / Microbiota-Drug: low information density for gene/drug subgraphs
}

TYPE_PRIORITY = [
    "Gene",
    "Drug",
    "Protein",
    "Transcript",
    "Disease",
    "Pathway",
    "Phenotype",
    "Metabolite",
    "Exposure",
]
