"""Design of models of transcription predictor by perturbations.

The model consists of perturbation encoder, genetic cell encoder, and perturbation decoder. The input of perturbation
encoder is sum of gene embeddings of perturbed genes with dimension size specified by #LLM. The input of genetic cell
encoder is the gene embeddings of all genes left unfiltered by preprocessing. The output of perturbation decoder is the
genetic cell embeddings output by model with same size of genetic cell embedding input.

Usage example:
"""

