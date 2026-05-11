import os
import subprocess

def generate_msa(query_sequence, msa_output_path=None, database="uniref90", mmseqs_bin="mmseqs", tmp_dir="tmp_mmseqs"):
    """
    Given a protein sequence, generate an MSA with sequences at most 35% identical.
    Uses local MMseqs2 installation for search.

    Args:
        query_sequence (str): The input protein sequence (1-letter code).
        msa_output_path (str or None): If provided, writes the MSA to this file in FASTA format.
        database (str): Path to MMseqs2 database to search against (default: "uniref90").
        mmseqs_bin (str): Path to mmseqs binary (default: "mmseqs").
        tmp_dir (str): Temporary directory for MMseqs2.

    Returns:
        msa_seqs (list of str): List of sequences in the MSA (including the query).
    """
    import tempfile
    import shutil

    # Prepare temporary files
    os.makedirs(tmp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=tmp_dir, suffix=".fa") as query_fasta:
        query_fasta.write(f">query\n{query_sequence}\n")
        query_fasta_path = query_fasta.name

    result_m8 = os.path.join(tmp_dir, "result.m8")
    result_msa = os.path.join(tmp_dir, "result.a3m")

    # Run MMseqs2 search
    # You must have a local MMseqs2 database (e.g., uniref90) available.
    # Example: mmseqs createdb uniref90.fasta uniref90_db
    try:
        # Search
        subprocess.run([
            mmseqs_bin, "search", query_fasta_path, database, os.path.join(tmp_dir, "result"), tmp_dir,
            "--max-seqs", "1000", "-a", "--min-seq-id", "0.35"
        ], check=True)
        # Convert alignment to a3m
        subprocess.run([
            mmseqs_bin, "convertalis", query_fasta_path, database, os.path.join(tmp_dir, "result"),
            result_msa, "--format-output", "query,target,alnseqT"
        ], check=True)
    except Exception as e:
        print("MMseqs2 search failed or MMseqs2 is not installed. Returning only the query sequence as the MSA.")
        if msa_output_path is not None:
            with open(msa_output_path, "w") as f:
                f.write(f">query\n{query_sequence}\n")
        return [query_sequence]
    finally:
        # Clean up query fasta
        if os.path.exists(query_fasta_path):
            os.remove(query_fasta_path)

    # Parse A3M or FASTA output
    msa_seqs = []
    if os.path.exists(result_msa):
        seq = ""
        with open(result_msa) as f:
            for line in f:
                if line.startswith(">"):
                    if seq:
                        msa_seqs.append(seq)
                        seq = ""
                else:
                    seq += line.strip()
            if seq:
                msa_seqs.append(seq)
        if msa_output_path is not None:
            shutil.copy(result_msa, msa_output_path)
        os.remove(result_msa)
    else:
        # Fallback: just the query
        msa_seqs = [query_sequence]
        if msa_output_path is not None:
            with open(msa_output_path, "w") as f:
                f.write(f">query\n{query_sequence}\n")

    # Clean up tmp_dir if empty
    try:
        if not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)
    except Exception:
        pass

    return msa_seqs

ref_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"

generate_msa(ref_seq)