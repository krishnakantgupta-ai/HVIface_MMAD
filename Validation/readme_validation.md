## Structural Confidence and Validation of Predicted Complexes

Structures are colored by pLDDT using a red–blue gradient (50–90).

To evaluate the structural reliability of the modelled host–virus complexes, we assessed AlphaFold-predicted structures using per-residue confidence scores (pLDDT). The majority of residues involved in the predicted interaction interfaces correspond to moderate-to-high confidence regions, as indicated by pLDDT-based color mapping (ranging from yellow to blue).

For visualization, pLDDT values were mapped onto the structures using a continuous color spectrum:

- **Color scheme:** red–yellow–green–cyan–blue  
- **Range:** minimum = 50, maximum = 90
  
**PyMOL command:**
```python
spectrum b, red_yellow_green_cyan_blue, minimum=50, maximum=90

## Convert Atom-wise pLDDT to Residue-wise pLDDT

The following Python script extracts **per-residue pLDDT scores** from an AlphaFold-generated PDB file by averaging atom-level pLDDT values.

```python
# Atom pLDDT to Residue pLDDT

from collections import defaultdict

# Dictionary to store pLDDT values for each residue
residue_plddt = defaultdict(list)

# Read PDB file
with open(r"C:\Users\SASTRA\Desktop\cornea-master\Project\Modeled structures\Integrase.pdb", "r") as f:
    for line in f:
        if line.startswith("ATOM"):
            res_id = int(line[22:26].strip())      # Residue number
            bfactor = float(line[60:66].strip())   # pLDDT score (stored in B-factor column)
            residue_plddt[res_id].append(bfactor)

# Compute average pLDDT per residue
residue_avg_plddt = {
    res: sum(vals) / len(vals) for res, vals in residue_plddt.items()
}

# Sort residues and print output
sorted_res = sorted(residue_avg_plddt)

print("S.No\tResidue\tpLDDT")
for i, res in enumerate(sorted_res, start=1):
    print(f"{i}\t{res}\t{residue_avg_plddt[res]:.2f}")
