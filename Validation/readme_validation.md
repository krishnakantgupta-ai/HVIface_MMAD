## Structural Confidence and Validation of Predicted Complexes

Structures are colored by pLDDT using a red–blue gradient (50–90).

To evaluate the structural reliability of the modelled host–virus complexes, we assessed AlphaFold-predicted structures using per-residue confidence scores (pLDDT). The majority of residues involved in the predicted interaction interfaces correspond to moderate-to-high confidence regions, as indicated by pLDDT-based color mapping (ranging from yellow to blue).

For visualization, pLDDT values were mapped onto the structures using a continuous color spectrum:

- **Color scheme:** red–yellow–green–cyan–blue  
- **Range:** minimum = 50, maximum = 90
  
**PyMOL command:**
```python
spectrum b, red_yellow_green_cyan_blue, minimum=50, maximum=90
