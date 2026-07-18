#import "@preview/elsearticle:3.1.0": *
// #import "../src/elsearticle.typ": *

#set math.equation(numbering: "(1)")
#let abstract
#let numbered_eq(content) = math.equation(
    block: true,
    numbering: "(1)",
    content,
)

#show: elsearticle.with(
  title: "Active and reactive cell shape change in silico",
  authors: (
    (name: [S. Utecht], affiliations: ("a", "b"), corresponding: true, email:"sebastian.utecht@ku.nbi.dk"),
    (name: [J. Schauser], affiliations: ("a",)),
  ),
  affiliations: (
    "a": [University of Copenhagen (UCPH), Copenhagen, Denmark],
    "b": [University of Copenhagen (UCPH), Copenhagen, Denmark],
  ),
  journal: "Guess we'll see",
  abstract: abstract,
  keywords: ("Agent base modelling", "In silico", "Cell shapes", ),
  // date: datetime(year: 2024, month: 7, day: 12),
  paper: "a5",
  format: "preprint",
  // numcol: 1,
  // line-numbering: true,
)

#set math.equation(numbering: "(1)")
= Introduction <intro>
- Morphogenesis & tissue adaptability
- Active and adaptive cells
- The polar-adhesion model
- Other modelling approaches - Where is the field now?


== The polar-adhesion model <intro_polar_adhesion>
We here present a slightly altered version of the agent-based polarity adhesion model as first presented in @nissen_theoretical_2018, which couples cell-cell adhesion to two types of cell polarity: apical-basal polarity (ABP) and planar cell polarity (PCP) through the pairwise potential 

#numbered_eq($V_(i j) = A_(i j) (exp(-||bold(r)_(i j)||)-exp(-(||bold(r)_(i j)||) / beta))$)<eq_pw_potential>

where

#numbered_eq($A_(i j) = lambda^0_(i j) + lambda^1_(i j) S^1_(i j) + lambda^2_(i j) S^2_(i j) + lambda^3_(i j) S^3_(i j) $) <eq_pw_potential_A>

and $S$-factors given by

#numbered_eq(
  $S^1_(i j) &= ((hat(bold(r))_(i j) dot bold(p)_i) times (hat(bold(r))_(i j) dot bold(p)_j)-1)/2 \
  S^2_(i j) &= ((bold(q)_i dot bold(p)_i) times (bold(q)_j dot bold(p)_j) - 1)/2 \
  S^3_(i j) &= ((hat(bold(r))_(i j) dot bold(q)_i) times (hat(bold(r))_(i j) dot bold(q)_j)-1)/2  $) <eq_pw_s_factors>

where $(hat(bold(r))_(i j)$ is the unit displacement vector between cells $i$, $bold(p)$'s are unit vector representations of ABP and $bold(q)$'s are unit vector representations of PCP. 

Put succinctly, $S_1$ enforces alignment between the ABPs of neighboring cells and adhesion between cells perpendicular to local ABP orientation, creating stable cell-sheet structures. $S_2$ enforces alignment between the PCPs of neighboring cells and perpendicularity to local ABP orientation, creating a locally aligned PCP-orientation in the cell-sheet plane. Finally, $S_3$ enforces alignment between the PCPs of neighboring cells and adhesion between cells perpendicular to local PCP orientation, creating convergent extension in the cell-sheet plane. The $lambda$'s are scalar valued parameters controlling the strengths of the respective $S$-factors.

@nielsen_model_2020 builds on the model by introducing cell-wedging through the addition of a new scalar-valued parameter $alpha$ which is used to peturb the $bold(p)$-vectors (ABP) in pairwise interactions between cells in the following manner

For isotropic wedging
#numbered_eq($tilde(bold(p))_(i, i j) = (bold(p)_(i) + alpha bold(hat(r))_(i j)) /(||bold(p)_(i) + alpha bold(hat(r))_(i j)||) $) <eq_iso_wedging>

For anisotropic wedging
#numbered_eq($tilde(bold(p))_(i, i j) =  (bold(p)_(i) + alpha_(i j) (bold(hat(r))_(i j) dot chevron.l bold(q)_(i j) chevron.r)chevron.l bold(q)_(i j) chevron.r) / (||bold(p)_(i) + alpha_(i j) (bold(hat(r))_(i j) dot chevron.l bold(q)_(i j) chevron.r)chevron.l bold(q)_(i j) chevron.r||)  $) <aniso_wedging>

Where the subscript of $bold(tilde(p))_(i, i j)$ denotes the $bold(tilde(p))_i$ used in the calculation of $V_(i j)$, which now deviates from the $bold(tilde(p))_i$ used in the calculation of $V_(i k)$, and $chevron.l bold(q)_(i j) chevron.r= (bold(q)_i + bold(q)_j) / (||bold(q)_i + bold(q)_j||)$ is the normalized average of $bold(q)_i$ and $bold(q)_j$.

= Active cell-shape change <active>
== Cell sheet curvature

In order to encode stable and versatile curvature into the simulated cell sheet we expand upon the existing wedging framework of @nielsen_model_2020 by replacing the two different wedging modes (eq. @eq_iso_wedging and @aniso_wedging) and the parameter $alpha_(i j)$ with a unified wedging framework controlled by two parameters $theta^parallel_i$ and $theta^perp_i$ used in cell-cell interactions to calculate $theta^parallel_(i j) = (theta^parallel_i + theta^parallel_j)/2$ and $theta^perp_(i j)=(theta^perp_i + theta^perp_j)/2$ which denote the equilibrium angle between $bold(p)_i$ and $bold(p)_j$ parallel and perpendicular to $chevron.l bold(q)_(i j) chevron.r$ respectively.

For each cell-cell interaction we define a rotation $bold(R)_(i,i j)(theta^parallel_(i j), theta^perp_(i j))$ such that

#numbered_eq($bold(R)_(i, i j)chevron.l bold(p)_(i j) chevron.r &= tilde(chevron.l bold(p)_(i, i j) chevron.r) \
bold(R)_(j, i j)chevron.l bold(p)_(i j) chevron.r &= bold(R)_(i, i j)^T chevron.l bold(p)_(i j) chevron.r =  tilde(chevron.l bold(p)_(j, i j) chevron.r)$) <eq_def_rotate>

where, analogous to eq. @aniso_wedging, 

#numbered_eq($tilde(chevron.l bold(p)_(i, i j) chevron.r) =  (chevron.l bold(p)_(i j) chevron.r + alpha^parallel_(i j) (bold(hat(r))_(i j) dot chevron.l bold(q)_(i j) chevron.r)chevron.l bold(q)_(i j) chevron.r + alpha^perp_(i j) (bold(hat(r))_(i j) dot chevron.l bold(w)_(i j) chevron.r)chevron.l bold(w)_(i j) chevron.r) / (||chevron.l bold(p)_(i j) chevron.r + alpha^parallel_(i j) (bold(hat(r))_(i j) dot chevron.l bold(q)_(i j) chevron.r)chevron.l bold(q)_(i j) chevron.r + alpha^perp_(i j) (bold(hat(r))_(i j) dot chevron.l bold(w)_(i j) chevron.r)chevron.l bold(w)_(i j) chevron.r||)  $) <eq_def_arrivevec>

where $alpha_(i j)^(parallel, perp) = tan(theta^(parallel,perp)/2)$ (the factor $1/2$ comes from only half of the desired equibrium angle is given by $tilde(bold(p))_(i, i j)$, the other half given by $tilde(bold(p))_(j, i j)$), $chevron.l bold(p)_(i j) chevron.r= (bold(p)_i + bold(p)_j) / (||bold(p)_i + bold(p)_j||)$ , and $bold(w)_(i j) = chevron.l bold(q)_(i j) chevron.r times chevron.l bold(p)_(i j) chevron.r $ being the in-plane unit vector perpendicular to the local PCP orientation. These rotations are applied to both the ABP and the PCP of cells $i$ and $j$ in the calculation of $V_(i j)$ (eq. @eq_pw_potential through @eq_pw_s_factors) such that $tilde(bold(q))_(i, i j)$ and $tilde(bold(p))_(i, i j)$ enter instead of $bold(p)_i$ and $bold(q)_j$ and eq. @eq_pw_s_factors. becomes  


#numbered_eq(
  $S^1_(i j) &=(|tilde(bold(p))_(i,i j) dot tilde(bold(p))_(j,i j)|(hat(bold(r))_(i j) dot tilde(bold(p))_(i,i j)) times (hat(bold(r))_(i j) dot tilde(bold(p))_(j,i j))-1)/2 \
  S^2_(i j) &= (|tilde(bold(q))_(i,i j) dot tilde(bold(q))_(j,i j)|(tilde(bold(q))_(i, i j) dot tilde(bold(p))_(i,i j)) times (tilde(bold(q))_(j, i j) dot tilde(bold(p))_(j,i j))-1)/2 \
  S^3_(i j) &= ((hat(bold(r))_(i j) dot tilde(bold(q))_(i, i j)) times (hat(bold(r))_(i j) dot tilde(bold(q))_(j, i j))-1)/2  $) <eq_pw_s_factors_wedging>

where

#numbered_eq($tilde(bold(p))_(i, i j) &= bold(R)_(i, i j)bold(p)_(i) #h(1cm)
tilde(bold(q))_(i, i j) = bold(R)_(i, i j)bold(q)_(i) \
tilde(bold(p))_(j, i j) &= bold(R)_(i, i j)^T bold(p)_(j) #h(1cm)  
tilde(bold(q))_(j, i j) = bold(R)_(i, i j)^T bold(q)_(j) $) <eq_use_rotation>

and the factors $|tilde(bold(p))_(i,i j) dot tilde(bold(q))_(j,i j)|$ and $|tilde(bold(p))_(i,i j) dot tilde(bold(q))_(j,i j)|$ fix an observed assymmetry between wedging parallel or perpendicular to $chevron.l bold(q)_(i j) chevron.r$ and does not qualitatively change the behavior of the system (see appendix [MAKE SECTION]).

This unified framework allows us to control anisotropic wedging both perpendicular and parallel to the PCP orientation while easily recuperating isotropic wedging by setting $theta^parallel_(i j) = theta^perp_(i j)$. Applying the rotations to $bold(q)_i$ and $bold(q)_j$ in addition to $bold(p)_i$ and $bold(p)_j$ helps alleviate energetic frustrations caused by misalignment of $bold(q)_i$ and $bold(q)_j$ in the original formulation of @nielsen_model_2020.

At equilibrium, ${chevron.l bold(p)_(i j) chevron.r,chevron.l bold(q)_(i j) chevron.r,chevron.l bold(w)_(i j) chevron.r}$ forms an approximately orthonormal set and for a single two cell interaction the effective wedging angle $theta^e_(i j)$ is therefore approximately given by

#numbered_eq($theta^e_(i j) approx 2arctan(sqrt(tan^2(theta^parallel/2)cos^2(phi_(chevron.l bold(q)_(i j) chevron.r)) + tan^2(theta^perp/2)sin^2(phi_(chevron.l bold(q)_(i j) chevron.r))))$) <eq_theta_eff>

which gives us the local curvature 

#numbered_eq($kappa^e_(i j) approx theta^e_(i j)/r_0 $) <eq_kappa_eff>

where $phi_(chevron.l bold(q)_(i j) chevron.r)$ is the angle between $chevron.l bold(q)_(i j) chevron.r$ and $bold(r)_(i j)$ and $r_0=ln(beta^(-1))/(1/beta - 1)$ is the equilbrium distance between cells. In the special cases where $bold(r)_(i j)$ lies perfectly parallel to $chevron.l bold(q)_(i j) chevron.r$ or $chevron.l bold(w)_(i j) chevron.r$ or when $theta^parallel_(i j) = theta^perp_(i j)$ eq. @eq_theta_eff reduces to the expected cases $theta_(i j)^e = theta_(i j)^parallel$, $theta_(i j)^e = theta_(i j)^perp$ and $theta_(i j)^e = theta_(i j)^parallel = theta_(i j)^perp$ respectively. [SHOULD WE COMPARE THIS WITH ACTUAL RESULTS? WOULD PROBABLY BE COOL TO HAVE DONE]

Figure @fig_fig1 displays different equilibrium structures resulting from various $theta^parallel$ and $theta^perp$ configurations. For all cells in a 10x10 cell sheet (A) the $theta$-values are subsequently set to the following values and simulated to equilibrium: $theta^parallel_i = theta^perp_i = 30 degree$ (B), $theta^parallel_i = 30 degree$ and $theta^perp_i = 0 degree$ (C) and $theta^parallel_i = 30 degree$ $theta^perp_i = -30 degree$ (D). [The figure also [SHOULD] displays the comparison between the measured local curvature and the theoretical values of eq. @eq_theta_eff]...     


#figure(
  image("FIgures/fig1.png"),
  caption: [Equilibrium structures resulting from various $theta^parallel$ and $theta^perp$ configurations. For all cells in a 10x10 cell sheet (A) the $theta$-values are subsequently set to the following values and simulated to equilibrium: $theta^parallel_i = theta^perp_i = 30 degree$ (B), $theta^parallel_i = 30 degree$ and $theta^perp_i = 0 degree$ (C) and $theta^parallel_i = 30 degree$ $theta^perp_i = -30 degree$ (D) [FIGURE NEEDS TO BE REMADE AND CAPTION EXPANDED UPON]. 
  ],
) <fig_fig1>


== Cell elongation
[FLUFF TEXT]

In order to model cell elongation and contraction we add a new parameter $gamma_i$ to the existing model framework. This gamma is used to control cell extent parallel and perpendicular to $chevron.l bold(q)_(i j) chevron.r$ by modifying the values of $||bold(r)_(i j)||$ used in eq. @eq_pw_potential and neighbor calculation as

#numbered_eq($tilde(||bold(r)_(i j)||) = gamma_(i j)^(cos(2phi_(chevron.l bold(q)_(i j) chevron.r)))||bold(r)_(i j)||$) <eq_gamma_use>

Where $gamma_(i j) = (gamma_i + gamma_j)/2$ and $phi_(chevron.l bold(q)_(i j) chevron.r)$ is the angle between $bold(r)_(i j)$ and $chevron.l bold(q)_(i j) chevron.r$. Utilization of eq. @eq_gamma_use will consequently alter the equilbrium distances between cells as


#numbered_eq($tilde(r)_(0, i j) = gamma_(i j)^(-cos(2phi_(chevron.l bold(q)_(i j) chevron.r)))r_0 $)


For $gamma_(i j) > 1$ cell pairs $i j$ for which $bold(r)_(i j)$ lies parallel to $chevron.l bold(q)_(i j) chevron.r$ will contract as their effective equilbrium distances become $tilde(r)_0 = gamma^(-1)r_0$, while it becomes $tilde(r)_0=gamma r_0$ for cells perpendicular to it. This behavior alongside the antisymmetry of $cos(2phi_(chevron.l bold(q)_(i j) chevron.r))$ around $phi_(chevron.l bold(q)_(i j) chevron.r) = pi/4$ keeps the cell area/volume approximately constant (see appendix [MAKE THE APPENDIX]).

Figure @fig_fig2 displays different equilibrium structures resulting from various $gamma$-values. The top of the figure displays a 10x10 cell sheet (A) for which the gamma-values are subsequently set to the following values and simulated to equilibrium: $gamma = 3/2$ (B) or $gamma = 2/3$ (C). Underneath a cylinder is shown (D) for which the same is true (E) and (F). 


#figure(
  image("FIgures/fig2.png"),
  caption: [Resulting elongations of simulated cell tissue in a ten by ten sheet of cells and 15 by 20 cylinder for different values of $gamma$. For $gamma < 1$ the cells are contracting along PCP direction and expanding perpendicular to it, for $gamma > 1$ the opposite is true [PROBABLY MORE]. Coloring is displaying depth.  
  ],
) <fig_fig2>

= Reactive shape change

[FLUFF]

To model reactive cell shape change we expand the original set of stochastic differential equations (SDE) of @nissen_theoretical_2018 to include the three new variables introduced. Assuming overdamped langevin dynamics the additional SDEs are

#numbered_eq($(d theta^parallel_i) / (d t) = - (partial V)/(partial theta^parallel_i) + eta$) <eq_SDE_theta_par>
#numbered_eq($(d theta^perp_i) / (d t) = - (partial V)/(partial theta^perp_i) + eta$) <eq_SDE_theta_perp>
#numbered_eq($(d tilde(gamma)_i) / (d t) = - (partial V)/(partial tilde(gamma)_i) + eta$) <eq_SDE_gamma>

where $V = sum V_(i j)$ (eq. @eq_pw_potential) and $gamma$ is reparametrized as $tilde(gamma) = ln(gamma)$ as $tilde(gamma) in [-ln(a), ln(a)]$ (where $a$ is the upper bound of contraction and elongation) treats contraction and elongation symmetrically as it is symmetric around the undeformed state $tilde(gamma) = 0$. With the aim of being able to model systems with both active and reactive cell deformations we make the inclusions of eq. @eq_SDE_theta_par through @eq_SDE_gamma in integration controllable on a per cell level, such that some cells can have a prescribed deformations that the surrounding cells adapt to.

== Reactive curvature

[REFORMULATE THIS TO INLCUDE ADAPTIVE GAMMA]
To test the effects of reactive cell wedging and reactive cell contraction and elongation we performed various mechanical deformations in different cell-sheet configurations (Figure @fig_fig3 row 1). In these simulations the initial configurations were run to equilibrium with no deformations (Figure @fig_fig3 row 2) ($alpha^parallel_i = alpha^perp_i = 0$ and $gamma_i=1$ for all cells with no inclusion of eq. @eq_SDE_theta_par through @eq_SDE_gamma in the integration). From here, equations @eq_SDE_theta_par and @eq_SDE_theta_perp were included in the integration and the structures were subjected to mechanical deformations (Figure @fig_fig3 row 3) until a specificed peak deformation was reached (Figure @fig_fig3 row 3) whereafter the structures were released and run to equilibrium (Figure @fig_fig3 row 4). The results clearly show that the cells around critical bends and folds adapt to the imposed deformation [REF TO FIGURE] and in many cases retain their deformed shape even after the structures have been released, displaying the plastic capabilities of the method. [THIS NEEDS TO BE A BIT MORE RIGOUROUS].   



#figure(
  image("FIgures/fig3.png"),
  caption: [Figure displays the effects of various mechanic deformations on various simulated cellular structures when $alpha_parallel$ and $alpha_perp$ are 'free' parameters, i.e. eq [REF] is included in the update equations. Coloring displays the value of either $alpha_parallel$ or $alpha_perp$ in degrees (see colorbar) depending on which is most important for the deformation shown. _Column 1_: A sheet of 20x10 cells is deformed by dragging the edge of the shortside of the sheet in a halfcircle and then down. The cells near the fold adapts and display much higher $alpha_parallel$ values than the remaining sheet. The resulting structure is stable. _Column 2_: A cylinder is squeezed between two infinite planes. $alpha_perp$ values near the folds are markedly larger and the structure is stable. _Column 3_: A sphere is squeezed between two infinite planes, again the cells around the folds adapt, but the structure is not completely stable and 'bounces back' a bit to become stable. _Column 4_: A sphere is stretched by dragging on a subset of cells on each side of it. The structure becomes increasingly tube-like as stretched and is stable.]
) <fig_fig3>


== Reactive elongation


#figure(
  image("FIgures/fig4.png"),
  caption: [Figure displays the effects of various mechanic deformations on simulated cellular structures when $gamma$ is a 'free' parameters, i.e. eq [REF] is included in the update equations. Coloring of cells displays values of $gamma$. _Column 1_: A 20x10 sheet of cells is pulled. [MORE]
  ],
)

== Combining curvature and elongation
- Figure 5: Ball being pulled with learnable alphas and gammas

= Active and adaptive domains
== The drosophila legs
= Discussion
= Conclusion

= Methods
== Simulation and model
 - Computational implementation of the model
 - Table of standard parameter choices
 - More?

= Appendix
== Area conservation of $gamma$

= Changes needed to be made
- *Figures*:
  - Consider keeping ABP and PCP as unit vectors on sphere figure
  - Include scale bars in all illustrations and seek to keep them as similar as possible
  - Keep all variables as close to each other as possible. Might need to redo some simulations here due to noise, deformation times, cell counts etc. Look into this.
  - Mark subset of pulled cells AND subset of screened out defect cells where applicable
  - Make null-case figures for everything
  - For figure 4: Redo stretch straight stretch such that the initial configuration doesn't look wonky
  - For figure 3+4: Add additional row displaying the peak deformation before it is let go.
  - Figure 1 should be reduced to 3 curvatures (as all others are rotations) and we need to add illustrations of cells wedging along the arrows) We also need to add cells elongating along the arrows of figure 2 
- *Results*:
 - Write out the math and be sure of it.


#bibliography("Adaptive Cell Shapes.bib")