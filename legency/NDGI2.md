# Neural Dynamic GI (NDGI) Method Summary

## 1. Problem Formulation

Given temporal lightmaps:
L = {L_i}, L_i ∈ R^{H×W×C}

Traditional interpolation:
I(u,v,t) = Interpolate(L_{i-1}, L_i)

---

## 2. Neural Representation

Replace discrete lightmaps with neural function:

I(u,v,t) = H_Θ(u,v,t)

---

## 3. Model Parameters

Θ = {F_uvt^3D, F_uv^2D, F_ut^2D, F_vt^2D, Φ}

---

## 4. Feature Sampling

V_uvt = F_uvt^3D(u,v,t)
V_uv = F_uv^2D(u,v)
V_ut = F_ut^2D(u,t)
V_vt = F_vt^2D(v,t)

---

## 5. Time Encoding

γ(t) =
[sin(2^k πt), cos(2^k πt)]

---

## 6. Decoder

I(u,v,t) = MLP_Φ(V_uvt, V_uv, V_ut, V_vt, γ(t))

---

## 7. BC Compression

f_p = (1 - w_p)e1 + w_p e2

---

## 8. Quantization

Ṽ = V + U(-0.5,0.5)/256

---

## 9. Final Model

L → Θ = {feature maps + MLP}

I(u,v,t) = H_Θ(u,v,t)

---

## 10. Key Idea

- Replace storage with neural function
- Hybrid spatial-temporal feature decomposition
- BC-aware training for GPU compatibility