# VAE Residual DNA Storage Scheme

## Overall Pipeline

```mermaid
flowchart LR
    X[Input CT image x] --> E[VAE Encoder]
    E --> Y[Latent y]
    Y --> QY[Quantize latent y_hat]
    QY --> D[VAE Decoder]
    D --> XT[Lossy approximation x_tilde]

    X --> R[Residual r = x - round x_tilde]
    XT --> R
    R --> QR[Near-lossless residual quantization q]

    QY --> LP[Latent stream]
    QR --> RP[Residual stream]
    LP --> PACK[Payload packing]
    RP --> PACK
    PACK --> DNA[Constrained DNA encoder]
    DNA --> SEQ[DNA sequence]
```

## Training Objective

```mermaid
flowchart TB
    X[Training CT patch x] --> MODEL[VAEResidualCodec]
    MODEL --> XT[x_tilde]
    MODEL --> XH[x_hat]
    MODEL --> LB[latent_bits]
    MODEL --> RB[residual_bits]

    XT --> MSE[MSE loss]
    XT --> L1[L1 loss]
    XT --> MSSSIM[MS-SSIM loss]
    X --> MSE
    X --> L1
    X --> MSSSIM

    LB --> RATE[rate_loss]
    RB --> RATE
    RATE --> LOSS[Total loss]
    MSE --> LOSS
    L1 --> LOSS
    MSSSIM --> LOSS
```

Total loss:

```text
loss = rate_loss
     + lambda_distortion * MSE(x_tilde, x)
     + lambda_l1 * L1(x_tilde, x)
     + lambda_ms_ssim * (1 - MS_SSIM(x_tilde, x))
```

## Near-lossless Reconstruction

```mermaid
flowchart LR
    X[Original image x] --> RES[Residual r]
    XT[Lossy approximation x_tilde] --> RES
    RES --> Q[Quantized residual q]
    Q --> RH[Residual compensation q * 2tau+1]
    XT --> SUM[Add residual compensation]
    RH --> SUM
    SUM --> XH[Near-lossless reconstruction x_hat]
```

Residual rule:

```text
step = 2 * tau + 1
q = round((x - round(x_tilde)) / step)
x_hat = round(x_tilde) + q * step
```

## DNA Payload Composition

```mermaid
flowchart LR
    META[Metadata\nwidth height channels tau\nlatent shape checksum] --> PAYLOAD[Binary payload]
    LAT[VAE latent y_hat] --> PAYLOAD
    RES[Residual symbols q] --> PAYLOAD
    PAYLOAD --> ZIP[Compressed payload]
    ZIP --> MAP[DNA constraint mapping]
    MAP --> FASTA[FASTA / DNA sequence]
```

Current measured composition on 512 x 512 CT examples:

```text
VAE latent sequence:     about 21,133 nt  ~= 8.2%
Residual sequence:       about 237,336 nt ~= 91.8%
Total DNA sequence:      about 258,468 nt
Approximate compression: about 4.06:1 versus raw 8-bit DNA mapping
```

## Decoding Pipeline

```mermaid
flowchart LR
    SEQ[DNA sequence] --> DEMAP[DNA decoder]
    DEMAP --> PAYLOAD[Recovered payload]
    PAYLOAD --> LAT[VAE latent y_hat]
    PAYLOAD --> RES[Residual symbols q]
    LAT --> DEC[VAE Decoder]
    DEC --> XT[x_tilde]
    RES --> COMP[Residual compensation]
    XT --> ADD[Add]
    COMP --> ADD
    ADD --> XH[Recovered image x_hat]
```

## Web Viewer

```mermaid
flowchart LR
    HTML[viewer.html] --> API[viewer_server.py]
    API --> CKPT[outputs/checkpoints/*.pth]
    API --> IMG[data/train val test images]
    CKPT --> INFER[Model inference]
    IMG --> INFER
    INFER --> OUT[Input image\nLossy approximation\nNear-lossless reconstruction\nPSNR MS-SSIM bpp]
    OUT --> HTML
```
