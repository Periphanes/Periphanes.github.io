---
layout: page
title: "DUST: Dual-Stream Diffusion"
permalink: /dust/
description: "Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model"
nav: false
nav_order: 99
_styles: >
  .dust-hero {
    text-align: center;
    padding: 2rem 0 1rem;
  }
  .dust-hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }
  .dust-venue {
    display: inline-block;
    background: #4a90d9;
    color: white;
    padding: 0.2rem 0.8rem;
    border-radius: 4px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 1rem;
  }
  .dust-authors {
    font-size: 1.1rem;
    margin-bottom: 0.3rem;
  }
  .dust-authors a {
    text-decoration: none;
  }
  .dust-affiliation {
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 1.2rem;
  }
  .dust-links {
    display: flex;
    justify-content: center;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
  }
  .dust-links a, .dust-links span {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 1rem;
    border: 1px solid #ddd;
    border-radius: 6px;
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    transition: background 0.2s;
  }
  .dust-links a:hover {
    background: #f0f0f0;
  }
  .dust-links .coming-soon {
    opacity: 0.5;
    cursor: default;
  }
  .dust-section {
    margin-bottom: 2.5rem;
  }
  .dust-section h2 {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #eee;
  }
  .dust-abstract {
    font-size: 1.0rem;
    line-height: 1.7;
    text-align: justify;
  }
  .dust-caption {
    text-align: center;
    font-size: 0.85rem;
    color: #666;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
  }
  .dust-results-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    text-align: center;
    margin: 1.5rem 0;
  }
  .dust-results-grid .result-card {
    padding: 1.5rem 1rem;
    border-radius: 8px;
    background: #f8f9fa;
  }
  .dust-results-grid .result-number {
    font-size: 2rem;
    font-weight: 700;
    color: #4a90d9;
  }
  .dust-results-grid .result-label {
    font-size: 0.85rem;
    color: #666;
    margin-top: 0.3rem;
  }
  .dust-bibtex {
    position: relative;
  }
  .dust-bibtex pre {
    background: #f5f5f5;
    padding: 1rem;
    border-radius: 6px;
    font-size: 0.8rem;
    overflow-x: auto;
  }
  @media (max-width: 768px) {
    .dust-results-grid {
      grid-template-columns: 1fr;
    }
    .dust-hero h1 {
      font-size: 1.6rem;
    }
  }
---

<div class="dust-hero">
  <h1>Dual-Stream Diffusion for World-Model<br>Augmented Vision-Language-Action Model</h1>
  <span class="dust-venue">ICML 2026</span>
  <div class="dust-authors">
    <a href="https://periphanes.github.io/">John Won</a>,
    <a href="#">Kyungmin Lee</a>,
    <a href="#">Huiwon Jang</a>,
    <a href="#">Dongyoung Kim</a>,
    <a href="#">Jinwoo Shin</a>
  </div>
  <div class="dust-affiliation">
    Kim Jaechul Graduate School of AI, KAIST &nbsp;&bull;&nbsp; RLWRLD
  </div>
  <div class="dust-links">
    <a href="https://arxiv.org/abs/2510.27607"><i class="fas fa-file-pdf"></i> arXiv</a>
    <span class="coming-soon"><i class="fab fa-github"></i> Code (Coming Soon)</span>
  </div>
</div>

<div class="dust-section">
  <h2>TL;DR</h2>
  <p class="dust-abstract">
    <b>DUST</b> augments VLAs with world modeling through a dual-stream diffusion transformer that jointly denoises actions and future visual states in separate-but-linked pathways.
  </p>
</div>

<div class="dust-section">
  <h2>Abstract</h2>
  <p class="dust-abstract">
    Augmenting Vision-Language-Action models (VLAs) with world models is promising for robotic policy learning but faces challenges in jointly predicting states and actions due to the modality gap.
    To address this, we propose <b>DU</b>al-<b>ST</b>ream diffusion (<b>DUST</b>), a world-model augmented VLA framework featuring a multimodal diffusion transformer that maintains separate modality streams while enabling cross-modal knowledge sharing.
    In addition, DUST utilizes independent noise perturbations and a decoupled flow matching loss to learn cross-modal causal relationships.
    We further introduce an asynchronous sampling method for action and vision tokens that enhances performance through inference-time scaling.
    Experimental results on simulated benchmarks like RoboCasa and GR-1 show that DUST achieves up to 6% gains over state-of-the-art VLA and world-modeling baselines, with inference-time scaling providing an additional 2&ndash;5% improvement.
    In real-world tasks using the Franka Research 3, DUST outperforms baselines by 10% in success rate.
    Finally, we demonstrate that DUST enables effective transfer learning through both pretraining on action-free videos and joint-training with heterogeneous robot and human datasets.
  </p>
</div>

<div class="dust-section">
  <h2>Key Results</h2>
  <div class="dust-results-grid">
    <div class="result-card">
      <div class="result-number">+6%</div>
      <div class="result-label">Over baseline VLAs on<br>simulation benchmarks</div>
    </div>
    <div class="result-card">
      <div class="result-number">+13%</div>
      <div class="result-label">Over baselines on<br>real-world Franka tasks</div>
    </div>
    <div class="result-card">
      <div class="result-number">~40Hz</div>
      <div class="result-label">Inference speed</div>
    </div>
  </div>
</div>

<div class="dust-section">
  <h2>Motivation</h2>
  <p>
    Existing approaches for joint world-modeling and action prediction face a fundamental trade-off. <b>Joint diffusion</b> models force both modalities into a single latent space, causing mismatches between low-dimensional actions and high-dimensional visual predictions. <b>Causal designs</b> separate modalities but limit information flow to one direction. DUST resolves this by maintaining <b>dual streams</b> that interact through shared attention while preserving modality-specific structure.
  </p>
  <div class="row mt-3">
    <div class="col-sm-4 mt-3 mt-md-0">
      {% include figure.liquid loading="eager" path="assets/img/dust/concept_a.png" title="Joint Diffusion" class="img-fluid rounded z-depth-1" %}
      <div class="dust-caption">(a) Joint Diffusion</div>
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
      {% include figure.liquid loading="eager" path="assets/img/dust/concept_b.png" title="Causal" class="img-fluid rounded z-depth-1" %}
      <div class="dust-caption">(b) Causal</div>
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
      {% include figure.liquid loading="eager" path="assets/img/dust/concept_c.png" title="Dual-Stream (Ours)" class="img-fluid rounded z-depth-1" %}
      <div class="dust-caption">(c) Dual-Stream (Ours)</div>
    </div>
  </div>
</div>

<div class="dust-section">
  <h2>Architecture</h2>
  <p>
    DUST is built upon a frozen vision-language model (VLM) backbone that provides semantic features from the current observation and task instruction. The core diffusion model uses a stack of <b>multimodal diffusion transformer (MMDiT)</b> blocks where action and vision token streams are propagated through separate pathways, concatenated only during shared cross-modal attention layers. Each stream receives its own timestep embedding via adaptive layer normalization, enabling <b>decoupled noise scheduling</b> during training. After the shared MMDiT layers, modality-specific DiT blocks handle specialized denoising for each stream.
  </p>
  <div class="row mt-3">
    <div class="col-12">
      {% include figure.liquid loading="eager" path="assets/img/dust/architecture.png" title="DUST Architecture" class="img-fluid rounded z-depth-1" %}
      <div class="dust-caption">DUST architecture. A frozen VLM provides conditioning features to the dual-stream diffusion model, which jointly denoises action and future observation tokens through shared MMDiT blocks followed by modality-specific DiT blocks.</div>
    </div>
  </div>
</div>

<div class="dust-section">
  <h2>Asynchronous Joint Sampling</h2>
  <p>
    During inference, DUST jointly samples actions and future visual observations. Since image embeddings operate in a higher-dimensional space and benefit from more denoising steps, we introduce <b>asynchronous denoising</b>: vision tokens are updated at every fine-grained step while action tokens are updated less frequently. This test-time scaling strategy provides a tunable trade-off between inference speed and predictive accuracy, yielding an additional 2&ndash;5% boost in success rate.
  </p>
  <div class="row mt-3">
    <div class="col-12">
      {% include figure.liquid loading="eager" path="assets/img/dust/diffusion_steps.png" title="Asynchronous Denoising" class="img-fluid rounded z-depth-1" %}
      <div class="dust-caption">Asynchronous joint sampling. Vision tokens receive more denoising steps than action tokens, enabling test-time scaling of visual prediction quality.</div>
    </div>
  </div>
</div>

<div class="dust-section">
  <h2>Qualitative Results</h2>
  <p>
    DUST produces physically consistent future predictions that guide accurate action generation across diverse manipulation tasks, including pick-and-place, insertion, and tool use.
  </p>
  <div class="row mt-3">
    <div class="col-12">
      {% include figure.liquid loading="eager" path="assets/img/dust/qualitative.jpg" title="Qualitative Results" class="img-fluid rounded z-depth-1" %}
      <div class="dust-caption">Example rollouts showing DUST's predicted future observations alongside actual task execution in real-world and simulated environments.</div>
    </div>
  </div>
</div>

<div class="dust-section">
  <h2>BibTeX</h2>
  <div class="dust-bibtex">

```bibtex
@inproceedings{won2026dust,
  title={Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model},
  author={Won, John and Lee, Kyungmin and Jang, Huiwon and Kim, Dongyoung and Shin, Jinwoo},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

  </div>
</div>
