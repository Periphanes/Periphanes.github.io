---
layout: page
title: "AFFECT: Action Effect Matching"
permalink: /affect/
description: "Learning What Actions Change: Action Effect Matching for Dynamics-Faithful World Models"
nav: false
nav_order: 99
_styles: >
  .proj-hero {
    text-align: center;
    padding: 2rem 0 1rem;
  }
  .proj-hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }
  .proj-venue {
    display: inline-block;
    background: #4a90d9;
    color: white;
    padding: 0.2rem 0.8rem;
    border-radius: 4px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 1rem;
  }
  .proj-authors {
    font-size: 1.1rem;
    margin-bottom: 0.3rem;
  }
  .proj-authors a {
    text-decoration: none;
  }
  .proj-affiliation {
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 1.2rem;
  }
  .proj-links {
    display: flex;
    justify-content: center;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
  }
  .proj-links a, .proj-links span {
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
  .proj-links a:hover {
    background: rgba(127, 127, 127, 0.12);
  }
  .proj-section {
    margin-bottom: 2.5rem;
  }
  .proj-section h2 {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid var(--global-divider-color);
  }
  .proj-section h3 {
    font-size: 1.15rem;
    font-weight: 600;
    margin-top: 1.5rem;
  }
  .proj-abstract {
    font-size: 1.0rem;
    line-height: 1.7;
    text-align: justify;
  }
  .proj-caption {
    text-align: center;
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
  }
  .proj-table-wrap {
    overflow-x: auto;
    margin: 1rem 0 0.5rem;
  }
  .proj-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    text-align: center;
    white-space: nowrap;
  }
  .proj-table th, .proj-table td {
    padding: 0.35rem 0.6rem;
    border-bottom: 1px solid var(--global-divider-color);
  }
  .proj-table th:first-child, .proj-table td:first-child {
    text-align: left;
  }
  .proj-table thead th {
    font-weight: 600;
  }
  .proj-table tr.group-start td {
    border-top: 2px solid var(--global-divider-color);
  }
  .proj-table tr.ours td {
    background: rgba(74, 144, 217, 0.12);
  }
  .proj-table .pearson {
    font-weight: 600;
  }
  .proj-split {
    display: grid;
    grid-template-columns: 3fr 2fr;
    gap: 1.5rem;
    align-items: center;
  }
  .proj-equation {
    overflow-x: auto;
    margin: 1rem 0;
  }
  .proj-bibtex pre {
    background: rgba(127, 127, 127, 0.1);
    padding: 1rem;
    border-radius: 6px;
    font-size: 0.8rem;
    overflow-x: auto;
  }
  .affect-build {
    position: relative;
    margin: 1rem 0 1.5rem;
  }
  .build-canvas {
    position: relative;
    aspect-ratio: 2700 / 1372;
    background: #fff;
    border-radius: 6px;
    overflow: hidden;
  }
  .build-layer {
    position: absolute;
    height: auto;
    max-width: none;
    transition: opacity 0.7s ease, transform 0.7s ease, clip-path 0.9s ease;
  }
  .affect-build.js .build-layer.future {
    opacity: 0;
    transform: translateY(6px);
    clip-path: inset(0 100% 0 0);
  }
  .affect-build.js .build-layer.past {
    opacity: 0.3;
  }
  .affect-build.js.done .build-layer.past {
    opacity: 0.45;
  }
  .affect-build.js.done.final .build-layer.past {
    opacity: 1;
  }
  .affect-build.js .build-layer.current.waiting {
    opacity: 0;
    transform: translateY(6px);
    clip-path: inset(0 100% 0 0);
  }
  .affect-build.js .build-layer.current {
    opacity: 1;
    transform: none;
    clip-path: inset(0 0 0 0);
  }
  .build-replay {
    position: absolute;
    bottom: 0.5rem;
    right: 0.5rem;
    z-index: 2;
    border: 1px solid #ccc;
    background: rgba(255, 255, 255, 0.9);
    color: #333;
    border-radius: 4px;
    font-size: 0.75rem;
    padding: 0.15rem 0.5rem;
    cursor: pointer;
    display: none;
  }
  .affect-build.js .build-replay {
    display: block;
  }
  @media (prefers-reduced-motion: reduce) {
    .build-layer { transition: none; }
  }
  .viz-root {
    --c-ours: #2a78d6;
    --c-base: #898781;
    --c-uniform: #bdbcb3;
    --c-dd: #898781;
    --ink: var(--global-text-color);
    --ink-2: var(--global-text-color-light);
    --grid: var(--global-divider-color);
    --axis: #c3c2b7;
    --surface: var(--global-bg-color);
  }
  :root[data-theme="dark"] .viz-root {
    --c-ours: #3987e5;
    --c-uniform: #55544f;
    --c-dd: #8f8e88;
    --axis: #4a4a47;
  }
  .viz-sub {
    color: var(--global-text-color-light);
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
  }
  .viz-chart {
    position: relative;
    margin: 0.5rem 0 1.5rem;
  }
  .viz-chart svg {
    display: block;
    overflow: visible;
  }
  .viz-chart text {
    font-family: inherit;
    font-size: 12px;
  }
  .viz-chart .t-ink { fill: var(--ink); }
  .viz-chart .t-2 { fill: var(--ink-2); }
  .viz-chart .t-strong { fill: var(--ink); font-weight: 600; }
  .viz-chart .gridline { stroke: var(--grid); stroke-width: 1; }
  .viz-chart .baseline { stroke: var(--axis); stroke-width: 1; }
  .viz-chart .hit { fill: transparent; cursor: default; outline: none; }
  .viz-chart .hit:hover, .viz-chart .hit:focus { fill: rgba(127, 127, 127, 0.08); }
  .viz-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 1.2rem;
    font-size: 0.85rem;
    color: var(--ink);
    margin-bottom: 0.4rem;
  }
  .viz-legend span.item {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .viz-legend i {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }
  .viz-legend i.dot { border-radius: 50%; width: 10px; height: 10px; }
  .viz-panels {
    display: flex;
    gap: 1.5rem;
  }
  .viz-panels > div { min-width: 0; }
  .viz-tip {
    position: absolute;
    pointer-events: none;
    background: var(--surface);
    color: var(--ink);
    border: 1px solid var(--grid);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    font-size: 0.8rem;
    line-height: 1.45;
    opacity: 0;
    transition: opacity 0.1s;
    z-index: 5;
    white-space: nowrap;
  }
  .viz-tip.show { opacity: 1; }
  .viz-tip .tt { color: var(--ink-2); margin-bottom: 0.15rem; }
  .viz-tip .row { display: flex; align-items: center; gap: 0.45rem; }
  .viz-tip .row b { font-weight: 600; }
  .viz-tip .row span.l { color: var(--ink-2); }
  .viz-tip .key { display: inline-block; width: 12px; height: 2px; border-radius: 1px; }
  .viz-chart .grow-x, .viz-chart .grow-x-rev, .viz-chart .grow-y {
    transform-box: fill-box;
    transition: transform 0.8s cubic-bezier(0.2, 0.7, 0.2, 1);
  }
  .viz-chart .grow-x { transform: scaleX(0); transform-origin: left center; }
  .viz-chart .grow-x-rev { transform: scaleX(0); transform-origin: right center; }
  .viz-chart .grow-y { transform: scaleY(0); transform-origin: center bottom; }
  .viz-chart .slide {
    transform: translateX(var(--dx));
    transition: transform 0.8s cubic-bezier(0.2, 0.7, 0.2, 1);
  }
  .viz-chart .fade { opacity: 0; transition: opacity 0.4s ease; }
  .viz-chart.in .grow-x, .viz-chart.in .grow-x-rev, .viz-chart.in .grow-y, .viz-chart.in .slide { transform: none; }
  .viz-chart.in .fade { opacity: 1; }
  .viz-details {
    margin: 0.5rem 0 1rem;
  }
  .viz-details summary {
    cursor: pointer;
    color: var(--global-theme-color);
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
  }
  @media (prefers-reduced-motion: reduce) {
    .viz-chart .grow-x, .viz-chart .grow-x-rev, .viz-chart .grow-y, .viz-chart .slide, .viz-chart .fade { transition: none; }
  }
  @media (max-width: 768px) {
      .proj-split {
      grid-template-columns: 1fr;
    }
    .proj-hero h1 {
      font-size: 1.6rem;
    }
  }
---

<div class="proj-hero">
  <h1>Learning What Actions Change:<br>Action Effect Matching for Dynamics-Faithful World Models</h1>
  <span class="proj-venue">Preprint</span>
  <div class="proj-authors">
    <a href="https://periphanes.github.io/">John Won</a><sup>1</sup>,
    <a href="#">Yonghoon Dong</a><sup>1</sup>,
    <a href="https://alinlab.kaist.ac.kr/shin.html">Jinwoo Shin</a><sup>1,2</sup>
  </div>
  <div class="proj-affiliation">
    <sup>1</sup>KAIST &nbsp;&bull;&nbsp; <sup>2</sup>RLWRLD
  </div>
  <div class="proj-links">
    <a href="https://arxiv.org/abs/2611.25980"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M14 14V4.5L9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2M9.5 3A1.5 1.5 0 0 0 11 4.5h2V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1h5.5z"/></svg> arXiv</a>
  </div>
</div>

<div class="proj-section">
  <h2>TL;DR</h2>
  <p class="proj-abstract">
    <b>AFFECT</b> is an auxiliary training objective that teaches robot world models <i>how</i> small action changes alter the future: generated videos are trained to respond to logged action perturbations the same way recorded videos do, making world models more faithful for policy evaluation and planning.
  </p>
</div>

<div class="proj-section">
  <h2>Motivation</h2>
  <p>
    To serve as policy evaluators or planners, robot world models must reflect how <b>small action changes</b> alter future outcomes. A slight change in reaching motion can decide whether a gripper grasps a cube, misses it, or pushes it away. Yet existing models tend to produce visually plausible videos that ignore such differences, predicting similar futures for actions with very different real consequences. We therefore treat action sensitivity as a <b>distributional problem</b>: a world model should capture how the distribution of future observations shifts as actions vary under a given context.
  </p>
  <div class="row mt-3">
    <div class="col-12">
      {% include figure.liquid loading="eager" path="assets/img/affect/motivation.jpg" title="Small action perturbations lead to distinct action effects" class="img-fluid rounded z-depth-0" %}
      <div class="proj-caption">From the same initial state, actions drawn from a narrow distribution change the gripper's approach only slightly, yet produce qualitatively different outcomes: the cube is grasped, missed, or pushed away.</div>
    </div>
  </div>
</div>

<div class="proj-section">
  <h2>Method</h2>
  <p>
    AFFECT asks generated videos to <b>respond to action perturbations the same way recorded videos do</b>, rather than asking each generated video to match its recorded counterpart. This needs interaction data in which actions are perturbed in a known way, and a loss that measures the response to those perturbations.
  </p>

  <h3>1. Perturbation play</h3>
  <figure class="affect-build" data-show="1">
    <div class="build-canvas">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_scene.png' | relative_url }}" alt="Robot scene" style="left:0%;top:0%;width:25.815%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_policy.png' | relative_url }}" alt="Play policy producing nominal actions" style="left:0%;top:47.376%;width:16.296%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_noise.png' | relative_url }}" alt="Random perturbation" style="left:0%;top:90.015%;width:16.296%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_perturbed.png' | relative_url }}" alt="Perturbed actions executed on the robot" style="left:16.296%;top:47.376%;width:10.667%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_recorded.png' | relative_url }}" alt="Recorded future" style="left:25.815%;top:9.475%;width:14.481%" loading="lazy">
      <img class="build-layer" data-step="2" src="{{ '/assets/img/affect/method/s2_worldmodel.png' | relative_url }}" alt="World model generating a future" style="left:26.963%;top:37.901%;width:13.333%" loading="lazy">
      <img class="build-layer" data-step="2" src="{{ '/assets/img/affect/method/s2_encoders.png' | relative_url }}" alt="Frozen encoders and action effects" style="left:11.926%;top:0%;width:52.519%" loading="lazy">
      <img class="build-layer" data-step="3" src="{{ '/assets/img/affect/method/s3_matching.png' | relative_url }}" alt="Distributional matching of action effects" style="left:64.444%;top:0%;width:35.556%" loading="lazy">
      <button type="button" class="build-replay" aria-label="Replay animation">&#8635; Replay</button>
    </div>
    <figcaption class="proj-caption">Step 1. The play policy proposes nominal actions, logged random noise perturbs them, and the robot executes the perturbed actions to record a future.</figcaption>
  </figure>
  <p>
    We train a <b>play policy</b> on task-agnostic human teleoperation and roll it out autonomously, which needs no scene resets or task-specific procedures. Before execution, we inject <b>temporally correlated noise</b> into each nominal action chunk and log the perturbation; the robot executes the perturbed chunk and we record the resulting future. Each window needs just a single recorded future, so <b>no repeated executions from the same initial state</b> are required.
  </p>

  <h3>2. Action effect encoding</h3>
  <figure class="affect-build" data-show="2">
    <div class="build-canvas">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_scene.png' | relative_url }}" alt="Robot scene" style="left:0%;top:0%;width:25.815%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_policy.png' | relative_url }}" alt="Play policy producing nominal actions" style="left:0%;top:47.376%;width:16.296%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_noise.png' | relative_url }}" alt="Random perturbation" style="left:0%;top:90.015%;width:16.296%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_perturbed.png' | relative_url }}" alt="Perturbed actions executed on the robot" style="left:16.296%;top:47.376%;width:10.667%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_recorded.png' | relative_url }}" alt="Recorded future" style="left:25.815%;top:9.475%;width:14.481%" loading="lazy">
      <img class="build-layer" data-step="2" src="{{ '/assets/img/affect/method/s2_worldmodel.png' | relative_url }}" alt="World model generating a future" style="left:26.963%;top:37.901%;width:13.333%" loading="lazy">
      <img class="build-layer" data-step="2" src="{{ '/assets/img/affect/method/s2_encoders.png' | relative_url }}" alt="Frozen encoders and action effects" style="left:11.926%;top:0%;width:52.519%" loading="lazy">
      <img class="build-layer" data-step="3" src="{{ '/assets/img/affect/method/s3_matching.png' | relative_url }}" alt="Distributional matching of action effects" style="left:64.444%;top:0%;width:35.556%" loading="lazy">
      <button type="button" class="build-replay" aria-label="Replay animation">&#8635; Replay</button>
    </div>
    <figcaption class="proj-caption">Step 2. The world model generates a future from the same context and actions. Frozen encoders &phi; and &psi; summarize the videos and the logged perturbation, and their products give the action effects.</figcaption>
  </figure>
  <p>
    The world model sees only the context frame and the executed actions, and generates a future from fresh noise with a short differentiable sampler, <b>never conditioned on the recorded future</b>. Two frozen encoders then summarize the videos (&phi;) and the logged perturbation (&psi;) with simple, hand-specified features. After centering the perturbation features by their expected value, the <b>action effect</b> is the product of perturbation and video features: positive when larger-than-expected perturbations come with larger values of a video feature. Keeping the encoders frozen prevents a trivial solution, and lets AFFECT plug into existing world models <b>without architectural changes</b>.
  </p>

  <h3>3. Distributional matching</h3>
  <figure class="affect-build" data-show="3">
    <div class="build-canvas">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_scene.png' | relative_url }}" alt="Robot scene" style="left:0%;top:0%;width:25.815%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_policy.png' | relative_url }}" alt="Play policy producing nominal actions" style="left:0%;top:47.376%;width:16.296%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_noise.png' | relative_url }}" alt="Random perturbation" style="left:0%;top:90.015%;width:16.296%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_perturbed.png' | relative_url }}" alt="Perturbed actions executed on the robot" style="left:16.296%;top:47.376%;width:10.667%" loading="lazy">
      <img class="build-layer" data-step="1" src="{{ '/assets/img/affect/method/s1_recorded.png' | relative_url }}" alt="Recorded future" style="left:25.815%;top:9.475%;width:14.481%" loading="lazy">
      <img class="build-layer" data-step="2" src="{{ '/assets/img/affect/method/s2_worldmodel.png' | relative_url }}" alt="World model generating a future" style="left:26.963%;top:37.901%;width:13.333%" loading="lazy">
      <img class="build-layer" data-step="2" src="{{ '/assets/img/affect/method/s2_encoders.png' | relative_url }}" alt="Frozen encoders and action effects" style="left:11.926%;top:0%;width:52.519%" loading="lazy">
      <img class="build-layer" data-step="3" src="{{ '/assets/img/affect/method/s3_matching.png' | relative_url }}" alt="Distributional matching of action effects" style="left:64.444%;top:0%;width:35.556%" loading="lazy">
      <button type="button" class="build-replay" aria-label="Replay animation">&#8635; Replay</button>
    </div>
    <figcaption class="proj-caption">Step 3. AFFECT aligns the distribution of generated action effects with the recorded ones, so generated futures respond to perturbations in the same direction and with the same strength as real ones.</figcaption>
  </figure>
  <p>
    Averaged over many windows, the action effects reveal how the scene responds to actions. AFFECT trains the world model so that the action effects of generated videos match those of recorded videos:
  </p>
  <div class="proj-equation">
    $$\mathcal{L}_{\mathrm{AFFECT}} = \sum_{i,j}\left(\frac{\mathbb{E}\big[g_{t,i}\,\big(\phi_j(O_t)-\phi_j(\hat{O}_t)\big)\big]}{s_{ij}}\right)^{2}, \qquad \mathcal{L} = \mathcal{L}_{\mathrm{flow}} + \lambda\,\mathcal{L}_{\mathrm{AFFECT}}.$$
  </div>
  <p>
    Centering turns each action effect into a <b>conditional covariance</b> between perturbation and video features, so scene-only errors (e.g. rendering a scene slightly too dark) cancel out, and the loss compares only the direction and strength of the response to actions. The full objective is a finite-feature <b>maximum mean discrepancy</b> over action effects. Unlike plain feature matching, it catches a model that moves an object left whenever the real object moves right.
  </p>
</div>

<div class="proj-section">
  <h2>Policy Evaluation Results</h2>
  <p>
    We apply AFFECT to two state-of-the-art robot world models, <b>DreamDojo</b> and <b>Cosmos3</b>, and run real policies in closed loop inside each world model. If the world model is faithful, policies that succeed more often on the real robot should also succeed more often in imagination.
  </p>

  <h3>Does the world model rank policies like the real robot?</h3>
  <p class="viz-sub">Pearson correlation between policy success rates in the world model and in the real environment. AFFECT improves it in all six backbone&ndash;embodiment combinations.</p>
  <div class="viz-root viz-chart" id="viz-pearson" role="group" aria-label="Pearson correlation of real versus imagined policy success, backbone versus backbone plus AFFECT, for six backbone and embodiment combinations. Values are in the table below."></div>

  <h3>Are the generated videos closer to reality?</h3>
  <p class="viz-sub">Gain in long-horizon PSNR (dB) after adding AFFECT. AFFECT improves it in all six combinations.</p>
  <div class="viz-root viz-chart" id="viz-psnr" role="group" aria-label="Gain in long-horizon PSNR after adding AFFECT for six backbone and embodiment combinations. Values are in the table below."></div>

  <details class="viz-details">
    <summary>Show full results tables (PSNR, LPIPS, FVD, Pearson, and other baselines)</summary>

    <h3>Real robots: DROID (Franka) and ALOHA-2</h3>
    <div class="proj-table-wrap">
      <table class="proj-table">
        <thead>
          <tr><th rowspan="2">Method</th><th colspan="4">DROID (Real)</th><th colspan="4">ALOHA-2 (Real)</th></tr>
          <tr><th>PSNR &uarr;</th><th>LPIPS &darr;</th><th>FVD &darr;</th><th>Pearson &uarr;</th><th>PSNR &uarr;</th><th>LPIPS &darr;</th><th>FVD &darr;</th><th>Pearson &uarr;</th></tr>
        </thead>
        <tbody>
          <tr><td>iVideoGPT</td><td>16.16</td><td>0.52</td><td>1946</td><td class="pearson">0.151</td><td>16.77</td><td>0.52</td><td>3808</td><td class="pearson">0.155</td></tr>
          <tr><td>CoCo</td><td>18.13</td><td>0.50</td><td>1676</td><td class="pearson">0.212</td><td>19.09</td><td>0.49</td><td>3959</td><td class="pearson">0.225</td></tr>
          <tr><td>Ctrl-World</td><td>17.48</td><td>0.36</td><td>1141</td><td class="pearson">0.236</td><td>20.32</td><td>0.11</td><td>396</td><td class="pearson">0.399</td></tr>
          <tr class="group-start"><td>Cosmos3</td><td>18.69</td><td>0.24</td><td>910</td><td class="pearson">0.425</td><td>22.94</td><td>0.08</td><td>333</td><td class="pearson">0.298</td></tr>
          <tr class="ours"><td><b>+ AFFECT</b></td><td><b>19.67</b></td><td>0.22</td><td>744</td><td class="pearson">0.583</td><td>24.05</td><td><b>0.05</b></td><td>257</td><td class="pearson"><b>0.694</b></td></tr>
          <tr class="group-start"><td>DreamDojo</td><td>18.53</td><td>0.16</td><td>807</td><td class="pearson">0.468</td><td>24.20</td><td>0.06</td><td><b>183</b></td><td class="pearson">0.325</td></tr>
          <tr class="ours"><td><b>+ AFFECT</b></td><td>19.04</td><td><b>0.14</b></td><td><b>622</b></td><td class="pearson"><b>0.790</b></td><td><b>24.91</b></td><td><b>0.05</b></td><td>188</td><td class="pearson">0.490</td></tr>
        </tbody>
      </table>
    </div>
    <div class="proj-caption">Long-horizon autoregressive video fidelity and Pearson correlation of real vs. generated policy success rates.</div>

    <h3>Simulation: OGBench UR5e</h3>
    <div class="proj-table-wrap">
      <table class="proj-table">
        <thead>
          <tr><th rowspan="2">Method</th><th rowspan="2">Pearson &uarr;</th><th colspan="3">Short Horizon</th><th colspan="3">Long Horizon</th></tr>
          <tr><th>PSNR &uarr;</th><th>LPIPS &darr;</th><th>FVD &darr;</th><th>PSNR &uarr;</th><th>LPIPS &darr;</th><th>FVD &darr;</th></tr>
        </thead>
        <tbody>
          <tr><td>iVideoGPT</td><td class="pearson">0.334</td><td>21.53</td><td>0.49</td><td>1134</td><td>20.20</td><td>0.53</td><td>1314</td></tr>
          <tr><td>CoCo</td><td class="pearson">0.130</td><td>21.69</td><td>0.49</td><td>1094</td><td>20.29</td><td>0.52</td><td>1331</td></tr>
          <tr><td>Ctrl-World</td><td class="pearson">0.310</td><td>24.99</td><td>0.10</td><td>156</td><td>24.27</td><td>0.11</td><td>196</td></tr>
          <tr class="group-start"><td>Cosmos3</td><td class="pearson">0.091</td><td>19.07</td><td>0.19</td><td>470</td><td>18.72</td><td>0.22</td><td>562</td></tr>
          <tr class="ours"><td><b>+ AFFECT</b></td><td class="pearson">0.334</td><td>19.95</td><td>0.18</td><td>463</td><td>19.21</td><td>0.20</td><td>540</td></tr>
          <tr class="group-start"><td>DreamDojo</td><td class="pearson">0.451</td><td>26.03</td><td>0.06</td><td>85</td><td>24.87</td><td>0.08</td><td>108</td></tr>
          <tr class="ours"><td><b>+ AFFECT</b></td><td class="pearson"><b>0.686</b></td><td><b>26.78</b></td><td><b>0.05</b></td><td><b>73</b></td><td><b>25.58</b></td><td><b>0.07</b></td><td><b>93</b></td></tr>
        </tbody>
      </table>
    </div>
    <div class="proj-caption">Single-window (short horizon) and autoregressive (long horizon) video fidelity in simulation, with Pearson correlation on OGBench pick-and-place tasks.</div>
  </details>
</div>

<div class="proj-section">
  <h2>Planning with Action-Faithful World Models</h2>
  <p>
    Five policy checkpoints propose candidate action chunks; the world model imagines a video for each, and a learned success estimator picks the best one. With the proposals and estimator held fixed, AFFECT raises average success from <b>60.0% to 68.3%</b> on UR5e pick-and-place and from <b>72.2% to 80.6%</b> on a long-horizon ALOHA-2 task (open a trash bin, then throw away two items).
  </p>
  <div class="viz-root viz-chart" id="viz-planning" role="group" aria-label="Planning success rates for Uniform, DreamDojo, and DreamDojo plus AFFECT on UR5e tasks and ALOHA-2 subtasks. Values are in the table below."></div>
  <div class="proj-caption">Uniform picks a candidate at random without a world model; DreamDojo and DreamDojo + AFFECT imagine each candidate and pick the highest-scoring one.</div>
  <details class="viz-details">
    <summary>Show planning results table</summary>
    <div class="proj-table-wrap">
      <table class="proj-table">
        <thead>
          <tr><th>Task</th><th>Uniform</th><th>DreamDojo</th><th>DreamDojo + AFFECT</th></tr>
        </thead>
        <tbody>
          <tr><td>UR5e &middot; Vertical2</td><td>33.3</td><td>25.0</td><td>33.3</td></tr>
          <tr><td>UR5e &middot; Horizontal</td><td>50.0</td><td>83.3</td><td>83.3</td></tr>
          <tr><td>UR5e &middot; Diagonal1</td><td>33.3</td><td>58.3</td><td>66.7</td></tr>
          <tr><td>UR5e &middot; Diagonal2</td><td>50.0</td><td>58.3</td><td>75.0</td></tr>
          <tr><td>UR5e &middot; Lateral Near</td><td>66.7</td><td>75.0</td><td>83.3</td></tr>
          <tr class="ours"><td><b>UR5e &middot; Average</b></td><td>46.7</td><td>60.0</td><td><b>68.3</b></td></tr>
          <tr class="group-start"><td>ALOHA-2 &middot; Subtask 1</td><td>83.3</td><td>91.7</td><td>91.7</td></tr>
          <tr><td>ALOHA-2 &middot; Subtask 2</td><td>58.3</td><td>66.7</td><td>83.3</td></tr>
          <tr><td>ALOHA-2 &middot; Subtask 3</td><td>25.0</td><td>58.3</td><td>66.7</td></tr>
          <tr class="ours"><td><b>ALOHA-2 &middot; Average</b></td><td>55.6</td><td>72.2</td><td><b>80.6</b></td></tr>
        </tbody>
      </table>
    </div>
  </details>
</div>

<div class="proj-section">
  <h2>Where Do the Features See Action Effects?</h2>
  <div class="proj-split">
    <p>
      On recorded UR5e videos, we measure how consistently perturbation&ndash;feature products keep the same sign across episodes at each latent cell, relative to a baseline that pairs perturbations with videos from other windows. Responses are <b>1.6&ndash;2.2&times; stronger</b> than this shuffled baseline in regions traversed by the arm, cube, or shadow, and stay at chance on the static table and backdrop, so the frozen features provide a <b>localized action-effect signal</b>.
    </p>
    <div>
      {% include figure.liquid loading="eager" path="assets/img/affect/effect_signal.jpg" title="Action-effect localization" class="img-fluid rounded z-depth-0" %}
      <div class="proj-caption">Frozen-feature responses relative to shuffled pairs; brighter cells indicate stronger responses.</div>
    </div>
  </div>
</div>

<div class="proj-section">
  <h2>BibTeX</h2>
  <div class="proj-bibtex">
<pre><code>@article{won2026affect,
  title={Learning What Actions Change: Action Effect Matching for Dynamics-Faithful World Models},
  author={Won, John and Dong, Yonghoon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2611.25980},
  year={2026}
}</code></pre>
  </div>
</div>

<script>
  (function () {
    var figs = document.querySelectorAll(".affect-build");
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    function play(fig) {
      var show = +fig.dataset.show;
      fig.classList.remove("done");
      var current = [];
      fig.querySelectorAll(".build-layer").forEach(function (el) {
        var st = +el.dataset.step;
        el.classList.remove("past", "current", "future", "waiting");
        el.style.transitionDelay = "0s";
        if (st < show) el.classList.add("past");
        else if (st > show) el.classList.add("future");
        else { el.classList.add("current", "waiting"); current.push(el); }
      });
      if (reduce) {
        current.forEach(function (el) { el.classList.remove("waiting"); });
        fig.classList.add("done");
        return;
      }
      void fig.offsetWidth;
      var gap = 450;
      current.forEach(function (el, i) {
        el.style.transitionDelay = (i * gap) / 1000 + "s";
        el.classList.remove("waiting");
      });
      clearTimeout(fig._t);
      fig._t = setTimeout(function () { fig.classList.add("done"); }, current.length * gap + 900);
    }
    figs.forEach(function (fig, i) {
      fig.classList.add("js");
      if (i === figs.length - 1) fig.classList.add("final");
      fig.querySelectorAll(".build-layer").forEach(function (el) {
        var st = +el.dataset.step, show = +fig.dataset.show;
        el.classList.add(st < show ? "past" : st > show ? "future" : "current");
        if (st === show) el.classList.add("waiting");
      });
      fig.querySelector(".build-replay").addEventListener("click", function () { play(fig); });
    });
    if (!("IntersectionObserver" in window)) { figs.forEach(play); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { play(e.target); io.unobserve(e.target); }
      });
    }, { threshold: 0.5 });
    figs.forEach(function (fig) { io.observe(fig); });
  })();
</script>

<script>
  (function () {
    var NS = "http://www.w3.org/2000/svg";
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    function el(tag, attrs, parent) {
      var e = document.createElementNS(NS, tag);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      if (parent) parent.appendChild(e);
      return e;
    }
    function txt(parent, x, y, str, cls, anchor) {
      var t = el("text", { x: x, y: y, class: cls || "t-2", "text-anchor": anchor || "start", "dominant-baseline": "middle" }, parent);
      t.textContent = str;
      return t;
    }
    function hbar(x0, x1, y, h, r) {
      var len = Math.abs(x1 - x0);
      r = Math.min(r, len, h / 2);
      var d = x1 >= x0 ? 1 : -1;
      return "M" + x0 + "," + y + "H" + (x1 - d * r) + "Q" + x1 + "," + y + " " + x1 + "," + (y + r) + "V" + (y + h - r) + "Q" + x1 + "," + (y + h) + " " + (x1 - d * r) + "," + (y + h) + "H" + x0 + "Z";
    }
    function vbar(x, w, y0, y1, r) {
      r = Math.min(r, y0 - y1, w / 2);
      return "M" + x + "," + y0 + "V" + (y1 + r) + "Q" + x + "," + y1 + " " + (x + r) + "," + y1 + "H" + (x + w - r) + "Q" + (x + w) + "," + y1 + " " + (x + w) + "," + (y1 + r) + "V" + y0 + "Z";
    }
    function fmt(v, d) { return v.toFixed(d).replace("-", "−"); }
    function legend(root, items) {
      var lg = document.createElement("div");
      lg.className = "viz-legend";
      items.forEach(function (it) {
        var s = document.createElement("span");
        s.className = "item";
        var i = document.createElement("i");
        if (it.dot) i.className = "dot";
        i.style.background = "var(" + it.color + ")";
        s.appendChild(i);
        s.appendChild(document.createTextNode(it.label));
        lg.appendChild(s);
      });
      root.appendChild(lg);
    }
    function tooltip(root) {
      var t = document.createElement("div");
      t.className = "viz-tip";
      root.appendChild(t);
      return t;
    }
    function bindTip(root, tip, target, title, rows) {
      target.setAttribute("tabindex", "0");
      function show(evt) {
        tip.textContent = "";
        var h = document.createElement("div");
        h.className = "tt";
        h.textContent = title;
        tip.appendChild(h);
        rows.forEach(function (r) {
          var row = document.createElement("div");
          row.className = "row";
          if (r.color) {
            var k = document.createElement("span");
            k.className = "key";
            k.style.background = "var(" + r.color + ")";
            row.appendChild(k);
          }
          var b = document.createElement("b");
          b.textContent = r.value;
          row.appendChild(b);
          var l = document.createElement("span");
          l.className = "l";
          l.textContent = r.label;
          row.appendChild(l);
          tip.appendChild(row);
        });
        var rb = root.getBoundingClientRect();
        var x, y;
        if (evt && evt.clientX !== undefined) { x = evt.clientX - rb.left; y = evt.clientY - rb.top; }
        else { var tb = target.getBoundingClientRect(); x = tb.left - rb.left + tb.width / 2; y = tb.top - rb.top; }
        tip.classList.add("show");
        var tw = tip.offsetWidth, th = tip.offsetHeight;
        var left = Math.min(Math.max(0, x + 14), rb.width - tw);
        if (x + 14 + tw > rb.width) left = Math.max(0, x - tw - 14);
        tip.style.left = left + "px";
        tip.style.top = Math.max(0, y - th - 10) + "px";
      }
      function hide() { tip.classList.remove("show"); }
      target.addEventListener("pointermove", show);
      target.addEventListener("pointerleave", hide);
      target.addEventListener("focus", function () { show(); });
      target.addEventListener("blur", hide);
    }

    var ROWS = [
      { g: "UR5e (sim)", m: "DreamDojo", pb: 0.451, po: 0.686, sb: 24.87, so: 25.58 },
      { g: "UR5e (sim)", m: "Cosmos3", pb: 0.091, po: 0.334, sb: 18.72, so: 19.21 },
      { g: "DROID (real)", m: "DreamDojo", pb: 0.468, po: 0.79, sb: 18.53, so: 19.04 },
      { g: "DROID (real)", m: "Cosmos3", pb: 0.425, po: 0.583, sb: 18.69, so: 19.67 },
      { g: "ALOHA-2 (real)", m: "DreamDojo", pb: 0.325, po: 0.49, sb: 24.2, so: 24.91 },
      { g: "ALOHA-2 (real)", m: "Cosmos3", pb: 0.298, po: 0.694, sb: 22.94, so: 24.05 },
    ];

    // Shared row layout: a group header row, then one row per backbone.
    function rowLayout(W) {
      var labelW = W < 520 ? 92 : 150, rowH = 30, headH = 24, gap = 8, y = 0, out = [], last = null;
      ROWS.forEach(function (r) {
        if (r.g !== last) { if (last) y += gap; out.push({ head: r.g, y: y + headH / 2 }); y += headH; last = r.g; }
        out.push({ row: r, y: y + rowH / 2 });
        y += rowH;
      });
      return { labelW: labelW, items: out, h: y };
    }

    function drawPearson(root) {
      var W = root.clientWidth;
      root.textContent = "";
      legend(root, [ { label: "Backbone (flow loss only)", color: "--c-base", dot: true }, { label: "+ AFFECT", color: "--c-ours", dot: true } ]);
      var tip = tooltip(root);
      var lay = rowLayout(W), axisH = 26, padR = W < 520 ? 74 : 104;
      var x0 = lay.labelW + 8, x1 = W - padR, max = 0.8;
      var sx = function (v) { return x0 + (v / max) * (x1 - x0); };
      var svg = el("svg", { width: W, height: lay.h + axisH }, root);
      [0, 0.2, 0.4, 0.6, 0.8].forEach(function (t) {
        el("line", { x1: sx(t), x2: sx(t), y1: 0, y2: lay.h, class: t === 0 ? "baseline" : "gridline" }, svg);
        txt(svg, sx(t), lay.h + 14, t.toFixed(1), "t-2", "middle");
      });
      var i = 0;
      lay.items.forEach(function (it) {
        if (it.head) { txt(svg, 0, it.y, it.head, "t-strong"); return; }
        var r = it.row, bx = sx(r.pb), ox = sx(r.po), y = it.y, delay = 0.15 + i * 0.12;
        i++;
        var hit = el("rect", { x: 0, y: y - 15, width: W, height: 30, rx: 4, class: "hit" }, svg);
        txt(svg, 12, y, r.m, "t-2");
        var bar = el("rect", { x: bx, y: y - 1, width: ox - bx, height: 2, rx: 1, fill: "var(--c-ours)", class: "grow-x" }, svg);
        bar.style.transitionDelay = delay + "s";
        el("circle", { cx: bx, cy: y, r: 5, fill: "var(--c-base)", stroke: "var(--surface)", "stroke-width": 2 }, svg);
        var dot = el("circle", { cx: ox, cy: y, r: 5, fill: "var(--c-ours)", stroke: "var(--surface)", "stroke-width": 2, class: "slide" }, svg);
        dot.style.setProperty("--dx", bx - ox + "px");
        dot.style.transitionDelay = delay + "s";
        var lab = el("text", { x: ox + 11, y: y, "dominant-baseline": "middle", class: "t-ink fade" }, svg);
        var v = el("tspan", { "font-weight": 600 }, lab);
        v.textContent = r.po.toFixed(3);
        var dl = el("tspan", { class: "t-2", dx: 5 }, lab);
        dl.textContent = "+" + (r.po - r.pb).toFixed(2);
        lab.style.transitionDelay = delay + 0.7 + "s";
        bindTip(root, tip, hit, r.g + " · " + r.m, [
          { value: r.po.toFixed(3), label: "+ AFFECT", color: "--c-ours" },
          { value: r.pb.toFixed(3), label: "Backbone", color: "--c-base" },
        ]);
        hit.setAttribute("aria-label", r.g + " " + r.m + ": backbone " + r.pb + ", with AFFECT " + r.po);
      });
    }

    function drawPSNR(root) {
      var W = root.clientWidth;
      root.textContent = "";
      var tip = tooltip(root);
      var lay = rowLayout(W), axisH = 26, padR = 70, max = 1.2;
      var x0 = lay.labelW + 8, x1 = W - padR;
      var sx = function (v) { return x0 + (v / max) * (x1 - x0); };
      var svg = el("svg", { width: W, height: lay.h + axisH }, root);
      [0, 0.5, 1.0].forEach(function (t) {
        el("line", { x1: sx(t), x2: sx(t), y1: 0, y2: lay.h, class: t === 0 ? "baseline" : "gridline" }, svg);
        txt(svg, sx(t), lay.h + 14, t === 0 ? "0 dB" : "+" + t.toFixed(1) + " dB", "t-2", "middle");
      });
      var i = 0;
      lay.items.forEach(function (it) {
        if (it.head) { txt(svg, 0, it.y, it.head, "t-strong"); return; }
        var r = it.row, gain = r.so - r.sb, y = it.y, delay = 0.15 + i * 0.12;
        i++;
        var hit = el("rect", { x: 0, y: y - 15, width: W, height: 30, rx: 4, class: "hit" }, svg);
        txt(svg, 12, y, r.m, "t-2");
        var bar = el("path", { d: hbar(sx(0), sx(gain), y - 7, 14, 4), fill: "var(--c-ours)", class: "grow-x" }, svg);
        bar.style.transitionDelay = delay + "s";
        var lab = txt(svg, sx(gain) + 6, y, "+" + gain.toFixed(2) + " dB", "t-ink fade");
        lab.style.transitionDelay = delay + 0.6 + "s";
        bindTip(root, tip, hit, r.g + " \u00b7 " + r.m, [
          { value: "+" + gain.toFixed(2) + " dB", label: "PSNR gain", color: "--c-ours" },
          { value: r.sb.toFixed(2) + " \u2192 " + r.so.toFixed(2), label: "PSNR, backbone \u2192 + AFFECT" },
        ]);
        hit.setAttribute("aria-label", r.g + " " + r.m + ": PSNR " + r.sb + " to " + r.so + " dB");
      });
    }

    var PLAN = {
      series: [ { name: "Uniform", color: "--c-uniform" }, { name: "DreamDojo", color: "--c-dd" }, { name: "DreamDojo + AFFECT", color: "--c-ours" } ],
      panels: [
        { title: "(a) UR5e pick-and-place", cats: ["Vertical2", "Horizontal", "Diagonal1", "Diagonal2", "Lateral Near", "Average"],
          data: [[33.3, 50, 33.3, 50, 66.7, 46.7], [25, 83.3, 58.3, 58.3, 75, 60], [33.3, 83.3, 66.7, 75, 83.3, 68.3]] },
        { title: "(b) ALOHA-2 long-horizon", cats: ["Subtask 1", "Subtask 2", "Subtask 3", "Average"],
          data: [[83.3, 58.3, 25, 55.6], [91.7, 66.7, 58.3, 72.2], [91.7, 83.3, 66.7, 80.6]] },
      ],
    };
    function drawPlanning(root) {
      var W = root.clientWidth;
      root.textContent = "";
      legend(root, PLAN.series.map(function (s) { return { label: s.name, color: s.color }; }));
      var tip = tooltip(root);
      var wrap = document.createElement("div");
      wrap.className = "viz-panels";
      root.appendChild(wrap);
      var side = W >= 640;
      wrap.style.flexDirection = side ? "row" : "column";
      var total = PLAN.panels.reduce(function (a, p) { return a + p.cats.length; }, 0);
      PLAN.panels.forEach(function (p, pi) {
        var box = document.createElement("div");
        wrap.appendChild(box);
        var axisL = 30;
        var pw = side ? Math.floor((W - 24) * (p.cats.length / total)) + (pi === 0 ? axisL / 2 : -axisL / 2) : W;
        box.style.width = pw + "px";
        var H = 250, top = 34, bottom = 36, y0 = H - bottom;
        var sy = function (v) { return y0 - (v / 100) * (y0 - top); };
        var svg = el("svg", { width: pw, height: H }, box);
        txt(svg, axisL, 10, p.title, "t-strong");
        [0, 25, 50, 75, 100].forEach(function (t) {
          el("line", { x1: axisL, x2: pw, y1: sy(t), y2: sy(t), class: t === 0 ? "baseline" : "gridline" }, svg);
          txt(svg, axisL - 6, sy(t), String(t), "t-2", "end");
        });
        var slot = (pw - axisL) / p.cats.length;
        var bw = Math.max(6, Math.min(20, (slot * 0.72 - 4) / 3));
        p.cats.forEach(function (c, ci) {
          var cx = axisL + slot * ci + slot / 2, gx = cx - (3 * bw + 4) / 2, avg = c === "Average";
          var hit = el("rect", { x: axisL + slot * ci, y: top - 6, width: slot, height: y0 - top + 6 + bottom, rx: 4, class: "hit" }, svg);
          if (avg) el("line", { x1: axisL + slot * ci, x2: axisL + slot * ci, y1: top, y2: y0, class: "gridline" }, svg);
          PLAN.series.forEach(function (s, si) {
            var v = p.data[si][ci], x = gx + si * (bw + 2);
            var b = el("path", { d: vbar(x, bw, y0, sy(v), 4), fill: "var(" + s.color + ")", class: "grow-y" }, svg);
            b.style.transitionDelay = 0.1 + ci * 0.1 + si * 0.08 + "s";
            if (si === 2 && (avg || bw >= 14)) {
              var lab = txt(svg, x + bw / 2, sy(v) - 9, v.toFixed(1), avg ? "t-strong fade" : "t-ink fade", "middle");
              lab.style.fontSize = avg ? "12px" : "10px";
              lab.style.transitionDelay = 0.8 + ci * 0.1 + "s";
            }
          });
          var words = c.split(" ");
          var lines = words.length > 1 && c.length * 6.5 > slot - 6 ? words : [c];
          lines.forEach(function (w, li) { txt(svg, cx, y0 + 14 + li * 13, w, avg ? "t-strong" : "t-2", "middle"); });
          bindTip(root, tip, hit, p.title.replace(/^\(.\) /, "") + " · " + c,
            PLAN.series.map(function (s, si) { return { value: p.data[si][ci].toFixed(1) + "%", label: s.name, color: s.color }; }).reverse());
          hit.setAttribute("aria-label", c + ": " + PLAN.series.map(function (s, si) { return s.name + " " + p.data[si][ci] + "%"; }).join(", "));
        });
      });
    }

    var charts = [ ["viz-pearson", drawPearson], ["viz-psnr", drawPSNR], ["viz-planning", drawPlanning] ];
    var io = "IntersectionObserver" in window && !reduce ? new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.35 }) : null;
    charts.forEach(function (c) {
      var root = document.getElementById(c[0]);
      if (!root) return;
      var lastW = 0;
      function render() { if (root.clientWidth !== lastW) { lastW = root.clientWidth; c[1](root); } }
      render();
      if (io) io.observe(root); else root.classList.add("in");
      var t;
      window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(render, 150); });
    });
  })();
</script>
