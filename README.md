# computational-neuro-from-scratch
Building neuroscience understanding from first principles —  single neuron simulation to brain-level concepts.

# neuro-from-scratch

I'm a computer engineer teaching myself neuroscience. 
This repo documents that process — as a series of simulations built from the ground up.

The philosophy: simulate first, understand second. 
Confusion about equations disappears when you can run 
them and watch what changes.

## Structure

Each layer adds one biological concept the previous layer couldn't model.

| Layer | Model | What it adds |
|-------|-------|-------------|
| 0 | Threshold gate | input, accumulation, fire, reset |
| 1 | State machine | spike phases, real timing, refractory period |
| 2 | Integrate & Fire (class) | summation, EPSP/IPSP, noise, modularity |
| 3 | Hodgkin-Huxley | conductances, gating variables, no phases needed |

## Why each layer exists

**Layer 0 → 1:** Layer 0 treats the spike as instantaneous. 
In reality a spike takes ~3ms and during that time the neuron 
physically cannot fire again. You need time to model this.

**Layer 1 → 2:** Layer 1 receives one input at one moment. 
Real neurons receive thousands of simultaneous inputs, 
some excitatory, some inhibitory. The neuron's job is 
to integrate all of them.

**Layer 2 → 3:** Layer 2 hardcodes spike behavior into 
phase logic. The phases are labels on top of conductances 
we weren't tracking. HH tracks the conductances directly — 
spike shape, refractory period, hyperpolarization all emerge 
from four coupled differential equations with no phase 
variable anywhere.



- EEG / fMRI — measuring population activity from outside the skull
- Brain regions and topology
- Neural circuits
