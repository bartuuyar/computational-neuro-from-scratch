
# Goal: build a single neuron from scratch, layer by layer.
# Each layer adds one real biological concept that the previous layer couldn't model.


# =============================================================================
# LAYER 0 — neuron as a black box threshold gate
# =============================================================================
# Question: what is the absolute minimum a neuron needs to do?
# Answer: receive input, accumulate charge, fire if enough arrives, reset.
#
# Biological grounding:
#   - rmp = -70 mV: resting membrane potential. the neuron's "off" state.
#     maintained by the Na+/K+ pump constantly pushing ions across the membrane.
#   - threshold = -55 mV: the point of no return. if voltage crosses this,
#     the neuron fires no matter what (all-or-nothing law).
#   - decay = 0.9: membrane leaks charge back toward resting between inputs.
#     if decay=1, charge accumulates forever (unrealistic).
#     if decay=0, voltage snaps to rmp instantly after each step (also unrealistic).
#     0.9 means ~10% of accumulated charge leaks away each time step.
#     test it: send input=0 repeatedly and watch voltage drift back to -70.
#
# What this layer gets right:
#   - subthreshold inputs accumulate (summation)
#   - crossing threshold triggers a spike
#   - neuron resets after firing
#
# What this layer cannot model:
#   - HOW the spike unfolds over time (it's instantaneous here)
#   - the refractory period (neuron can fire again immediately after reset)
#   - the actual voltage shape seen on an oscilloscope
#   - why a neuron physically CANNOT fire twice in 1ms
#
# Why we needed to go further:
#   layer 0 treats the spike as a single event with no duration.
#   in reality a spike takes ~3ms to complete and during that time
#   the neuron is physically incapable of firing again.
#   without modeling this, a strong enough input fires the neuron
#   every single time step — biologically impossible.

def neuron_l0(current_input, membrane_potential):
    threshold = -55
    rmp       = -70
    decay     = 0.9

    # decay pulls voltage back toward rmp, then input pushes it up
    # rmp + (vm - rmp)*decay: if vm==rmp this equals rmp (no drift)
    #                         if vm > rmp this slowly returns toward rmp
    membrane_potential = rmp + (membrane_potential - rmp) * decay + current_input

    if membrane_potential >= threshold:
        membrane_potential = rmp  # reset — no model of HOW it resets, just does
        return 1, membrane_potential
    else:
        return 0, membrane_potential


# --- layer 0 demo ---
current_v = -70.0
spike, current_v = neuron_l0(10, current_v)   # subthreshold — accumulates
print(spike,current_v)
spike, current_v = neuron_l0(0,  current_v)   # no input — watch decay pull it back
print(spike,current_v)
spike, current_v = neuron_l0(30,  current_v)   #fire
print(spike,current_v)
