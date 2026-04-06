# =============================================================================
# LAYER 1 — neuron as a state machine with real phase durations
# =============================================================================
# Question: what actually happens during a spike, and how long does it take?
# Answer: 6 distinct phases, each with a real duration in milliseconds.
#
# Biological grounding:
#   - dt = 0.1ms: time resolution. each function call = 0.1ms of real time.
#     the loop is the clock. the neuron does not control time.
#   - depolarizing (0.5ms): Na+ channels burst open. sodium floods in.
#     voltage rockets from -55 toward +40mV. driven by electrochemical gradient.
#   - peak (0.2ms): voltage at +40mV. Na+ channels begin inactivating.
#     this inactivation is the physical basis of the absolute refractory period.
#   - repolarizing (1.0ms): K+ channels open. potassium rushes out.
#     voltage falls rapidly back toward -70mV.
#   - hyperpolarizing (1.5ms): K+ channels slow to close. voltage overshoots
#     below -70mV to about -80mV. this is the relative refractory period —
#     the neuron CAN fire but needs a stronger-than-normal input.
#     (correction from layer 0 thinking: hyperpolarization = relative refractory,
#      NOT absolute. absolute refractory is depolarizing+peak+repolarizing.)
#   - recovering (2.0ms): K+ channels finally close. Na+/K+ pump restores balance.
#     voltage drifts back to -70mV. still relative refractory but easier than hyperpol.
#
# What this layer adds:
#   - spike has a shape you can plot (the classic oscilloscope waveform)
#   - absolute refractory period: physically cannot fire during depol/peak/repol
#   - relative refractory period: can fire during hyperpol/recovering but needs more input
#   - time is real — a spike takes ~3.4ms total
#
# What this layer cannot model:
#   - multiple inputs arriving simultaneously from different sources
#   - excitatory vs inhibitory inputs
#   - one neuron's spike triggering another neuron
#
# Why we needed to go further:
#   layer 1 still receives input as a single value at one moment.
#   real neurons receive thousands of inputs constantly, from many sources,
#   some pushing voltage up (excitatory), some pushing it down (inhibitory).
#   the neuron's job is to integrate all of them and decide whether to fire.
#   layer 1 cannot model this integration.

dt    = 0.1   # ms — time resolution. smaller = more accurate, slower to run.
TOTAL = 20    # ms — how long to simulate

def neuron_l1(current_input, vm, phase, timer):
    threshold = -55
    rmp       = -70
    decay     = 0.9
    spike     = 0

    if phase == "resting":
        vm = rmp + (vm - rmp) * decay + current_input
        timer += dt
        if vm >= threshold:
            phase = "depolarizing"
            timer = 0
            spike = 1   # spike registered at threshold crossing

    elif phase == "depolarizing":
        # input ignored — absolute refractory
        # 0.7 factor: exponential approach to +40. fast at first, slows near peak.
        vm += (40 - vm) * 0.7
        timer += dt
        if timer >= 0.5:
            phase = "peak"
            timer = 0

    elif phase == "peak":
        # input ignored — absolute refractory
        vm = 40
        timer += dt
        if timer >= 0.2:
            phase = "repolarizing"
            timer = 0

    elif phase == "repolarizing":
        # input ignored — absolute refractory
        # 0.35 factor: exponential fall toward rmp
        vm += (-70 - vm) * 0.35
        timer += dt
        if timer >= 1.0:
            phase = "hyperpolarizing"
            timer = 0

    elif phase == "hyperpolarizing":
        # input ignored here in layer 1 — corrected in layer 2
        # biologically this IS relative refractory (neuron can fire with strong input)
        # but layer 1 keeps it simple: held at -80, no input accepted
        vm = -80
        timer += dt
        if timer >= 1.5:
            phase = "recovering"
            timer = 0

    elif phase == "recovering":
        # 0.08 factor: slow exponential drift back to rmp
        vm += (-70 - vm) * 0.08
        timer += dt
        if timer >= 2.0:
            phase = "resting"
            timer = 0

    return spike, vm, phase, timer


# --- layer 1 simulation loop ---
vm    = -70.0
phase = "resting"
timer = 0.0
time  = 0.0
t_history, v_history = [], []
spikes = 0

INPUT_STRENGTH = 16    # mV — try values below 15 to see subthreshold behavior
INPUT_AT       = 2.0   # ms — when to deliver the input

steps      = round(TOTAL / dt)
input_step = round(INPUT_AT / dt)

for i in range(steps):
    inp = INPUT_STRENGTH if i == input_step else 0
    spike, vm, phase, timer = neuron_l1(inp, vm, phase, timer)
    if spike:
        spikes += 1
    t_history.append(time)
    v_history.append(vm)
    time += dt

print(f"Total spikes: {spikes}")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(t_history, v_history, color='steelblue', linewidth=1.5)
plt.axhline(-55, color='gray',      linestyle='--', linewidth=0.8, label='threshold')
plt.axhline(-70, color='lightgray', linestyle='--', linewidth=0.8, label='resting')
plt.xlabel('time (ms)')
plt.ylabel('membrane potential (mV)')
plt.legend()
plt.tight_layout()
plt.show()
