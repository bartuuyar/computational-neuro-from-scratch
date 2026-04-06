# =============================================================================
# LAYER 2 — multiple inputs, summation, excitatory/inhibitory, noise
# =============================================================================
# Question: how does a neuron handle thousands of inputs from different sources?
# Answer: it sums them all each time step. excitatory inputs add, inhibitory subtract.
#
# Biological grounding:
#   - spatial summation: inputs from many dendrites arrive simultaneously.
#     the neuron adds them all. two weak inputs can together cross threshold
#     even if neither could alone. modeled by summing all inputs each dt step.
#   - EPSP (excitatory postsynaptic potential): input pushes vm UP toward threshold.
#     positive strength value.
#   - IPSP (inhibitory postsynaptic potential): input pushes vm DOWN away from threshold.
#     negative strength value. one neuron can actively suppress another.
#   - relative refractory (corrected from layer 1):
#     hyperpolarizing phase = relative refractory, not absolute.
#     the neuron CAN fire during hyperpolarization with strong enough input.
#     absolute refractory = depolarizing + peak + repolarizing only.
#   - noise: real ion channels open/close probabilistically.
#     modeled as gaussian noise added to vm each step.
#     effect: near-threshold inputs sometimes fire, sometimes don't.
#     makes behavior realistic rather than perfectly deterministic.
#
# Key design changes from layer 1:
#   - neuron is now a CLASS: state lives on the object.
#     this means you can do n1 = Neuron(), n2 = Neuron() and run them independently.
#     essential for layer 3 where neurons talk to each other.
#   - simulate() and plot() are standalone functions, not methods.
#     neuron owns its state. loop owns time. plot owns visualization.
#     each piece does exactly one job.
#   - all magic numbers are constructor parameters.
#     Neuron(threshold=-50) gives a more excitable neuron.
#     Neuron(dur_hyperpolarizing=3.0) gives a longer refractory period.
#   - inputs are lists, not single values.
#     multiple inputs at the same time step are summed, not overwritten.
#
# What this layer adds:
#   - true summation of simultaneous inputs
#   - excitatory and inhibitory inputs (positive/negative strengths)
#   - correct absolute/relative refractory periods
#   - optional biological noise
#   - class structure ready for multi-neuron networks
#
# What this layer cannot model:
#   - synaptic chemistry (neurotransmitter release, receptor binding)
#   - spike propagation along the axon
#   - neurons connected to each other 
#   - learning / synaptic weight changes 

import matplotlib.pyplot as plt
import random

class Neuron:
    def __init__(
        self,
        threshold=-55,
        rmp=-70,
        decay=0.9,
        dt=0.1,
        total_time=40,
        noise_std=0.0,
        dur_depolarizing=0.5,
        dur_peak=0.2,
        dur_repolarizing=1.0,
        dur_hyperpolarizing=1.5,
        dur_recovering=2.0,
        v_peak=40,
        v_hyperpolar=-80,
    ):
        self.threshold        = threshold
        self.rmp              = rmp
        self.decay            = decay
        self.dt               = dt
        self.total_time       = total_time
        self.noise_std        = noise_std
        self.dur_depolarizing    = dur_depolarizing
        self.dur_peak            = dur_peak
        self.dur_repolarizing    = dur_repolarizing
        self.dur_hyperpolarizing = dur_hyperpolarizing
        self.dur_recovering      = dur_recovering
        self.v_peak           = v_peak
        self.v_hyperpolar     = v_hyperpolar

        self.vm            = rmp
        self.phase         = "resting"
        self.timer         = 0.0
        self.spike         = 0
        self.current_input = 0.0

    def step(self):
        self.spike = 0
        noise = random.gauss(0, self.noise_std) if self.noise_std > 0 else 0

        if self.phase == "resting":
            self.vm = self.rmp + (self.vm - self.rmp) * self.decay + self.current_input + noise
            self.timer += self.dt
            if self.vm >= self.threshold:
                self.phase = "depolarizing"
                self.timer = 0
                self.spike = 1

        elif self.phase == "depolarizing":
            # absolute refractory — input ignored
            self.vm += (self.v_peak - self.vm) * 0.7
            self.timer += self.dt
            if self.timer >= self.dur_depolarizing:
                self.phase = "peak"
                self.timer = 0

        elif self.phase == "peak":
            # absolute refractory — input ignored
            self.vm = self.v_peak
            self.timer += self.dt
            if self.timer >= self.dur_peak:
                self.phase = "repolarizing"
                self.timer = 0

        elif self.phase == "repolarizing":
            # absolute refractory — input ignored
            self.vm += (self.rmp - self.vm) * 0.35
            self.timer += self.dt
            if self.timer >= self.dur_repolarizing:
                self.phase = "hyperpolarizing"
                self.timer = 0

        elif self.phase == "hyperpolarizing":
            # relative refractory — input accepted, spike possible with strong input
            self.vm += (self.v_hyperpolar - self.vm) * 0.2 + self.current_input + noise
            self.timer += self.dt
            if self.vm >= self.threshold:
                self.phase = "depolarizing"
                self.timer = 0
                self.spike = 1
            elif self.timer >= self.dur_hyperpolarizing:
                self.phase = "recovering"
                self.timer = 0

        elif self.phase == "recovering":
            # relative refractory — easier to fire than hyperpolarizing
            self.vm += (self.rmp - self.vm) * 0.08 + self.current_input + noise
            self.timer += self.dt
            if self.vm >= self.threshold:
                self.phase = "depolarizing"
                self.timer = 0
                self.spike = 1
            elif self.timer >= self.dur_recovering:
                self.phase = "resting"
                self.timer = 0


def simulate(neuron, input_strengths, input_times):
    t_history, v_history, phase_history = [], [], []
    total_spikes = 0
    current_time = 0.0
    steps = int(neuron.total_time / neuron.dt)

    for _ in range(steps):
        # sum all inputs arriving at this time step
        # positive strength = excitatory (EPSP), negative = inhibitory (IPSP)
        neuron.current_input = sum(
            strength
            for strength, t_trigger in zip(input_strengths, input_times)
            if abs(current_time - t_trigger) < (neuron.dt / 2)
        )

        neuron.step()
        if neuron.spike:
            total_spikes += 1

        t_history.append(current_time)
        v_history.append(neuron.vm)
        phase_history.append(neuron.phase)
        current_time += neuron.dt

    return t_history, v_history, phase_history, total_spikes


def plot(t, v, neuron, total_spikes):
    plt.figure(figsize=(10, 4))
    plt.plot(t, v, color="steelblue", linewidth=1.5, label="membrane potential")
    plt.axhline(neuron.threshold, color="red",  linestyle="--", alpha=0.5, linewidth=0.8, label="threshold")
    plt.axhline(neuron.rmp,       color="gray", linestyle="--", alpha=0.4, linewidth=0.8, label="resting")
    plt.xlabel("time (ms)")
    plt.ylabel("mV")
    plt.title(f"action potential simulation — {total_spikes} spike(s)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- layer 2 experiments ---

# experiment 1: two simultaneous excitatory inputs that sum to cross threshold
# neither 16 nor 15 alone would fire the neuron — together they do
n = Neuron(total_time=40, noise_std=0.0)
t, v, phases, spikes = simulate(
    n,
    input_strengths=[16, 15, 45],
    input_times=[2.0, 2.0, 12.0]   # 16+15=31 mV arrive together at 2ms
)
plot(t, v, n, spikes)

# experiment 2: inhibitory input prevents firing

n2 = Neuron(total_time=40)
t2, v2, phases2, spikes2 = simulate(
    n2,
    input_strengths=[45, -35],
    input_times=[12.0, 12.0]        
)
plot(t2, v2, n2, spikes2)

# experiment 3: noise makes near-threshold behavior probabilistic
# run this multiple times — sometimes fires, sometimes doesn't
n3 = Neuron(total_time=40, noise_std=1.5)
t3, v3, phases3, spikes3 = simulate(
    n3,
    input_strengths=[14],           # just below threshold without noise
    input_times=[2.0]
)
plot(t3, v3, n3, spikes3)
print(f"spikes: {spikes3}")
